# 多智能体能力三阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 HaAgent 按 profile 稳定派出不同用途的 worker，并逐步补齐 worker 通知、运行中通信、权限请求和后续隔离能力。

**Architecture:** 先在 `haagent.multi_agent` 内增加 profile 解析层，让 `MultiAgentRuntime` 从明确配置创建 worker。再把 worker 通知、消息和权限请求变成结构化数据，继续复用 `TeamStore`、`ToolRouter`、`AgentSession` 和 episode trace。最后通过 backend 接口为 subprocess/worktree 做隔离扩展点，但不在前两阶段改变默认执行模型。

**Tech Stack:** Python 3.11+、dataclasses、pytest、现有 `ModelGateway`、`ToolRouter`、`AgentSession`、`TeamStore`、Textual TUI 事件流。

## Global Constraints

- HaAgent 仍然是 TUI-first 的本地个人助手，不变成复杂团队编排平台。
- 普通用户不需要理解“多智能体框架”才能获益。
- 不增加模型输入 token 的常驻负担；只有派 worker 时加载精简 profile 摘要。
- 所有模型调用继续走 `ModelGateway`。
- 所有工具调用继续走 `ToolRouter`。
- worker 的文件修改和命令执行仍受 workspace root、path policy、approval policy 约束。
- 第一阶段不引入 subprocess、tmux、远程 agent、浏览器自动化或大型 swarm UI。
- 不为历史 `.runs` 或旧 team 文件增加兼容层。
- 本计划不要求 Git commit；项目 AGENTS 默认只允许只读 Git 操作，除非用户明确要求提交。

---

## File Structure

- Create: `src/haagent/multi_agent/profiles.py`
  - 负责 agent profile 数据结构、内置 profile、用户 profile 加载、字段校验。
- Modify: `src/haagent/multi_agent/runtime.py`
  - 在 `spawn_worker` 中解析 profile、加载指定 `model_profile`、创建 worker session。
- Modify: `src/haagent/multi_agent/permissions.py`
  - 保持现有 `worker_tool_policy`，只增加 profile 覆盖工具列表时的合并入口。
- Modify: `src/haagent/tools/registry.py`
  - 扩展 `agent` 工具 schema，使 `profile` 成为稳定字段，保留 `subagent_type` 兼容当前调用。
- Create: `src/haagent/multi_agent/messages.py`
  - 定义 `WorkerNotification`、`WorkerMessage`、`WorkerPermissionRequest`。
- Modify: `src/haagent/multi_agent/team_store.py`
  - 增加结构化消息和权限请求的持久化方法。
- Modify: `src/haagent/runtime/orchestration/orchestrator.py`
  - 读取结构化 worker 通知并形成精简上下文。
- Modify: `src/haagent/runtime/session/turn.py`
  - 确保 chat task 允许新增多智能体工具字段，不引入新的普通用户入口。
- Create: `src/haagent/multi_agent/backends.py`
  - 定义第三阶段 backend 接口，先适配现有 in-process worker。
- Create: `src/haagent/multi_agent/worktree.py`
  - 第三阶段 Git worktree 管理入口，第一版只做路径和 slug 校验。
- Test: `tests/unit/multi_agent/test_profiles.py`
- Test: `tests/integration/multi_agent/test_agent_profiles.py`
- Test: `tests/unit/multi_agent/test_messages.py`
- Test: `tests/unit/multi_agent/test_team_store_messages.py`
- Test: `tests/integration/multi_agent/test_worker_messaging.py`
- Test: `tests/unit/multi_agent/test_backends.py`
- Test: `tests/unit/multi_agent/test_worktree.py`

---

### Task 1: Agent Profile 数据结构和内置 profile

**Files:**
- Create: `src/haagent/multi_agent/profiles.py`
- Test: `tests/unit/multi_agent/test_profiles.py`

**Interfaces:**
- Produces: `AgentProfile`
- Produces: `load_builtin_agent_profiles() -> dict[str, AgentProfile]`
- Produces: `get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile`

- [ ] **Step 1: Write the failing tests**

```python
"""
tests/unit/multi_agent/test_profiles.py - 多智能体 profile 加载测试

验证内置 agent profile 的稳定字段和显式失败行为。
"""

import pytest

from haagent.multi_agent.profiles import AgentProfile, get_agent_profile, load_builtin_agent_profiles


def test_load_builtin_agent_profiles_has_core_roles() -> None:
    profiles = load_builtin_agent_profiles()

    assert set(profiles) == {"explorer", "worker", "verification"}
    assert profiles["explorer"].subagent_type == "explorer"
    assert profiles["worker"].subagent_type == "worker"
    assert profiles["verification"].subagent_type == "verification"


def test_get_agent_profile_returns_builtin_profile() -> None:
    profile = get_agent_profile("explorer")

    assert isinstance(profile, AgentProfile)
    assert profile.name == "explorer"
    assert profile.allowed_tools == ["file_list", "file_search", "file_read", "skill_list", "skill_read"]


def test_get_agent_profile_unknown_name_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown agent profile: missing"):
        get_agent_profile("missing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_profiles.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'haagent.multi_agent.profiles'`.

- [ ] **Step 3: Write minimal implementation**

```python
"""
haagent/multi_agent/profiles.py - 多智能体 profile 定义与加载

定义 worker 角色配置，供 MultiAgentRuntime 用明确配置创建后台帮手。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentProfile:
    name: str
    description: str
    subagent_type: str
    system_prompt: str
    allowed_tools: list[str] | None = None
    approval_allowed_tools: list[str] | None = None
    approved_tools: list[str] | None = None
    model_profile: str | None = None
    max_turns: int | None = None
    enable_web: bool | None = None


def load_builtin_agent_profiles() -> dict[str, AgentProfile]:
    return {
        "explorer": AgentProfile(
            name="explorer",
            description="只读探索文件、资料和项目结构。",
            subagent_type="explorer",
            system_prompt="你是只读探索助手。读取资料、总结发现，不修改文件。",
            allowed_tools=["file_list", "file_search", "file_read", "skill_list", "skill_read"],
        ),
        "worker": AgentProfile(
            name="worker",
            description="按主助手授权执行普通任务。",
            subagent_type="worker",
            system_prompt="你是执行助手。按任务要求完成工作，并清楚汇报结果。",
        ),
        "verification": AgentProfile(
            name="verification",
            description="运行验证、读取结果并指出风险。",
            subagent_type="verification",
            system_prompt="你是验证助手。运行检查、解释结果，不做无关修改。",
            allowed_tools=["file_read", "file_search", "shell", "code_run"],
        ),
    }


def get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile:
    del config_dir
    profiles = load_builtin_agent_profiles()
    if name not in profiles:
        raise ValueError(f"unknown agent profile: {name}")
    return profiles[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/multi_agent/test_profiles.py -q`

Expected: PASS.

---

### Task 2: 用户自定义 agent profile 文件加载

**Files:**
- Modify: `src/haagent/multi_agent/profiles.py`
- Test: `tests/unit/multi_agent/test_profiles.py`

**Interfaces:**
- Consumes: `AgentProfile`
- Produces: `load_user_agent_profiles(config_dir: Path) -> dict[str, AgentProfile]`
- Produces: `get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_user_agent_profiles_reads_json_file(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "doc-editor.json").write_text(
        """
        {
          "name": "doc-editor",
          "description": "润色文档草稿。",
          "subagent_type": "worker",
          "system_prompt": "你是文档编辑助手。",
          "allowed_tools": ["file_read", "file_write"],
          "model_profile": "long-context",
          "max_turns": 8,
          "enable_web": false
        }
        """,
        encoding="utf-8",
    )

    profiles = load_user_agent_profiles(tmp_path)

    assert profiles["doc-editor"].model_profile == "long-context"
    assert profiles["doc-editor"].allowed_tools == ["file_read", "file_write"]
    assert profiles["doc-editor"].max_turns == 8
    assert profiles["doc-editor"].enable_web is False


def test_user_agent_profile_overrides_builtin_by_name(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "explorer.json").write_text(
        """
        {
          "name": "explorer",
          "description": "项目索引助手。",
          "subagent_type": "explorer",
          "system_prompt": "只读取项目结构。",
          "allowed_tools": ["file_list"]
        }
        """,
        encoding="utf-8",
    )

    profile = get_agent_profile("explorer", config_dir=tmp_path)

    assert profile.description == "项目索引助手。"
    assert profile.allowed_tools == ["file_list"]


def test_user_agent_profile_invalid_subagent_type_fails(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "bad.json").write_text(
        """
        {
          "name": "bad",
          "description": "bad",
          "subagent_type": "admin",
          "system_prompt": "bad"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid subagent_type"):
        load_user_agent_profiles(tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_profiles.py -q`

Expected: FAIL because `load_user_agent_profiles` is not defined.

- [ ] **Step 3: Write minimal implementation**

Add imports and functions to `src/haagent/multi_agent/profiles.py`:

```python
import json
from typing import Any

VALID_SUBAGENT_TYPES = {"explorer", "worker", "verification"}


def load_user_agent_profiles(config_dir: Path) -> dict[str, AgentProfile]:
    agents_dir = config_dir / "agents"
    if not agents_dir.exists():
        return {}
    profiles: dict[str, AgentProfile] = {}
    for path in sorted(agents_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        profile = _profile_from_dict(raw, source=str(path))
        profiles[profile.name] = profile
    return profiles


def _profile_from_dict(raw: dict[str, Any], *, source: str) -> AgentProfile:
    name = _required_string(raw, "name", source=source)
    description = _required_string(raw, "description", source=source)
    subagent_type = _required_string(raw, "subagent_type", source=source)
    if subagent_type not in VALID_SUBAGENT_TYPES:
        raise ValueError(f"invalid subagent_type in {source}: {subagent_type}")
    system_prompt = _required_string(raw, "system_prompt", source=source)
    return AgentProfile(
        name=name,
        description=description,
        subagent_type=subagent_type,
        system_prompt=system_prompt,
        allowed_tools=_optional_string_list(raw, "allowed_tools", source=source),
        approval_allowed_tools=_optional_string_list(raw, "approval_allowed_tools", source=source),
        approved_tools=_optional_string_list(raw, "approved_tools", source=source),
        model_profile=_optional_string(raw, "model_profile", source=source),
        max_turns=_optional_positive_int(raw, "max_turns", source=source),
        enable_web=_optional_bool(raw, "enable_web", source=source),
    )


def _required_string(raw: dict[str, Any], field_name: str, *, source: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required string {field_name} in {source}")
    return value


def _optional_string(raw: dict[str, Any], field_name: str, *, source: str) -> str | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid string {field_name} in {source}")
    return value


def _optional_string_list(raw: dict[str, Any], field_name: str, *, source: str) -> list[str] | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"invalid string list {field_name} in {source}")
    return list(value)


def _optional_positive_int(raw: dict[str, Any], field_name: str, *, source: str) -> int | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid positive integer {field_name} in {source}")
    return value


def _optional_bool(raw: dict[str, Any], field_name: str, *, source: str) -> bool | None:
    value = raw.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"invalid boolean {field_name} in {source}")
    return value
```

Update `get_agent_profile`:

```python
def get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile:
    profiles = load_builtin_agent_profiles()
    if config_dir is not None:
        profiles.update(load_user_agent_profiles(config_dir))
    if name not in profiles:
        raise ValueError(f"unknown agent profile: {name}")
    return profiles[name]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/multi_agent/test_profiles.py -q`

Expected: PASS.

---

### Task 3: agent 工具使用 profile 创建 worker

**Files:**
- Modify: `src/haagent/tools/registry.py`
- Modify: `src/haagent/tools/router.py`
- Modify: `src/haagent/multi_agent/runtime.py`
- Test: `tests/integration/multi_agent/test_agent_profiles.py`

**Interfaces:**
- Consumes: `get_agent_profile(name: str, *, config_dir: Path | None = None) -> AgentProfile`
- Produces: `MultiAgentRuntime.spawn_worker(..., profile: str | None = None, ...) -> dict[str, Any]`

- [ ] **Step 1: Write the failing test**

```python
"""
tests/integration/multi_agent/test_agent_profiles.py - worker profile 集成测试

验证 agent 工具可以通过 profile 派出 worker。
"""

from pathlib import Path

from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import ModelResponse
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


def test_spawn_worker_accepts_profile_name(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read", "file_list"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
    )

    result = runtime.spawn_worker(
        description="Inspect files",
        prompt="Say done",
        subagent_type="worker",
        profile="explorer",
        team_id="team-test",
    )

    assert result["status"] == "running"
    assert result["profile"] == "explorer"
    assert runtime.wait_for_task(result["task_id"], timeout=5)["status"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/multi_agent/test_agent_profiles.py -q`

Expected: FAIL with `TypeError` because `spawn_worker` has no `profile` parameter.

- [ ] **Step 3: Extend tool schema**

In `src/haagent/tools/registry.py`, add `profile` to the `agent` tool properties:

```python
"profile": {
    "type": "string",
    "description": "agent profile name; defaults to subagent_type when omitted",
},
```

Keep existing `subagent_type` required for now. This avoids changing current model/tool contracts in the same task.

- [ ] **Step 4: Route profile from ToolRouter**

In `ToolRouter._agent`, pass `profile=args.get("profile")`:

```python
return _agent_runtime_result(
    self._agent_runtime.spawn_worker(
        description=str(args["description"]),
        prompt=str(args["prompt"]),
        subagent_type=args["subagent_type"],
        team_id=args.get("team"),
        model_profile=args.get("model_profile"),
        profile=args.get("profile"),
    ),
)
```

- [ ] **Step 5: Apply profile in MultiAgentRuntime**

Update `MultiAgentRuntime.spawn_worker`:

```python
from haagent.multi_agent.profiles import get_agent_profile


def spawn_worker(
    self,
    *,
    description: str,
    prompt: str,
    subagent_type: WorkerType,
    team_id: str | None = None,
    model_profile: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    agent_profile = get_agent_profile(profile or subagent_type)
    resolved_subagent_type = agent_profile.subagent_type
    resolved_model_profile = model_profile or agent_profile.model_profile
    if resolved_model_profile:
        return {
            "is_error": True,
            "error": "model_profile override is not implemented for in-process workers in v1",
        }
    ...
    agent_id = f"{resolved_subagent_type}-{uuid.uuid4().hex[:8]}"
    session = self._create_worker_session(agent_id=agent_id, subagent_type=resolved_subagent_type)
    ...
    record = WorkerRecord(
        agent_id=agent_id,
        task_id=task_id,
        subagent_type=resolved_subagent_type,
        description=description,
        status="running",
        session_id=session.session_id,
    )
    ...
    return {
        "agent_id": agent_id,
        "task_id": task_id,
        "team_id": team.team_id,
        "status": "running",
        "profile": agent_profile.name,
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/integration/multi_agent/test_agent_profiles.py -q`

Expected: PASS.

---

### Task 4: model_profile 覆盖创建 worker gateway

**Files:**
- Modify: `src/haagent/multi_agent/runtime.py`
- Modify: `src/haagent/runtime/orchestration/orchestrator.py`
- Test: `tests/integration/multi_agent/test_agent_profiles.py`

**Interfaces:**
- Consumes: `load_provider_profile(profile_name, environ=..., config_dir=...)`
- Consumes: `gateway_from_profile(profile)`
- Produces: worker session can use profile-specific `ModelGateway`

- [ ] **Step 1: Write the failing test**

```python
def test_spawn_worker_rejects_unknown_model_profile(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read", "file_list"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
    )

    result = runtime.spawn_worker(
        description="Inspect",
        prompt="Say done",
        subagent_type="worker",
        model_profile="missing-profile",
    )

    assert result["is_error"] is True
    assert "missing-profile" in result["error"]
```

- [ ] **Step 2: Run test to verify current behavior fails for the right reason**

Run: `uv run pytest tests/integration/multi_agent/test_agent_profiles.py::test_spawn_worker_rejects_unknown_model_profile -q`

Expected: FAIL because current error is the generic “not implemented” message.

- [ ] **Step 3: Add gateway factory dependency**

In `MultiAgentRuntime.__init__`, add optional dependencies:

```python
from collections.abc import Callable, Mapping
from haagent.models.gateway_registry import gateway_from_profile
from haagent.models.provider_profile import ProviderProfile, ProviderProfileError, load_provider_profile

GatewayFactory = Callable[[ProviderProfile], ModelGateway]


def __init__(
    ...,
    environ: Mapping[str, str] | None = None,
    gateway_factory: GatewayFactory = gateway_from_profile,
) -> None:
    ...
    self.environ = environ
    self.gateway_factory = gateway_factory
```

In `RunOrchestrator.run`, pass the process environment only if the surrounding runtime already has one available. If not available, leave `environ=None` and rely on existing provider profile loading behavior.

- [ ] **Step 4: Resolve worker gateway explicitly**

Add helper in `MultiAgentRuntime`:

```python
def _worker_gateway(self, model_profile: str | None) -> ModelGateway:
    if model_profile is None:
        return self.model_gateway
    try:
        profile = load_provider_profile(
            model_profile,
            environ=self.environ,
            config_dir=user_config_dir(),
        )
    except ProviderProfileError as error:
        raise ValueError(str(error)) from error
    return self.gateway_factory(profile)
```

Use it in `_create_worker_session`:

```python
def _create_worker_session(
    self,
    *,
    agent_id: str,
    subagent_type: WorkerType,
    restart_count: int = 0,
    model_profile: str | None = None,
) -> Any:
    ...
    return AgentSession(
        ...
        model_gateway=self._worker_gateway(model_profile),
        model_profile_name=model_profile,
        ...
    )
```

Pass `model_profile=resolved_model_profile` from `spawn_worker` and from restarted workers in `send_message` after storing the resolved profile in `WorkerRecord` in a later task. Until `WorkerRecord` has a field, restart should use the original record status and inherited gateway; this limitation is removed in Task 5.

- [ ] **Step 5: Convert profile load errors into tool errors**

In `spawn_worker`, wrap `_create_worker_session`:

```python
try:
    session = self._create_worker_session(
        agent_id=agent_id,
        subagent_type=resolved_subagent_type,
        model_profile=resolved_model_profile,
    )
except ValueError as error:
    return {"is_error": True, "error": str(error)}
```

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/integration/multi_agent/test_agent_profiles.py -q`

Expected: PASS.

---

### Task 5: WorkerRecord 保存 profile 和 model_profile

**Files:**
- Modify: `src/haagent/multi_agent/team_store.py`
- Modify: `src/haagent/multi_agent/runtime.py`
- Test: `tests/unit/multi_agent/test_team_store.py`
- Test: `tests/integration/multi_agent/test_agent_profiles.py`

**Interfaces:**
- Produces: `WorkerRecord.profile: str`
- Produces: `WorkerRecord.model_profile: str`

- [ ] **Step 1: Write the failing tests**

```python
def test_worker_record_persists_profile_fields(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")
    store.upsert_worker(
        "team-1",
        WorkerRecord(
            agent_id="explorer-1",
            task_id="task-1",
            subagent_type="explorer",
            description="Inspect",
            status="completed",
            session_id="session-1",
            profile="explorer",
            model_profile="fast",
        ),
    )

    team = store.load_team("team-1")

    assert team is not None
    assert team.agents[0].profile == "explorer"
    assert team.agents[0].model_profile == "fast"
```

Add integration assertion to `test_spawn_worker_accepts_profile_name`:

```python
record = runtime.task_get(result["task_id"])
assert record["profile"] == "explorer"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/multi_agent/test_team_store.py tests/integration/multi_agent/test_agent_profiles.py -q`

Expected: FAIL because `WorkerRecord` lacks `profile` and `model_profile`.

- [ ] **Step 3: Extend WorkerRecord**

In `src/haagent/multi_agent/team_store.py`:

```python
@dataclass
class WorkerRecord:
    agent_id: str
    task_id: str
    subagent_type: WorkerType
    description: str
    status: WorkerStatus
    session_id: str = ""
    episode_path: str = ""
    restart_count: int = 0
    status_note: str = ""
    profile: str = ""
    model_profile: str = ""
```

In `load_team`, when reconstructing records, use defaults for missing fields by calling `WorkerRecord(**worker)` only after setting missing keys:

```python
for worker in raw.get("agents", []):
    worker.setdefault("profile", "")
    worker.setdefault("model_profile", "")
```

This is not legacy compatibility for old releases; it prevents a same-run partial file from breaking after this field is introduced during development.

- [ ] **Step 4: Include fields in runtime payloads**

In `spawn_worker`, set:

```python
record = WorkerRecord(
    ...
    profile=agent_profile.name,
    model_profile=resolved_model_profile or "",
)
```

In `_worker_record_payload`, add:

```python
"profile": worker.profile,
"model_profile": worker.model_profile,
```

In `send_message`, pass `model_profile=record.model_profile or None` into `_create_worker_session`.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/multi_agent/test_team_store.py tests/integration/multi_agent/test_agent_profiles.py -q`

Expected: PASS.

---

### Task 6: 结构化 worker 通知数据模型

**Files:**
- Create: `src/haagent/multi_agent/messages.py`
- Modify: `src/haagent/multi_agent/runtime.py`
- Test: `tests/unit/multi_agent/test_messages.py`
- Test: `tests/integration/multi_agent/test_agent_tools.py`

**Interfaces:**
- Produces: `WorkerNotification`
- Produces: `WorkerNotification.to_dict() -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

```python
"""
tests/unit/multi_agent/test_messages.py - worker 消息结构测试

验证 worker 通知和权限请求的稳定序列化字段。
"""

from haagent.multi_agent.messages import WorkerNotification


def test_worker_notification_to_dict_has_stable_fields() -> None:
    notification = WorkerNotification(
        event_type="worker_completed",
        team_id="team-1",
        agent_id="explorer-1",
        task_id="task-1",
        status="completed",
        summary="done",
        result_excerpt="done",
        episode_path=".runs/session/episode",
        error="",
        needs_attention=False,
    )

    assert notification.to_dict() == {
        "event_type": "worker_completed",
        "team_id": "team-1",
        "agent_id": "explorer-1",
        "task_id": "task-1",
        "status": "completed",
        "summary": "done",
        "result_excerpt": "done",
        "episode_path": ".runs/session/episode",
        "error": "",
        "needs_attention": False,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_messages.py -q`

Expected: FAIL because `messages.py` does not exist.

- [ ] **Step 3: Add messages module**

```python
"""
haagent/multi_agent/messages.py - worker 通信数据结构

定义主 Agent 与 worker 之间的结构化通知、消息和权限请求。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkerNotification:
    event_type: str
    team_id: str
    agent_id: str
    task_id: str
    status: str
    summary: str
    result_excerpt: str
    episode_path: str
    error: str
    needs_attention: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

- [ ] **Step 4: Use WorkerNotification in runtime**

In `MultiAgentRuntime._notification`, return `WorkerNotification(...).to_dict()`:

```python
from haagent.multi_agent.messages import WorkerNotification


return WorkerNotification(
    event_type="worker_status",
    team_id=worker.team_id,
    agent_id=worker.agent_id,
    task_id=worker.task_id,
    status=status,
    summary=summary,
    result_excerpt=result_excerpt,
    episode_path=episode_path,
    error=error,
    needs_attention=bool(error),
).to_dict()
```

- [ ] **Step 5: Run existing multi-agent tests**

Run: `uv run pytest tests/unit/multi_agent/test_messages.py tests/integration/multi_agent/test_agent_tools.py -q`

Expected: PASS.

---

### Task 7: TeamStore 支持运行中消息队列

**Files:**
- Modify: `src/haagent/multi_agent/messages.py`
- Modify: `src/haagent/multi_agent/team_store.py`
- Test: `tests/unit/multi_agent/test_team_store_messages.py`

**Interfaces:**
- Produces: `WorkerMessage`
- Produces: `TeamStore.write_worker_message(team_id: str, agent_id: str, message: WorkerMessage) -> Path`
- Produces: `TeamStore.read_worker_messages(team_id: str, agent_id: str) -> list[WorkerMessage]`

- [ ] **Step 1: Write the failing tests**

```python
"""
tests/unit/multi_agent/test_team_store_messages.py - worker mailbox 测试

验证主 Agent 写给 worker 的消息可以按顺序读取。
"""

from pathlib import Path

from haagent.multi_agent.messages import WorkerMessage
from haagent.multi_agent.team_store import TeamStore


def test_team_store_writes_and_reads_worker_messages(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")

    store.write_worker_message(
        "team-1",
        "worker-1",
        WorkerMessage(sender="coordinator", recipient="worker-1", content="first"),
    )
    store.write_worker_message(
        "team-1",
        "worker-1",
        WorkerMessage(sender="coordinator", recipient="worker-1", content="second"),
    )

    messages = store.read_worker_messages("team-1", "worker-1")

    assert [message.content for message in messages] == ["first", "second"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_team_store_messages.py -q`

Expected: FAIL because `WorkerMessage` or TeamStore methods do not exist.

- [ ] **Step 3: Add WorkerMessage**

In `messages.py`:

```python
import time
import uuid


@dataclass(frozen=True)
class WorkerMessage:
    sender: str
    recipient: str
    content: str
    message_id: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["message_id"] = self.message_id or f"msg-{uuid.uuid4().hex[:12]}"
        payload["created_at"] = self.created_at or time.time()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WorkerMessage":
        return cls(
            sender=str(payload["sender"]),
            recipient=str(payload["recipient"]),
            content=str(payload["content"]),
            message_id=str(payload["message_id"]),
            created_at=float(payload["created_at"]),
        )
```

- [ ] **Step 4: Add TeamStore methods**

In `team_store.py`:

```python
from haagent.multi_agent.messages import WorkerMessage


def write_worker_message(self, team_id: str, agent_id: str, message: WorkerMessage) -> Path:
    payload = message.to_dict()
    inbox = self._team_dir(team_id) / "agents" / _safe_id(agent_id) / "messages"
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"{payload['created_at']}-{payload['message_id']}.json"
    return _atomic_write_json(path, payload)


def read_worker_messages(self, team_id: str, agent_id: str) -> list[WorkerMessage]:
    inbox = self._team_dir(team_id) / "agents" / _safe_id(agent_id) / "messages"
    if not inbox.exists():
        return []
    messages = []
    for path in sorted(inbox.glob("*.json")):
        messages.append(WorkerMessage.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return messages
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/multi_agent/test_team_store_messages.py -q`

Expected: PASS.

---

### Task 8: send_message 允许运行中 worker 在安全点接收消息

**Files:**
- Modify: `src/haagent/multi_agent/runtime.py`
- Test: `tests/integration/multi_agent/test_worker_messaging.py`

**Interfaces:**
- Consumes: `TeamStore.write_worker_message(...)`
- Produces: `MultiAgentRuntime.send_message(to: str, message: str) -> dict[str, Any]` returns queued status for running worker

- [ ] **Step 1: Write the failing test**

```python
"""
tests/integration/multi_agent/test_worker_messaging.py - worker 运行中通信测试

验证运行中的 worker 可以接收排队消息，而不是直接拒绝。
"""

from pathlib import Path

from haagent.models.gateway import ModelResponse
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


class _NeverFinishGateway:
    def generate(self, messages, tool_schemas):
        return ModelResponse(content="", tool_calls=[])


def test_send_message_to_running_worker_is_queued(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=_NeverFinishGateway(),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
        worker_max_turns=20,
    )
    worker = runtime.spawn_worker(description="Loop", prompt="Keep going", subagent_type="worker")

    result = runtime.send_message(worker["agent_id"], "new instruction")

    assert result["status"] == "queued"
    runtime.stop_task(worker["task_id"], force=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/multi_agent/test_worker_messaging.py -q`

Expected: FAIL because current `send_message` returns `agent ... is still running`.

- [ ] **Step 3: Queue message instead of rejecting running worker**

In `MultiAgentRuntime.send_message`:

```python
if worker.thread is not None and worker.thread.is_alive() and not worker.done.is_set():
    self.store.write_worker_message(
        worker.team_id,
        worker.agent_id,
        WorkerMessage(sender="coordinator", recipient=worker.agent_id, content=message),
    )
    return {
        "agent_id": worker.agent_id,
        "task_id": worker.task_id,
        "status": "queued",
    }
```

Keep the existing restart behavior for completed workers.

- [ ] **Step 4: Run focused test**

Run: `uv run pytest tests/integration/multi_agent/test_worker_messaging.py -q`

Expected: PASS.

---

### Task 9: Worker 权限请求结构

**Files:**
- Modify: `src/haagent/multi_agent/messages.py`
- Modify: `src/haagent/multi_agent/team_store.py`
- Test: `tests/unit/multi_agent/test_messages.py`
- Test: `tests/unit/multi_agent/test_team_store_messages.py`

**Interfaces:**
- Produces: `WorkerPermissionRequest`
- Produces: `TeamStore.write_permission_request(request: WorkerPermissionRequest) -> Path`
- Produces: `TeamStore.read_permission_requests(team_id: str, *, status: str = "pending") -> list[WorkerPermissionRequest]`

- [ ] **Step 1: Write the failing tests**

```python
from haagent.multi_agent.messages import WorkerPermissionRequest


def test_worker_permission_request_to_dict_has_stable_fields() -> None:
    request = WorkerPermissionRequest(
        request_id="perm-1",
        team_id="team-1",
        agent_id="worker-1",
        task_id="task-1",
        tool_name="shell",
        tool_args_summary="uv run pytest",
        reason="需要运行测试确认修改。",
        status="pending",
    )

    assert request.to_dict()["status"] == "pending"
    assert request.to_dict()["tool_name"] == "shell"
```

In `test_team_store_messages.py`:

```python
def test_team_store_writes_and_reads_permission_requests(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")
    request = WorkerPermissionRequest(
        request_id="perm-1",
        team_id="team-1",
        agent_id="worker-1",
        task_id="task-1",
        tool_name="shell",
        tool_args_summary="uv run pytest",
        reason="需要验证。",
        status="pending",
    )

    store.write_permission_request(request)

    requests = store.read_permission_requests("team-1")
    assert [item.request_id for item in requests] == ["perm-1"]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/unit/multi_agent/test_messages.py tests/unit/multi_agent/test_team_store_messages.py -q`

Expected: FAIL because permission request model and store methods do not exist.

- [ ] **Step 3: Add WorkerPermissionRequest**

In `messages.py`:

```python
@dataclass(frozen=True)
class WorkerPermissionRequest:
    request_id: str
    team_id: str
    agent_id: str
    task_id: str
    tool_name: str
    tool_args_summary: str
    reason: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WorkerPermissionRequest":
        return cls(
            request_id=str(payload["request_id"]),
            team_id=str(payload["team_id"]),
            agent_id=str(payload["agent_id"]),
            task_id=str(payload["task_id"]),
            tool_name=str(payload["tool_name"]),
            tool_args_summary=str(payload["tool_args_summary"]),
            reason=str(payload["reason"]),
            status=str(payload["status"]),
        )
```

- [ ] **Step 4: Add TeamStore permission methods**

In `team_store.py`:

```python
from haagent.multi_agent.messages import WorkerPermissionRequest


def write_permission_request(self, request: WorkerPermissionRequest) -> Path:
    directory = self._team_dir(request.team_id) / "permissions" / request.status
    directory.mkdir(parents=True, exist_ok=True)
    return _atomic_write_json(directory / f"{_safe_id(request.request_id)}.json", request.to_dict())


def read_permission_requests(self, team_id: str, *, status: str = "pending") -> list[WorkerPermissionRequest]:
    directory = self._team_dir(team_id) / "permissions" / _safe_id(status)
    if not directory.exists():
        return []
    requests = []
    for path in sorted(directory.glob("*.json")):
        requests.append(WorkerPermissionRequest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
    return requests
```

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/multi_agent/test_messages.py tests/unit/multi_agent/test_team_store_messages.py -q`

Expected: PASS.

---

### Task 10: Backend 接口为 subprocess 隔离预留扩展点

**Files:**
- Create: `src/haagent/multi_agent/backends.py`
- Test: `tests/unit/multi_agent/test_backends.py`

**Interfaces:**
- Produces: `WorkerBackend` protocol
- Produces: `InProcessWorkerBackend`

- [ ] **Step 1: Write the failing tests**

```python
"""
tests/unit/multi_agent/test_backends.py - worker backend 接口测试

验证第三阶段隔离能力有稳定接口，但默认仍是 in-process。
"""

from haagent.multi_agent.backends import InProcessWorkerBackend


def test_in_process_backend_type_is_stable() -> None:
    backend = InProcessWorkerBackend()

    assert backend.backend_type == "in_process"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_backends.py -q`

Expected: FAIL because `backends.py` does not exist.

- [ ] **Step 3: Add backend protocol**

```python
"""
haagent/multi_agent/backends.py - worker 执行后端接口

为后续 subprocess 和 worktree 隔离预留扩展点，默认不改变当前 in-process 行为。
"""

from __future__ import annotations

from typing import Any, Protocol


class WorkerBackend(Protocol):
    @property
    def backend_type(self) -> str:
        """返回后端类型。"""

    def spawn(self, runtime: Any, worker: Any, prompt: str) -> None:
        """启动 worker。"""


class InProcessWorkerBackend:
    @property
    def backend_type(self) -> str:
        return "in_process"

    def spawn(self, runtime: Any, worker: Any, prompt: str) -> None:
        runtime._start_worker_thread(worker, prompt)
```

- [ ] **Step 4: Run focused test**

Run: `uv run pytest tests/unit/multi_agent/test_backends.py -q`

Expected: PASS.

---

### Task 11: Worktree slug 校验

**Files:**
- Create: `src/haagent/multi_agent/worktree.py`
- Test: `tests/unit/multi_agent/test_worktree.py`

**Interfaces:**
- Produces: `validate_worktree_slug(slug: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
"""
tests/unit/multi_agent/test_worktree.py - worker worktree 路径校验测试

验证后续隔离工作区不会接受路径穿越或绝对路径。
"""

import pytest

from haagent.multi_agent.worktree import validate_worktree_slug


def test_validate_worktree_slug_accepts_simple_slug() -> None:
    assert validate_worktree_slug("fix-tests") == "fix-tests"


@pytest.mark.parametrize("slug", ["", ".", "..", "../x", "x/../y", "/tmp/x", "C:\\\\tmp", "bad name"])
def test_validate_worktree_slug_rejects_unsafe_values(slug: str) -> None:
    with pytest.raises(ValueError):
        validate_worktree_slug(slug)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/multi_agent/test_worktree.py -q`

Expected: FAIL because `worktree.py` does not exist.

- [ ] **Step 3: Add slug validator**

```python
"""
haagent/multi_agent/worktree.py - worker 隔离工作区管理

提供 Git worktree 前置校验，后续再接入真实创建和清理流程。
"""

from __future__ import annotations

import re

_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_worktree_slug(slug: str) -> str:
    if not slug or slug.startswith("/") or "\\" in slug or ":" in slug:
        raise ValueError("invalid worktree slug")
    parts = slug.split("/")
    for part in parts:
        if part in {"", ".", ".."}:
            raise ValueError("invalid worktree slug")
        if not _SEGMENT_PATTERN.fullmatch(part):
            raise ValueError("invalid worktree slug")
    if len(slug) > 64:
        raise ValueError("invalid worktree slug")
    return slug
```

- [ ] **Step 4: Run focused test**

Run: `uv run pytest tests/unit/multi_agent/test_worktree.py -q`

Expected: PASS.

---

### Task 12: Final integration gate

**Files:**
- Modify: no source changes beyond previous tasks
- Test: multi-agent, tool registry, runtime session focused suites

**Interfaces:**
- Consumes: all previous tasks
- Produces: verified implementation ready for user review

- [ ] **Step 1: Run focused multi-agent suite**

Run: `uv run pytest tests/unit/multi_agent tests/integration/multi_agent -q`

Expected: PASS.

- [ ] **Step 2: Run tool registry and router focused tests**

Run: `uv run pytest tests/unit/tools/test_tool_registry.py tests/integration/tools/test_tool_router.py -q`

Expected: PASS.

- [ ] **Step 3: Run runtime session focused tests**

Run: `uv run pytest tests/unit/runtime/test_agent_session.py tests/integration/runtime/test_chat_turn_runner.py -q`

Expected: PASS.

- [ ] **Step 4: Run default fast suite because runtime/tool contracts changed**

Run: `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Run quality gate**

Run: `uv run haagent check`

Expected: PASS.

## Self-Review

- Spec coverage: 阶段一由 Tasks 1-5 覆盖；阶段二由 Tasks 6-9 覆盖；阶段三扩展点由 Tasks 10-11 覆盖；最终验证由 Task 12 覆盖。
- Placeholder scan: 未发现待填写或延后补充内容。
- Type consistency: `AgentProfile`、`WorkerNotification`、`WorkerMessage`、`WorkerPermissionRequest`、`WorkerBackend` 在首次任务中定义，后续任务只消费这些名称。
- Scope check: 计划保留为三阶段路线图，但每个任务都是可单独测试的小交付；subprocess 真实实现和真实 worktree 创建没有塞进前两阶段。

## Execution Handoff

计划已保存到 `docs/superpowers/plans/2026-07-04-multi-agent-capability-roadmap.md`。后续执行可选两种方式：

1. Subagent-Driven：每个任务派一个新的 subagent，任务之间人工复核，适合快速并行推进。
2. Inline Execution：在当前会话按 executing-plans 逐步执行，适合需要更强连续上下文的实现。
