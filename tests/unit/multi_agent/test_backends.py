"""
tests/unit/multi_agent/test_backends.py - worker backend 接口测试

验证第三阶段隔离能力有稳定接口，但默认仍是 in-process。
"""

import os
from pathlib import Path

from haagent.models.config.config_store import ModelConfigStore
from haagent.models.config.connections import ProviderConnectionRecord
from haagent.models.fake import FakeModelGateway
from haagent.models.model_options import ModelParameterConfig
from haagent.models.model_ref import ModelRef
from haagent.models.types import ModelResponse
from haagent.multi_agent.backends import InProcessWorkerBackend
from haagent.multi_agent.profiles import WorkerProfileRuntime
from haagent.multi_agent import runtime as multi_agent_runtime_module
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


def test_in_process_backend_type_is_stable() -> None:
    backend = InProcessWorkerBackend()

    assert backend.backend_type == "in_process"


def test_runtime_defaults_environ_to_os_environ_not_empty_mapping(tmp_path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent"],
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

    # 未传 environ 时必须继承进程环境，不能变成 {} 导致 env 凭据不可用。
    assert runtime.environ is os.environ
    assert runtime.model_runtime.environ is os.environ


def test_runtime_env_credential_resolve_uses_inherited_environ(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    store = ModelConfigStore(config_dir / "providers.json")
    connection = ProviderConnectionRecord(
        id="worker-conn",
        name="worker-conn",
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="HAAGENT_TEST_WORKER_API_KEY",
        credential_source="env",
        models={"gpt-4.1-mini": ModelParameterConfig({}, {})},
    )
    store.save_connection(connection, expected_digest=store.load().digest)
    from haagent.models.config.selection_store import ModelSelectionStore

    ModelSelectionStore(config_dir).save_active(ModelRef("worker-conn", "gpt-4.1-mini"))
    monkeypatch.setattr(multi_agent_runtime_module, "user_config_dir", lambda: config_dir)

    environ = {"HAAGENT_TEST_WORKER_API_KEY": "env-secret-key"}
    captured: list[object] = []

    def _fake_gateway(resolved):
        captured.append(resolved)
        return FakeModelGateway(ModelResponse(content="ok", tool_calls=[]))

    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
        environ=environ,
        gateway_factory=_fake_gateway,
    )

    # credential_source=env + worker model_profile 必须走继承 environ，不能因空映射失败。
    status = runtime.model_runtime.credential_status("worker-conn")
    assert status.api_key_available is True
    assert status.credential_source_used == "env"

    gateway = runtime._worker_gateway("worker-conn")
    assert isinstance(gateway, FakeModelGateway)
    assert len(captured) == 1
    resolved = captured[0]
    assert resolved.credential.api_key == "env-secret-key"
    assert resolved.credential.source_used == "env"
    assert resolved.ref == ModelRef("worker-conn", "gpt-4.1-mini")


def test_runtime_selects_subprocess_backend_from_profile(tmp_path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
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
    )
    runtime._profile_resolver = lambda *args, **kwargs: WorkerProfileRuntime(
        name="code-worker",
        subagent_type="worker",
        system_prompt="你是代码实现助手。",
        model_profile=None,
        allowed_tools=None,
        approval_allowed_tools=None,
        approved_tools=None,
        max_turns=None,
        enable_web=None,
        backend="subprocess",
        worktree=False,
    )

    result = runtime.spawn_worker(
        description="edit code",
        prompt="inspect",
        subagent_type="worker",
        profile="code-worker",
    )

    assert result["backend"] == "subprocess"
    assert runtime.wait_for_task(result["task_id"], timeout=15)["status"] == "completed"
