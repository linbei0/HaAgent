"""
tests/integration/cli/test_cli_personal_assistant.py - 个人助手启动体验测试

验证用户级模型配置、默认 haagent 入口和目录会话恢复。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent import cli
from haagent.models.types import ModelResponse
from haagent.models.model_ref import ModelRef
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.contracts.task import load_task


class RecordingGateway:
    provider_name = "recording"

    def __init__(self) -> None:
        self.model_inputs: list[str] = []

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        self.model_inputs.append(model_input)
        return ModelResponse(f"done: {' '.join(m.get('content', '') for m in messages if m.get('role') == 'user')}", [])


class ConciseRecordingGateway(RecordingGateway):
    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        self.model_inputs.append(model_input)
        return ModelResponse("done", [])


class SmartCompactGateway(RecordingGateway):
    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        self.model_inputs.append(model_input)
        if "HaAgent full compact summarizer" in model_input:
            return ModelResponse(
                json.dumps(
                    {
                        "task_focus": "继续当前用户会话",
                        "completed_work": ["保留用户已确认的实现方向"],
                        "open_issues": [],
                        "important_files": ["src/haagent/runtime/chat_session.py"],
                        "tool_results": ["旧轮次已压缩为智能摘要"],
                        "constraints": ["后续对话继续使用压缩后的会话记忆"],
                        "verification": ["压缩后继续运行下一轮"],
                        "risks": [],
                    },
                    ensure_ascii=False,
                ),
                [],
            )
        return ModelResponse("done", [])


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


def test_legacy_personal_commands_point_to_tui(capsys) -> None:
    for argv in (
        ["setup"],
        ["chat", "--web", "Hello"],
        ["sessions", "--workspace-root", "."],
        ["memory", "confirm", "candidate-1"],
        ["tui", "--web"],
    ):
        exit_code = cli.main(argv)
        output = capsys.readouterr().out

        assert exit_code == 1
        assert "请运行 haagent 打开 TUI" in output


def test_haagent_without_subcommand_starts_tui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runs_root = tmp_path / "runs"
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.initial_resume = kwargs["initial_resume"]
            self.initial_continue = kwargs["initial_continue"]

    monkeypatch.setattr("haagent.cli_commands.AssistantService", FakeService)
    def fake_run_tui(service) -> int:
        captured["initial_resume"] = service.initial_resume
        captured["initial_continue"] = service.initial_continue
        captured["ran"] = True
        return 0

    monkeypatch.setattr("haagent.cli_commands.run_tui", fake_run_tui)

    exit_code = cli.main(
        [
            "--workspace-root",
            str(workspace),
            "--runs-root",
            str(runs_root),
            "--resume",
            "session-abc",
            "--web",
        ],
    )

    assert exit_code == 0
    assert captured["workspace_root"] == workspace
    assert captured["runs_root"] == runs_root
    assert captured["enable_web"] is True
    assert captured["initial_resume"] == "session-abc"
    assert captured["initial_continue"] is False
    assert captured["ran"] is True


def test_haagent_tui_default_workspace_is_current_directory(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("haagent.cli_commands.AssistantService", FakeService)
    monkeypatch.setattr("haagent.cli_commands.run_tui", lambda service: 0)

    exit_code = cli.main([])

    assert exit_code == 0
    assert captured["workspace_root"] == workspace


def test_chat_task_does_not_include_web_tools_by_default(tmp_path: Path, monkeypatch) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
    )

    result = session.run_prompt("Do local work")
    task = load_task(result.episode_path / "task.yaml")

    assert "web_search" not in task.allowed_tools
    assert "web_fetch" not in task.allowed_tools
    assert "skill_market_search" in task.allowed_tools
    assert "skill_list" in task.allowed_tools
    assert "skill_read" in task.allowed_tools
    assert "code_run" in task.allowed_tools
    assert "apply_patch" in task.allowed_tools
    assert "agent" in task.allowed_tools


def test_chat_task_includes_skill_tools_when_user_skills_exist(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    _set_home(monkeypatch, home)
    skill_dir = home / ".haagent" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review workflow.\n---\n\nPRIVATE BODY",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = RecordingGateway()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=gateway,
    )

    result = session.run_prompt("Review files")
    task = load_task(result.episode_path / "task.yaml")
    manifest = json.loads((result.episode_path / "contexts" / "0001-manifest.json").read_text(encoding="utf-8"))

    assert "skill_list" in task.allowed_tools
    assert "skill_read" in task.allowed_tools
    assert "Available Skills:" in gateway.model_inputs[0]
    assert "When a listed skill clearly applies to the task" in gateway.model_inputs[0]
    assert "- review [user]: Review workflow." in gateway.model_inputs[0]
    assert "- haagent-config (invoke: customize-haagent) [builtin]" in gateway.model_inputs[0]
    assert "PRIVATE BODY" not in gateway.model_inputs[0]
    assert manifest["source_diagnostics"]["skills"]["available_count"] == 2


def test_chat_task_includes_web_tools_when_explicitly_enabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
        enable_web=True,
    )

    result = session.run_prompt("Search the web")
    task = load_task(result.episode_path / "task.yaml")

    assert "web_search" in task.allowed_tools
    assert "web_fetch" in task.allowed_tools
    assert "skill_market_search" in task.allowed_tools
    assert "web_search" not in task.policy["approval_allowed_tools"]
    assert "web_fetch" not in task.policy["approval_allowed_tools"]
    assert "skill_market_search" not in task.policy["approval_allowed_tools"]


def test_chat_session_auto_compacts_old_turn_summaries_without_model_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = ConciseRecordingGateway()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=gateway,
    )

    results = [session.run_prompt(f"turn {index}") for index in range(1, 9)]
    final_episode = results[-1].episode_path
    context_manifest = json.loads((final_episode / "contexts" / "0001-manifest.json").read_text(encoding="utf-8"))
    transcript = [
        json.loads(line)
        for line in (final_episode / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert "[session_memory_compacted 1 earlier turns]" in gateway.model_inputs[-1]
    assert "- user_request: turn 1" not in gateway.model_inputs[-1].splitlines()
    # 新行为：最近轮完整问答以 `user:` 原文进入模型输入，而非截断摘要行。
    for index in range(2, 8):
        assert f"user: turn {index}" in gateway.model_inputs[-1].splitlines()
    assert context_manifest["session_compaction"]["decision"] == "compacted"
    assert context_manifest["session_compaction"]["compacted_turn_count"] == 1
    assert any(event.get("event") == "session_memory_compaction" for event in transcript)


def test_chat_session_manual_compact_uses_smart_summary_for_next_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = SmartCompactGateway()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=gateway,
    )
    for index in range(1, 9):
        session.run_prompt(f"turn {index}")

    compact_result = session.compact_current_session()
    session.run_prompt("continue after compact")

    assert compact_result.applied is True
    assert compact_result.reason == "applied"
    assert any("HaAgent full compact summarizer" in item for item in gateway.model_inputs)
    assert "Full Compact Summary:" in gateway.model_inputs[-1]
    assert "保留用户已确认的实现方向" in gateway.model_inputs[-1]
    assert "- user_request: turn 1" not in gateway.model_inputs[-1].splitlines()
    for index in range(3, 9):
        assert f"- user_request: turn {index}" in gateway.model_inputs[-1].splitlines()


def test_session_metadata_records_model_profile_without_api_key(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession(
        workspace_root=workspace,
        runs_root=tmp_path / ".runs",
        model_gateway=RecordingGateway(),
        model_ref=ModelRef("router", "openai/gpt-5.2-chat"),
    )
    session.switch_model_gateway(
        ModelRef("requesty", "openai/gpt-5.2"),
        RecordingGateway(),
    )

    metadata_text = (session.session_path / "session.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)

    assert metadata["model_ref"] == {"connection_id": "requesty", "model": "openai/gpt-5.2"}
    assert "api_key" not in metadata
    assert "sk-" not in metadata_text


def test_continue_restores_latest_current_workspace_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    old_session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
    )
    old_session.run_prompt("old")
    latest_session = AgentSession(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        model_gateway=RecordingGateway(),
    )
    latest_session.run_prompt("latest")
    captured: dict[str, object] = {}

    class FakeService:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.initial_continue = kwargs["initial_continue"]

    monkeypatch.chdir(workspace)
    monkeypatch.setattr("haagent.cli_commands.AssistantService", FakeService)
    def fake_run_tui(service) -> int:
        captured["initial_continue"] = service.initial_continue
        return 0

    monkeypatch.setattr("haagent.cli_commands.run_tui", fake_run_tui)

    exit_code = cli.main(["--continue"])

    assert exit_code == 0
    assert captured["initial_continue"] is True
    assert captured["workspace_root"] == workspace


def test_haagent_tui_resume_and_continue_conflict(capsys) -> None:
    exit_code = cli.main(["--resume", "session-abc", "--continue"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert output.strip() == "error: --resume cannot be combined with --continue"
