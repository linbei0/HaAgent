"""
tests/integration/multi_agent/test_agent_profiles.py - worker profile 集成测试

验证 agent 工具可以通过 worker profile 派出 worker。
"""

from pathlib import Path

from haagent.models.fake import FakeModelGateway
from haagent.models.types import ModelResponse
from haagent.models.model_connections import ModelSelection, ProviderProfile, ProviderProfileError
from haagent.models.model_options import empty_resolved_config
import haagent.multi_agent.runtime as runtime_module
from haagent.multi_agent.profiles import WorkerProfileRuntime
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
    record = runtime.task_get(result["task_id"])
    assert record["task"]["profile"] == "explorer"


def test_spawn_worker_rejects_unknown_model_profile(tmp_path: Path, monkeypatch) -> None:
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
    monkeypatch.setattr(
        runtime_module,
        "load_active_model_selection",
        lambda *args, **kwargs: ModelSelection("local", "gpt-test"),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_model_selection_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ProviderProfileError("provider connection not found: missing-profile")
        ),
    )

    result = runtime.spawn_worker(
        description="Inspect",
        prompt="Say done",
        subagent_type="worker",
        model_profile="missing-profile",
    )

    assert result["is_error"] is True
    assert "missing-profile" in result["error"]


def test_spawn_worker_profile_overrides_session_settings_and_model_input(tmp_path: Path, monkeypatch) -> None:
    profile_gateway = FakeModelGateway(ModelResponse(content="done", tool_calls=[]))
    seen_profiles: list[ProviderProfile] = []

    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="leader", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read", "file_list", "shell"],
        inherited_approval_allowed_tools=["shell"],
        inherited_approved_tools=["shell"],
        event_sink=None,
        interaction_handler=None,
        enable_web=True,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
        gateway_factory=lambda profile: seen_profiles.append(profile) or profile_gateway,
    )
    runtime._profile_resolver = lambda *args, **kwargs: WorkerProfileRuntime(
        name="verification-fast",
        subagent_type="verification",
        system_prompt="你是验证助手。",
        model_profile="writer-model",
        allowed_tools=["file_read", "shell"],
        approval_allowed_tools=["shell"],
        approved_tools=[],
        max_turns=2,
        enable_web=False,
        backend="in_process",
        worktree=False,
    )
    monkeypatch.setattr(
        runtime_module,
        "load_active_model_selection",
        lambda *args, **kwargs: ModelSelection("writer-model", "gpt-test"),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_model_selection_profile",
        lambda *args, **kwargs: ProviderProfile(
            name="writer-model",
            provider="openai",
            base_url="https://example.test/v1",
            model="gpt-test",
            api_key_env="OPENAI_API_KEY",
            credential_source="env",
            credential_source_used="env",
            api_key="test-key",
            request_config=empty_resolved_config(
                connection_id="writer-model",
                model_id="gpt-test",
            ),
        ),
    )

    result = runtime.spawn_worker(
        description="verify",
        prompt="say done",
        subagent_type="worker",
        profile="verification-fast",
    )

    worker = runtime._find_worker(result["agent_id"])
    assert worker is not None
    assert worker.session.max_turns == 2
    assert worker.session.enable_web is False
    assert worker.session.model_profile_name == "writer-model"
    assert worker.session.model_gateway is profile_gateway
    assert worker.session._allowed_tools_override == ["file_read", "shell"]
    assert worker.session._approval_allowed_tools_override == ["shell"]
    assert worker.session._approved_tools_override == []

    finished = runtime.wait_for_task(result["task_id"], timeout=5)

    assert finished["status"] == "completed"
    assert seen_profiles[0].name == "writer-model"
    assert "你是验证助手。" in profile_gateway.calls[0]["model_input"]
