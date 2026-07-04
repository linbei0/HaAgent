"""
tests/unit/multi_agent/test_backends.py - worker backend 接口测试

验证第三阶段隔离能力有稳定接口，但默认仍是 in-process。
"""

from haagent.multi_agent.backends import InProcessWorkerBackend
from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import ModelResponse
from haagent.multi_agent.profiles import WorkerProfileRuntime
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


def test_in_process_backend_type_is_stable() -> None:
    backend = InProcessWorkerBackend()

    assert backend.backend_type == "in_process"


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
