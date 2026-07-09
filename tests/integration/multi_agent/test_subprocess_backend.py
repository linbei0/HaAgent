"""
tests/integration/multi_agent/test_subprocess_backend.py - subprocess worker 集成测试

验证 subprocess backend 会启动真实子进程，并通过现有 task/output 接口回收结果。
"""

import os
from pathlib import Path

from haagent.models.fake import FakeModelGateway
from haagent.models.types import ModelResponse
from haagent.multi_agent.profiles import WorkerProfileRuntime
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


def test_subprocess_backend_worker_completes_and_outputs_result(tmp_path: Path) -> None:
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
        name="subprocess-worker",
        subagent_type="worker",
        system_prompt="你是子进程 worker。",
        model_profile=None,
        allowed_tools=None,
        approval_allowed_tools=None,
        approved_tools=None,
        max_turns=None,
        enable_web=None,
        backend="subprocess",
        worktree=False,
    )

    worker = runtime.spawn_worker(
        description="inspect",
        prompt="say done",
        subagent_type="worker",
    )

    assert worker["backend"] == "subprocess"
    assert isinstance(worker["process_id"], int)
    assert worker["process_id"] != os.getpid()
    finished = runtime.wait_for_task(worker["task_id"], timeout=10)
    assert finished["status"] == "completed"
    output = runtime.task_output(worker["task_id"])
    assert "done" in output["output"]
