"""
tests/integration/multi_agent/test_worker_permissions.py - worker 权限请求集成测试

验证 worker 高风险工具会形成结构化 pending request，而不是直接静默失败。
"""

from pathlib import Path

from haagent.models.gateway import ModelResponse, ToolCall
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


class _ShellApprovalGateway:
    provider_name = "fake"

    def generate(self, messages, tool_schemas):
        return ModelResponse(
            content="run shell",
            tool_calls=[
                ToolCall(
                    name="shell",
                    args={"command": "uv run pytest -q"},
                    id="call-shell-1",
                ),
            ],
        )


class _ApprovalResumeGateway:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, tool_schemas):
        self.calls += 1
        if any(message.get("role") == "tool" for message in messages):
            return ModelResponse(content="completed after approved shell", tool_calls=[])
        return ModelResponse(
            content="run approved shell",
            tool_calls=[
                ToolCall(
                    name="shell",
                    args={"command": "python -c \"print('approved run')\""},
                    id=f"call-shell-{self.calls}",
                ),
            ],
        )


def _runtime(tmp_path: Path, gateway) -> MultiAgentRuntime:
    return MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=gateway,
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "shell", "file_read"],
        inherited_approval_allowed_tools=["shell"],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
    )


def test_worker_high_risk_tool_creates_pending_permission_request(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _ShellApprovalGateway())

    worker = runtime.spawn_worker(
        description="run tests",
        prompt="use shell to run pytest",
        subagent_type="verification",
        team_id="team-test",
    )

    notification = runtime.wait_for_task(worker["task_id"], timeout=5)
    requests = runtime.store.read_permission_requests(worker["team_id"], status="pending")

    assert notification["status"] == "awaiting_approval"
    assert len(requests) == 1
    assert requests[0].tool_name == "shell"
    assert requests[0].task_id == worker["task_id"]
    assert "pytest" in requests[0].tool_args_summary
    task_state = runtime.task_get(worker["task_id"])
    assert task_state["task"]["status"] == "awaiting_approval"


def test_approved_permission_resumes_worker_and_completes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _ApprovalResumeGateway())
    worker = runtime.spawn_worker(
        description="run tests",
        prompt="use shell to run pytest then report",
        subagent_type="verification",
        team_id="team-test",
    )
    notification = runtime.wait_for_task(worker["task_id"], timeout=5)
    assert notification["status"] == "awaiting_approval"
    pending = runtime.store.read_permission_requests(worker["team_id"], status="pending")[0]

    approved = runtime.approve_permission(pending.request_id, message="可以执行")

    assert approved["status"] == "approved"
    finished = runtime.wait_for_task(worker["task_id"], timeout=5)
    assert finished["status"] == "completed"
    assert runtime.store.read_permission_requests(worker["team_id"], status="pending") == []
    assert runtime.store.read_permission_requests(worker["team_id"], status="approved") == []
    consumed = runtime.store.read_permission_requests(worker["team_id"], status="consumed")
    assert [request.request_id for request in consumed] == [pending.request_id]
    tool_calls = (Path(str(finished["episode_path"])) / "tool-calls.jsonl").read_text(encoding="utf-8")
    assert "approved run" in tool_calls


def test_rejected_permission_fails_worker_and_consumes_request(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, _ApprovalResumeGateway())
    worker = runtime.spawn_worker(
        description="run tests",
        prompt="use shell to run pytest then report",
        subagent_type="verification",
        team_id="team-test",
    )
    notification = runtime.wait_for_task(worker["task_id"], timeout=5)
    assert notification["status"] == "awaiting_approval"
    pending = runtime.store.read_permission_requests(worker["team_id"], status="pending")[0]

    rejected = runtime.reject_permission(pending.request_id, message="不要执行")

    assert rejected["status"] == "rejected"
    finished = runtime.wait_for_task(worker["task_id"], timeout=5)
    assert finished["status"] == "failed"
    assert "rejected" in finished["error"]
    assert runtime.store.read_permission_requests(worker["team_id"], status="pending") == []
    assert runtime.store.read_permission_requests(worker["team_id"], status="rejected") == []
    consumed = runtime.store.read_permission_requests(worker["team_id"], status="consumed")
    assert [request.request_id for request in consumed] == [pending.request_id]
