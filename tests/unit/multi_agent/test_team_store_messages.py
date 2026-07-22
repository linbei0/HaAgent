"""
tests/unit/multi_agent/test_team_store_messages.py - worker mailbox 测试

验证主 Agent 写给 worker 的消息可以按顺序读取。
"""

from pathlib import Path

from haagent.multi_agent.messages import WorkerMessage, WorkerPermissionRequest
from haagent.multi_agent.team_store import TeamStore


def test_team_store_writes_and_reads_worker_messages(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")

    store.write_worker_message(
        "team-1",
        "worker-1",
        WorkerMessage(
            sender="coordinator",
            recipient="worker-1",
            content="first",
            message_id="z-first",
            created_at=100.0,
        ),
    )
    store.write_worker_message(
        "team-1",
        "worker-1",
        WorkerMessage(
            sender="coordinator",
            recipient="worker-1",
            content="second",
            message_id="a-second",
            created_at=100.0,
        ),
    )

    messages = store.read_worker_messages("team-1", "worker-1")

    assert [message.content for message in messages] == ["first", "second"]


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
    assert requests[0].status == "pending"
