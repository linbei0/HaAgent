"""
tests/unit/multi_agent/test_messages.py - worker 消息结构测试

验证 worker 通知和权限请求的稳定序列化字段。
"""

from haagent.multi_agent.messages import WorkerNotification, WorkerPermissionRequest


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
        request_id="",
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
        "request_id": "",
    }


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
