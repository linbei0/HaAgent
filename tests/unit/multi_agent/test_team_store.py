"""
tests/unit/multi_agent/test_team_store.py - 多智能体 team 存储测试

验证用户级 team 目录、mailbox 与通知记录的稳定 JSON 行为。
"""

import json
from pathlib import Path

from haagent.multi_agent.messages import WorkerNotification, WorkerPermissionRequest
from haagent.multi_agent.team_store import TeamStore, WorkerRecord


def test_team_store_writes_team_and_notifications(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / ".haagent" / "teams")
    team = store.ensure_team(
        team_id="team-demo",
        workspace_root=tmp_path / "workspace",
        leader_session_id="leader-1",
    )
    worker = WorkerRecord(
        agent_id="explorer-abc123",
        task_id="task-abc123",
        subagent_type="explorer",
        description="Inspect files",
        status="running",
    )

    store.upsert_worker(team.team_id, worker)
    store.append_notification(
        team.team_id,
        {
            "task_id": worker.task_id,
            "agent_id": worker.agent_id,
            "team_id": team.team_id,
            "status": "completed",
            "summary": "done",
            "result_excerpt": "README summarized",
            "usage": {},
            "error": "",
        },
    )

    team_file = tmp_path / ".haagent" / "teams" / "team-demo" / "team.json"
    saved_team = json.loads(team_file.read_text(encoding="utf-8"))
    assert saved_team["team_id"] == "team-demo"
    assert saved_team["agents"][0]["agent_id"] == "explorer-abc123"

    notification_lines = (
        tmp_path / ".haagent" / "teams" / "team-demo" / "notifications.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(notification_lines) == 1
    assert json.loads(notification_lines[0])["result_excerpt"] == "README summarized"


def test_team_store_marks_team_inactive_without_deleting_audit_data(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / ".haagent" / "teams")
    store.ensure_team(
        team_id="team-demo",
        workspace_root=tmp_path / "workspace",
        leader_session_id="leader-1",
    )

    store.mark_inactive("team-demo")

    saved_team = store.load_team("team-demo")
    assert saved_team is not None
    assert saved_team.active is False
    assert (tmp_path / ".haagent" / "teams" / "team-demo" / "team.json").exists()


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
            parent_step_id="todo-1",
        ),
    )

    team = store.load_team("team-1")

    assert team is not None
    assert team.agents[0].profile == "explorer"
    assert team.agents[0].model_profile == "fast"
    assert team.agents[0].parent_step_id == "todo-1"


def test_notification_consumers_have_independent_acknowledged_cursors(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")
    notification = WorkerNotification(
        event_type="worker_status",
        team_id="team-1",
        agent_id="explorer-1",
        task_id="task-1",
        status="completed",
        summary="done",
        result_excerpt="full output",
        episode_path="episode",
        error="",
        needs_attention=False,
    ).to_dict()
    store.append_notification("team-1", notification)

    model_batch = store.read_unread_notifications("team-1", consumer="model")
    ui_batch = store.read_unread_notifications("team-1", consumer="ui")
    assert [item["task_id"] for item in model_batch["notifications"]] == ["task-1"]
    assert [item["task_id"] for item in ui_batch["notifications"]] == ["task-1"]

    store.acknowledge_notifications(
        "team-1",
        consumer="ui",
        cursor=str(ui_batch["next_cursor"]),
    )
    assert store.read_unread_notifications("team-1", consumer="ui")["notifications"] == []
    assert store.read_unread_notifications("team-1", consumer="model")["notifications"]

    store.acknowledge_notifications(
        "team-1",
        consumer="model",
        cursor=str(model_batch["next_cursor"]),
    )
    assert store.read_unread_notifications("team-1", consumer="model")["notifications"] == []


def test_terminal_task_consumes_pending_permission_request(tmp_path: Path) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader")
    store.write_permission_request(
        WorkerPermissionRequest(
            request_id="perm-1",
            team_id="team-1",
            agent_id="worker-1",
            task_id="task-1",
            tool_name="shell",
            tool_args_summary="pytest",
            reason="approval required",
            status="pending",
        ),
    )

    cancelled = store.cancel_pending_permissions_for_task(
        "team-1",
        "task-1",
        reason="task settled as interrupted",
    )

    assert cancelled == 1
    assert store.read_permission_requests("team-1", status="pending") == []
    consumed = store.read_permission_requests("team-1", status="consumed")
    assert [request.request_id for request in consumed] == ["perm-1"]
