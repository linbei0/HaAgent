"""
tests/integration/multi_agent/test_worker_notifications.py - worker 通知摘要集成测试

验证存储中的结构化通知会被 leader 摘要消费。
"""

from pathlib import Path

import haagent.runtime.orchestration.orchestrator as orchestrator_module
from haagent.multi_agent.messages import WorkerNotification
from haagent.multi_agent.team_store import TeamStore


def test_worker_notification_context_reads_structured_notification_from_store(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader-1")
    store.append_notification(
        "team-1",
        WorkerNotification(
            event_type="worker_status",
            team_id="team-1",
            agent_id="verification-1",
            task_id="task-123",
            status="awaiting_approval",
            summary="shell uv run pytest",
            result_excerpt="",
            episode_path="",
            error="",
            needs_attention=True,
            request_id="perm-456",
        ).to_dict(),
    )
    monkeypatch.setattr(orchestrator_module, "user_config_dir", lambda: tmp_path)

    context = orchestrator_module._worker_notification_context("leader-1")

    assert context is not None
    assert "verification-1" in context
    assert "task-123" in context
    assert "perm-456" in context
