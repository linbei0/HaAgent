"""
tests/unit/multi_agent/test_notification_context.py - worker notification 上下文注入测试

验证 coordinator 下一轮只读取紧凑通知摘要，不复制 worker 完整输出。
"""

from pathlib import Path

from haagent.multi_agent.messages import WorkerNotification
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.orchestration import orchestrator as orchestrator_module


def test_worker_notification_context_uses_summary_without_full_result_excerpt(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "home" / ".haagent"
    monkeypatch.setattr(orchestrator_module, "user_config_dir", lambda: config_dir)
    store = TeamStore(config_dir / "teams")
    team = store.ensure_team(
        team_id="team-leader",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
    )
    store.append_notification(
        team.team_id,
        WorkerNotification(
            event_type="worker_status",
            task_id="task-1",
            agent_id="explorer-1",
            team_id=team.team_id,
            status="completed",
            summary="README summarized",
            result_excerpt="FULL WORKER TRANSCRIPT SHOULD STAY OUT",
            episode_path="episode",
            error="",
            needs_attention=False,
        ).to_dict(),
    )

    context, acknowledgements = orchestrator_module._worker_notification_context("leader-session")

    assert context is not None
    assert "explorer-1 completed: README summarized (task=task-1)" in context
    assert "FULL WORKER TRANSCRIPT" not in context
    assert acknowledgements and acknowledgements[0][0] == team.team_id

    orchestrator_module._acknowledge_worker_notifications(acknowledgements, consumer="model")
    repeated, repeated_acknowledgements = orchestrator_module._worker_notification_context("leader-session")
    assert repeated is None
    assert repeated_acknowledgements == []
