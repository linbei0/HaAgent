"""
tests/unit/runtime/test_worker_notification_summary.py - worker 通知摘要测试

验证 leader 看到的 worker 通知来自结构化字段，而不是黑盒自由文本。
"""

from pathlib import Path

import haagent.runtime.orchestration.orchestrator as orchestrator_module
from haagent.multi_agent.messages import WorkerNotification
from haagent.multi_agent.team_store import TeamStore


def test_worker_notification_context_includes_attention_and_request_id(tmp_path: Path, monkeypatch) -> None:
    store = TeamStore(tmp_path / "teams")
    store.ensure_team(team_id="team-1", workspace_root=tmp_path, leader_session_id="leader-1")
    store.append_notification(
        "team-1",
        WorkerNotification(
            event_type="worker_status",
            team_id="team-1",
            agent_id="worker-1",
            task_id="task-1",
            status="awaiting_approval",
            summary="shell needs approval",
            result_excerpt="",
            episode_path="",
            error="",
            needs_attention=True,
            request_id="perm-1",
        ).to_dict(),
    )
    monkeypatch.setattr(orchestrator_module, "user_config_dir", lambda: tmp_path)

    context = orchestrator_module._worker_notification_context("leader-1")

    assert context is not None
    assert "awaiting_approval" in context
    assert "perm-1" in context
    assert "attention" in context
