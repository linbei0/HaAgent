"""
tests/unit/multi_agent/test_notification_context.py - worker notification 上下文注入测试

验证 coordinator 下一轮只读取紧凑通知摘要，不复制 worker 完整输出。
"""

from pathlib import Path

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
        {
            "task_id": "task-1",
            "agent_id": "explorer-1",
            "team_id": team.team_id,
            "status": "completed",
            "summary": "README summarized",
            "result_excerpt": "FULL WORKER TRANSCRIPT SHOULD STAY OUT",
            "usage": {},
            "error": "",
        },
    )

    context = orchestrator_module._worker_notification_context("leader-session")

    assert context == "Worker Notifications:\n- explorer-1 completed: README summarized (task=task-1)"
    assert "FULL WORKER TRANSCRIPT" not in context
