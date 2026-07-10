"""
tests/unit/multi_agent/test_assistant_service_agents.py - 服务层 worker 状态查询测试

验证 TUI 可通过 AssistantService 从用户级 team 存储读取当前 session 的 worker 摘要。
"""

from pathlib import Path
from types import SimpleNamespace

from haagent.app import workspace_usecases
from haagent.app.assistant_service import AssistantService
from haagent.multi_agent.team_store import TeamStore, WorkerRecord


def test_assistant_service_lists_agents_for_current_session(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "home" / ".haagent"
    monkeypatch.setattr(workspace_usecases, "user_config_dir", lambda: config_dir)
    store = TeamStore(config_dir / "teams")
    team = store.ensure_team(
        team_id="team-session-test",
        workspace_root=tmp_path,
        leader_session_id="session-test",
    )
    store.upsert_worker(
        team.team_id,
        WorkerRecord(
            agent_id="explorer-1",
            task_id="task-1",
            subagent_type="explorer",
            description="Inspect project",
            status="running",
        ),
    )
    other_team = store.ensure_team(
        team_id="team-other",
        workspace_root=tmp_path,
        leader_session_id="session-other",
    )
    store.upsert_worker(
        other_team.team_id,
        WorkerRecord(
            agent_id="worker-other",
            task_id="task-other",
            subagent_type="worker",
            description="Ignore me",
            status="completed",
        ),
    )
    service = AssistantService(workspace_root=tmp_path)
    service._context.session = SimpleNamespace(session_id="session-test")

    agents = service.workspace.list_agents()

    assert agents == [
        {
            "team_id": "team-session-test",
            "agent_id": "explorer-1",
            "task_id": "task-1",
            "subagent_type": "explorer",
            "description": "Inspect project",
            "status": "running",
            "episode_path": "",
        },
    ]


def test_assistant_service_lists_no_agents_without_session(tmp_path: Path) -> None:
    service = AssistantService(workspace_root=tmp_path)

    assert service.workspace.list_agents() == []
