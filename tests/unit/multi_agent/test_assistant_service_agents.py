"""
tests/unit/multi_agent/test_assistant_service_agents.py - 服务层 worker 状态查询测试

验证 TUI 可通过 AssistantService 从用户级 team 存储读取当前 session 的 worker 摘要。
"""

from pathlib import Path
from haagent.app import workspace_usecases
from haagent.app.assistant_service import AssistantService
from haagent.multi_agent.messages import WorkerNotification
from haagent.multi_agent.team_store import TeamStore, WorkerRecord


class _Session:
    provider_name = "openai-chat"
    turn_count = 0

    def __init__(self, *, workspace_root: Path, runs_root: Path, **kwargs) -> None:
        del kwargs
        self.session_id = "session-test"
        self.workspace_root = workspace_root
        self.runs_root = runs_root
        self.session_path = runs_root / "sessions" / self.session_id
        self.max_turns = None
        self.model_variant = None


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
    service = AssistantService(
        workspace_root=tmp_path,
        gateway_factory=lambda profile: object(),
        session_cls=_Session,  # type: ignore[arg-type]
    )
    service._context.session = _Session(workspace_root=tmp_path, runs_root=tmp_path / ".runs")

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


def test_assistant_service_delivers_ui_notifications_once(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "home" / ".haagent"
    monkeypatch.setattr(workspace_usecases, "user_config_dir", lambda: config_dir)
    store = TeamStore(config_dir / "teams")
    store.ensure_team(
        team_id="team-session-test",
        workspace_root=tmp_path,
        leader_session_id="session-test",
    )
    store.append_notification(
        "team-session-test",
        WorkerNotification(
            event_type="worker_status",
            team_id="team-session-test",
            agent_id="explorer-1",
            task_id="task-1",
            status="completed",
            summary="done",
            result_excerpt="full output",
            episode_path="episode",
            error="",
            needs_attention=False,
        ).to_dict(),
    )
    service = AssistantService(
        workspace_root=tmp_path,
        gateway_factory=lambda profile: object(),
        session_cls=_Session,  # type: ignore[arg-type]
    )
    service._context.session = _Session(workspace_root=tmp_path, runs_root=tmp_path / ".runs")
    delivered = []

    first_count = service.workspace.poll_worker_notifications(delivered.append)
    second_count = service.workspace.poll_worker_notifications(delivered.append)

    assert first_count == 1
    assert second_count == 0
    assert delivered[0].task_id == "task-1"


def test_assistant_service_lists_no_agents_without_session(tmp_path: Path) -> None:
    service = AssistantService(workspace_root=tmp_path)

    assert service.workspace.list_agents() == []
