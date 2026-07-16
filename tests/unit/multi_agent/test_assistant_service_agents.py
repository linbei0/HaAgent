"""
tests/unit/multi_agent/test_assistant_service_agents.py - 服务层 worker 状态查询测试

验证 TUI 可通过 AssistantService 从用户级 team 存储读取当前 session 的 worker 摘要。
"""

from pathlib import Path
from types import SimpleNamespace

from haagent.app import session_usecases, workspace_usecases
from haagent.app.assistant_service import AssistantService
from haagent.models.model_connections import ModelSelection
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
    monkeypatch.setattr(
        session_usecases,
        "load_active_model_selection",
        lambda **kwargs: ModelSelection("test", "test-model"),
    )
    monkeypatch.setattr(
        session_usecases,
        "load_model_selection_profile",
        lambda selection, **kwargs: SimpleNamespace(
            name=selection.connection_id,
            provider="openai-chat",
            model=selection.model,
            base_url="https://example.test",
        ),
    )
    service = AssistantService(
        workspace_root=tmp_path,
        gateway_factory=lambda profile: object(),
        session_cls=_Session,  # type: ignore[arg-type]
    )
    service.sessions.create()

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
