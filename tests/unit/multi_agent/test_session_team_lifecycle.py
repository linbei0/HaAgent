"""
tests/unit/multi_agent/test_session_team_lifecycle.py - 主会话关闭时的 team 生命周期测试

验证主 AgentSession 结束只标记 team inactive，不删除多智能体审计目录。
"""

from pathlib import Path

from haagent.models.fake import FakeModelGateway
from haagent.multi_agent.team_store import TeamStore, WorkerRecord
from haagent.runtime.session import agent as agent_module
from haagent.runtime.session.agent import AgentSession


def test_agent_session_close_marks_teams_inactive_without_deleting_records(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "home" / ".haagent"
    monkeypatch.setattr(agent_module, "user_config_dir", lambda: config_dir)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=FakeModelGateway(),
        session_id="session-team-close",
    )
    store = TeamStore(config_dir / "teams")
    store.ensure_team(
        team_id="team-session-team-close",
        workspace_root=tmp_path,
        leader_session_id=session.session_id,
    )

    session.close()

    team = store.load_team("team-session-team-close")
    assert team is not None
    assert team.active is False
    assert (config_dir / "teams" / "team-session-team-close" / "team.json").exists()

    # close 必须幂等，重复退出不能重复关闭资源或改写审计状态。
    session.close()


def test_agent_session_close_reconciles_orphaned_running_workers(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "home" / ".haagent"
    monkeypatch.setattr(agent_module, "user_config_dir", lambda: config_dir)
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=FakeModelGateway(),
        session_id="session-orphan-close",
    )
    store = TeamStore(config_dir / "teams")
    store.ensure_team(
        team_id="team-session-orphan-close",
        workspace_root=tmp_path,
        leader_session_id=session.session_id,
    )
    store.upsert_worker(
        "team-session-orphan-close",
        WorkerRecord(
            agent_id="explorer-orphan",
            task_id="task-orphan",
            subagent_type="explorer",
            description="orphaned task",
            status="running",
        ),
    )

    session.close()

    team = store.load_team("team-session-orphan-close")
    assert team is not None
    assert team.agents[0].status == "interrupted"
    notifications = store.read_notifications(team.team_id)
    assert notifications[-1]["status"] == "interrupted"
