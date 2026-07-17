"""
tests/unit/runtime/test_session_snapshot.py - SessionSnapshot / SessionResources 契约

验证 create→save→resume 与 reload 的 snapshot 等价性，以及 live 资源不进磁盘 schema。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.lifecycle import (
    SESSION_SNAPSHOT_SCHEMA_VERSION,
    SessionResources,
    SessionSnapshot,
    apply_state,
    build_create_state,
    build_resume_state,
)
from haagent.runtime.session.package import read_session_metadata


def test_apply_state_only_binds_snapshot_and_resources(tmp_path: Path) -> None:
    state = build_create_state(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        max_turns=3,
    )
    session = AgentSession.__new__(AgentSession)
    apply_state(session, state)

    assert session.snapshot is state.snapshot
    assert session.resources is state.resources
    assert set(session.__dict__) == {"_snapshot", "_resources"}
    assert isinstance(session.snapshot, SessionSnapshot)
    assert isinstance(session.resources, SessionResources)
    assert session.snapshot.schema_version == SESSION_SNAPSHOT_SCHEMA_VERSION


def test_create_save_resume_snapshot_equivalence(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    session = AgentSession(workspace_root=tmp_path, runs_root=runs_root, max_turns=5, enable_web=True)
    session.turn_count = 2
    # summaries 来自 turns.jsonl；写入一条 turn 以验证 resume 装载。
    from haagent.runtime.session.package import append_turn_record

    append_turn_record(
        session.session_path,
        turn_index=1,
        request="hello",
        summary="turn-1",
        status="completed",
        episode_path=tmp_path / "ep-1",
        verification_status="passed",
        final_response="ok",
    )
    session._write_session_metadata()
    session._write_working_state()
    session._write_task_ledger()

    metadata = read_session_metadata(session.session_path)
    assert "model_gateway" not in metadata
    assert "mcp_runtime" not in metadata
    assert "worker_permission_requester" not in metadata
    assert "tool_registry" not in metadata

    resumed = AgentSession.resume(
        session.session_id,
        runs_root=runs_root,
        max_turns=5,
        enable_web=True,
    )
    assert resumed.snapshot.session_id == session.snapshot.session_id
    assert resumed.snapshot.turn_count == session.snapshot.turn_count
    assert resumed.snapshot.workspace_root == session.snapshot.workspace_root
    assert resumed.snapshot.enable_web is True
    assert resumed.snapshot.summaries == ["turn-1"]
    assert resumed.resources.model_gateway is None
    assert resumed.resources.allowed_tools_override is None
    assert resumed.resources.mcp_runtime is not None


def test_reload_keeps_resources_and_refreshes_snapshot(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    session = AgentSession(workspace_root=tmp_path, runs_root=runs_root, max_turns=4)
    mcp = session.resources.mcp_runtime
    registry = session.resources.tool_registry
    session.turn_count = 1
    session._write_session_metadata()
    session._write_working_state()
    session._write_task_ledger()

    session.reload(session.session_id, runs_root=runs_root)
    assert session.resources.mcp_runtime is mcp
    assert session.resources.tool_registry is registry
    assert session.snapshot.turn_count == 1
    resumed = build_resume_state(
        session.session_id,
        runs_root=runs_root,
        max_turns=4,
        mcp_runtime=mcp,
        tool_registry=registry,
        owns_mcp_runtime=False,
    )
    assert resumed.snapshot.turn_count == session.snapshot.turn_count
    assert resumed.snapshot.session_id == session.snapshot.session_id


def test_session_snapshot_schema_version_persisted_and_restored(tmp_path: Path) -> None:
    """v1 写入 session.json，resume 后仍为 v1，不得静默改写为其他版本。"""
    runs_root = tmp_path / ".runs"
    session = AgentSession(workspace_root=tmp_path, runs_root=runs_root, max_turns=2)
    session._write_session_metadata()
    session._write_working_state()
    session._write_task_ledger()

    metadata = read_session_metadata(session.session_path)
    assert metadata["session_snapshot_schema_version"] == SESSION_SNAPSHOT_SCHEMA_VERSION

    resumed = AgentSession.resume(session.session_id, runs_root=runs_root, max_turns=2)
    assert resumed.snapshot.schema_version == SESSION_SNAPSHOT_SCHEMA_VERSION


def test_legacy_session_without_schema_version_migrates_to_current(tmp_path: Path) -> None:
    """旧 package 无版本字段按 v0 处理，恢复时显式迁移到当前 schema。"""
    import json

    runs_root = tmp_path / ".runs"
    session = AgentSession(workspace_root=tmp_path, runs_root=runs_root, max_turns=2)
    session._write_session_metadata()
    session._write_working_state()
    session._write_task_ledger()

    metadata_path = session.session_path / "session.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    del raw["session_snapshot_schema_version"]
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    resumed = AgentSession.resume(session.session_id, runs_root=runs_root, max_turns=2)
    assert resumed.snapshot.schema_version == SESSION_SNAPSHOT_SCHEMA_VERSION
    # 迁移后写回磁盘应落盘当前版本
    resumed._write_session_metadata()
    rewritten = read_session_metadata(session.session_path)
    assert rewritten["session_snapshot_schema_version"] == SESSION_SNAPSHOT_SCHEMA_VERSION


def test_unknown_session_snapshot_schema_version_rejected(tmp_path: Path) -> None:
    import json

    from haagent.runtime.session.package import ChatSessionError

    runs_root = tmp_path / ".runs"
    session = AgentSession(workspace_root=tmp_path, runs_root=runs_root, max_turns=2)
    session._write_session_metadata()
    session._write_working_state()
    session._write_task_ledger()

    metadata_path = session.session_path / "session.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["session_snapshot_schema_version"] = 99
    metadata_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        AgentSession.resume(session.session_id, runs_root=runs_root, max_turns=2)
        raise AssertionError("expected ChatSessionError for unknown schema version")
    except ChatSessionError as error:
        assert "session_snapshot_schema_version" in str(error)
