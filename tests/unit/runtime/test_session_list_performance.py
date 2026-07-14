"""
tests/unit/runtime/test_session_list_performance.py - 会话列表与历史热路径性能合同

覆盖 list_sessions 不解析完整 turns、turn_summaries 复用内存缓存，
以及 create 复用现有 session 的 MCP/gateway。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from haagent.models.model_connections import ModelSelection, ProviderProfile
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.package import (
    list_sessions,
    peek_first_turn_request,
    write_session_metadata,
)
from haagent.runtime.execution.path_policy import default_path_policy
from haagent.runtime.session.turn_completion import ChatTurnResult


def _write_session_package(
    runs_root: Path,
    workspace_root: Path,
    session_id: str,
    *,
    turn_count: int,
    first_request: str,
) -> Path:
    session_path = runs_root / "sessions" / session_id
    session_path.mkdir(parents=True)
    write_session_metadata(
        session_path,
        session_id=session_id,
        workspace_root=workspace_root,
        path_policy=default_path_policy(workspace_root),
        provider="fake",
        model_profile_name=None,
        model_connection_id=None,
        model_name=None,
        model_base_url=None,
        enable_web=False,
        last_user_image_attachments=[],
        image_attachment_history=[],
        created_at="2026-01-01T00:00:00+00:00",
        turn_count=turn_count,
    )
    lines: list[str] = []
    for index in range(1, turn_count + 1):
        request = first_request if index == 1 else f"follow-up {index}"
        record = {
            "turn_index": index,
            "request": request,
            "summary": f"summary {index}",
            "status": "completed",
            "episode_path": str(runs_root / "episodes" / f"ep-{index}"),
            "verification_status": "success",
            "assistant_display_text": f"answer {index}",
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    (session_path / "turns.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return session_path


def test_peek_first_turn_request_reads_only_first_line(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    workspace = tmp_path
    session_path = _write_session_package(
        runs_root,
        workspace,
        "session-a",
        turn_count=50,
        first_request="first prompt",
    )

    assert peek_first_turn_request(session_path) == "first prompt"


def test_list_sessions_does_not_parse_full_turns_file(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    workspace = tmp_path
    _write_session_package(
        runs_root,
        workspace,
        "session-a",
        turn_count=80,
        first_request="list me",
    )

    with patch(
        "haagent.runtime.session.package.read_session_turns",
        side_effect=AssertionError("list_sessions must not parse full turns.jsonl"),
    ):
        sessions = list_sessions(runs_root, workspace)

    assert len(sessions) == 1
    assert sessions[0].first_request == "list me"
    assert sessions[0].turn_count == 80


def test_assistant_create_reuses_existing_session_package(tmp_path: Path) -> None:
    from haagent.app.assistant_context import AssistantContext
    from haagent.app.session_usecases import AssistantSessions

    first = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    context = AssistantContext(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        environ={},
        gateway_factory=lambda profile: None,
        session_factory=AgentSession,
        max_turns=None,
        enable_web=False,
        initial_resume=None,
        initial_continue=False,
        session=first,
    )
    sessions = AssistantSessions(context)
    first_id = first.session_id
    mcp_before = first._mcp_runtime
    create_calls = {"count": 0}
    original_init = AgentSession.__init__

    def counting_init(self, *args, **kwargs):
        create_calls["count"] += 1
        return original_init(self, *args, **kwargs)

    with patch.object(AgentSession, "__init__", counting_init):
        second = sessions.create()

    assert create_calls["count"] == 0
    assert second.session_id != first_id
    assert context.session is first
    assert context.session._mcp_runtime is mcp_before
    assert second.turn_count == 0


def test_assistant_resume_reuses_existing_mcp_runtime(tmp_path: Path) -> None:
    """切换会话不得 close+重建 MCP；对齐 OpenCode 的进程级 runtime 复用。"""
    from haagent.app.assistant_context import AssistantContext
    from haagent.app.session_usecases import AssistantSessions

    first = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        memory_extraction_enabled=False,
    )
    result = ChatTurnResult(
        session_id=first.session_id,
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-1",
        provider="fake",
        final_response="answer",
        verification_status="success",
    )
    first.turn_count = 1
    first._record_turn("resume target", result, "summary")
    target_path = first.session_path
    first.new()

    context = AssistantContext(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        environ={},
        gateway_factory=lambda profile: None,
        session_factory=AgentSession,
        max_turns=None,
        enable_web=False,
        initial_resume=None,
        initial_continue=False,
        session=first,
    )
    sessions = AssistantSessions(context)
    mcp_before = first._mcp_runtime
    resume_profile = ProviderProfile(
        name="test",
        provider="openai",
        base_url="https://example.invalid/v1",
        model="test-model",
        api_key_env="TEST_API_KEY",
        credential_source="env",
        credential_source_used="env",
        api_key="test-key",
    )

    with patch(
        "haagent.app.session_usecases.load_active_model_selection",
        return_value=ModelSelection("test", "test-model"),
    ), patch(
        "haagent.app.session_usecases.load_model_selection_profile",
        return_value=resume_profile,
    ), patch(
        "haagent.runtime.session.lifecycle.bootstrap_mcp",
        side_effect=AssertionError("resume must reuse existing MCP runtime"),
    ), patch(
        "haagent.runtime.session.lifecycle.SyncMcpRuntime",
        side_effect=AssertionError("resume must not construct SyncMcpRuntime"),
    ):
        status = sessions.resume(target_path)

    assert context.session is first
    assert context.session._mcp_runtime is mcp_before
    assert status.session_id == target_path.name
    assert status.turn_count == 1


def test_list_sessions_prefers_metadata_first_request(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    workspace = tmp_path
    session_path = _write_session_package(
        runs_root,
        workspace,
        "session-meta",
        turn_count=3,
        first_request="from-turns",
    )
    metadata_path = session_path / "session.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["first_request"] = "from-metadata"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with patch(
        "haagent.runtime.session.package.peek_first_turn_request",
        side_effect=AssertionError("list_sessions must use session.json first_request when present"),
    ):
        sessions = list_sessions(runs_root, workspace)

    assert sessions[0].first_request == "from-metadata"
