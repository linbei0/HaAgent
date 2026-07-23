"""
tests/tui/test_sessions.py - HaAgent TUI sessions 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from haagent.app.assistant_types import AssistantSessionTurn
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.overlays.sessions import SessionOverlayState
from haagent.tui.widgets import PromptInput

from tests.tui.support import FakeAssistantService, _all_text, _session_summary, _text

def test_tui_compact_command_compacts_session_without_running_prompt(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/compact"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.compacted_count == 1
            assert service.prompts == []
            assert "已压缩当前会话" in _text(app, "#conversation")
            await pilot.press("up")
            assert input_widget.value == ""

    asyncio.run(run())

def test_tui_session_overlay_state_filters_and_selects_sessions(tmp_path: Path) -> None:
    sessions = [
        _session_summary(tmp_path, "session-alpha", "整理会议纪要", 3),
        _session_summary(tmp_path, "session-beta", "分析 CSV", 1),
    ]
    state = SessionOverlayState(sessions=sessions)

    filtered = state.with_query("csv")
    selected = filtered.move(1)
    empty = filtered.with_query("none")

    assert [item.session_id for item in filtered.visible_sessions] == ["session-beta"]
    assert selected.selected_session.session_id == "session-beta"
    assert empty.selected_session is None
    assert "无匹配会话" in empty.render()

def test_tui_plain_s_does_not_open_sessions_or_search(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            assert input_widget.has_focus
            await pilot.press("s")
            await pilot.pause(0.1)
            assert input_widget.value == "s"
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)
            assert "范围: conversation" not in _all_text(app)

    asyncio.run(run())

def test_tui_sessions_overlay_search_resume_continue_new_and_escape(tmp_path: Path) -> None:
    sessions = [
        _session_summary(tmp_path, "session-alpha", "整理会议纪要", 3),
        _session_summary(tmp_path, "session-beta", "分析 CSV", 1),
    ]

    async def run_resume() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            sessions=sessions,
            session_histories={
                "session-beta": [
                    AssistantSessionTurn(
                        turn_index=1,
                        request="分析 CSV",
                        summary="用户要分析 sales.csv，助手已说明会检查列名和异常值。",
                        status="completed",
                        episode_path=tmp_path / ".runs" / "episode-1",
                        verification_status="success",
                    ),
                ],
            },
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" in _all_text(app)
            await pilot.press("c", "s", "v")
            await pilot.pause(0.1)
            assert "session-beta" in _all_text(app)
            assert "session-alpha" not in str(app.screen.render())
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.resumed_sessions == [str(sessions[1].session_path)]
            assert service.current_session_id == "session-beta"
            assert "session-beta" not in _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "分析 CSV" in conversation
            assert "用户要分析 sales.csv" in conversation
            assert "当前会话：session-beta" not in conversation
            assert "整理会议纪要" not in conversation
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)
            await pilot.press("up")
            assert input_widget.value == "分析 CSV"

    async def run_continue_new_escape() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            sessions=sessions,
            session_histories={
                "session-alpha": [
                    AssistantSessionTurn(
                        turn_index=1,
                        request="整理会议纪要",
                        summary="用户要整理会议纪要。",
                        status="completed",
                        episode_path=tmp_path / ".runs" / "episode-alpha",
                        verification_status="success",
                    ),
                ],
            },
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("l")
            await pilot.pause(0.1)
            assert service.continued_latest_count == 1
            assert service.current_session_id == "session-alpha"
            conversation = _text(app, "#conversation")
            assert "整理会议纪要" in conversation
            assert "用户要整理会议纪要。" in conversation
            await pilot.press("up")
            assert input_widget.value == "整理会议纪要"

            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.1)
            assert service.created_sessions == ["session-new-1"]
            assert "当前会话：session-new-1" not in _text(app, "#conversation")
            assert service.current_session_id == "session-new-1"
            assert "sid:" not in _text(app, "#status-bar")
            await pilot.press("up")
            assert input_widget.value == ""

            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)

    asyncio.run(run_resume())
    asyncio.run(run_continue_new_escape())

def test_tui_resume_command_replays_latest_session_history(tmp_path: Path) -> None:
    session = _session_summary(tmp_path, "session-alpha", "整理会议纪要", 1)
    service = FakeAssistantService(
        workspace_root=tmp_path,
        sessions=[session],
        session_histories={
            "session-alpha": [
                AssistantSessionTurn(
                    turn_index=1,
                    request="整理会议纪要",
                    summary="用户要整理会议纪要。",
                    status="completed",
                    episode_path=tmp_path / ".runs" / "episode-alpha",
                    verification_status="success",
                ),
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "整理会议纪要" in conversation
            assert "用户要整理会议纪要。" in conversation

    asyncio.run(run())


def test_tui_new_session_command_clears_previous_timeline(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path, assistant_content="旧回答")

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "旧问题"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "旧问题" in _text(app, "#conversation")
            assert "旧回答" in _text(app, "#conversation")

            input_widget.value = "/new"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert service.created_sessions == ["session-new-1"]
            assert "旧问题" not in conversation
            assert "旧回答" not in conversation
            assert "新建会话：session-new-1" not in conversation
            assert service.current_session_id == "session-new-1"
            assert "sid:" not in _text(app, "#status-bar")

    asyncio.run(run())

def test_tui_mcp_command_renders_configured_not_loaded_without_session(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        mcp_status={
            "configured_count": 1,
            "connected_count": 0,
            "failed_count": 0,
            "servers": [
                {
                    "name": "exa",
                    "state": "configured",
                    "detail": "not loaded; create or resume a session to connect",
                    "tool_count": 0,
                    "resource_count": 0,
                }
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/mcp"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "MCP servers:" in conversation
            assert "exa: configured - not loaded; create or resume a session to connect" in conversation

    asyncio.run(run())

def test_tui_restored_session_renders_final_response_not_raw_turn_summary(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-raw", "恢复旧会话", 1)]
    raw_summary = "\n".join(
        [
            "- user_request: 恢复旧会话",
            "  status: completed",
            f"  episode_path: {tmp_path / '.runs' / 'episode-1'}",
            "  assistant_final_response: 这是恢复后应该看到的回答。",
            "  verification: success",
        ],
    )
    service = FakeAssistantService(
        workspace_root=tmp_path,
        sessions=sessions,
        session_histories={
            "session-raw": [
                AssistantSessionTurn(
                    turn_index=1,
                    request="恢复旧会话",
                    summary=raw_summary,
                    status="completed",
                    episode_path=tmp_path / ".runs" / "episode-1",
                    verification_status="success",
                ),
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "恢复旧会话" in conversation
            assert "这是恢复后应该看到的回答。" in conversation
            assert "user_request:" not in conversation
            assert "episode_path:" not in conversation
            assert "verification:" not in conversation

    asyncio.run(run())

def test_tui_restored_session_prefers_assistant_display_text(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-display", "恢复长回答", 1)]
    raw_summary = "\n".join(
        [
            "- user_request: 恢复长回答",
            "  status: completed",
            f"  episode_path: {tmp_path / '.runs' / 'episode-1'}",
            "  assistant_final_response: 摘要里的短回答... [truncated]",
            "  verification: success",
        ],
    )
    full_display_text = "这是用于恢复展示的较完整回答，不应该退回到摘要里的截断文本。"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        sessions=sessions,
        session_histories={
            "session-display": [
                AssistantSessionTurn(
                    turn_index=1,
                    request="恢复长回答",
                    summary=raw_summary,
                    status="completed",
                    episode_path=tmp_path / ".runs" / "episode-1",
                    verification_status="success",
                    assistant_display_text=full_display_text,
                ),
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert full_display_text in conversation
            assert "摘要里的短回答" not in conversation

    asyncio.run(run())

def test_tui_restores_initial_resume_session_on_mount(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        service.initial_resume = "session-from-cli"
        ui_thread_id = threading.get_ident()
        resume_thread_ids: list[int] = []
        original_resume = service.sessions.resume

        def recording_resume(session_path):
            resume_thread_ids.append(threading.get_ident())
            return original_resume(session_path)

        service.sessions.resume = recording_resume
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert service.resumed_sessions == ["session-from-cli"]
            assert resume_thread_ids and resume_thread_ids[0] != ui_thread_id
            assert service.current_session_id == "session-from-cli"
            assert "sid:" not in _text(app, "#status-bar")

    asyncio.run(run())

def test_tui_continues_initial_latest_session_on_mount(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-alpha", "整理会议纪要", 3)]

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, sessions=sessions)
        service.initial_continue = True
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert service.continued_latest_count == 1
            assert service.current_session_id == "session-alpha"
            assert "sid:" not in _text(app, "#status-bar")

    asyncio.run(run())

