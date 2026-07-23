"""
tests/tui/test_memory.py - HaAgent TUI memory 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from haagent.runtime.events import MemoryNoticeEvent
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.design.keys import APP_BINDINGS, footer_text, help_body
from haagent.tui.design.renderers import memory_panel_text
from haagent.tui.widgets import PromptInput

from tests.tui.support import (
    FakeAssistantService,
    _all_text,
    _memory_candidate,
    _open_memory_panel,
    _text,
)

def test_tui_chat_memory_entry_is_only_slash_command() -> None:
    chat_footer = footer_text("chat")
    chat_help = help_body("chat")
    binding_keys = {binding.key if hasattr(binding, "key") else binding[0] for binding in APP_BINDINGS}
    input_binding_keys = {binding.key for binding in PromptInput.BINDINGS}

    assert "/memory" in chat_help
    assert "[m]记忆" not in chat_footer
    assert "m" not in binding_keys
    assert "m" not in input_binding_keys

def test_tui_memory_panel_renderer_marks_selection_and_detail() -> None:
    candidates = [
        _memory_candidate("cand_first", "第一条"),
        _memory_candidate("cand_second", "第二条"),
    ]

    list_text = memory_panel_text(
        candidates=candidates,
        selected_index=1,
        detail_mode=False,
        notice="发现候选",
        error=None,
    )
    detail_text = memory_panel_text(
        candidates=candidates,
        selected_index=1,
        detail_mode=True,
        notice=None,
        error=None,
    )

    assert "  发现候选" in list_text
    assert "  cand_first" in list_text
    assert "> cand_second" in list_text
    assert "候选编号：cand_second" in detail_text
    assert "候选编号：cand_first" not in detail_text

def test_tui_m_key_no_longer_opens_memory_mode(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)

            assert "记忆候选" not in _text(app, "#conversation")
            assert list(app.query("#side-bar")) == []

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "记忆候选" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_escape_closes_memory_mode_and_restores_empty_chat(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)

            await pilot.press("escape")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "记忆候选" not in conversation
            assert "可以开始了" in conversation
            assert _text(app, "#footer-bar") == footer_text("chat")
            assert app.query_one("#prompt-input").has_focus

    asyncio.run(run())


def test_tui_escape_closes_memory_mode_and_restores_existing_chat(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            app._conversation.append_line("原有对话内容")
            await pilot.pause(0.1)
            await _open_memory_panel(app, pilot)

            await pilot.press("escape")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "记忆候选" not in conversation
            assert "原有对话内容" in conversation

    asyncio.run(run())


def test_tui_help_modal_is_contextual_for_memory_modes(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert "记忆候选列表" in _all_text(app)
            assert "↑/↓" in _all_text(app)
            assert "g/G" in _all_text(app)
            assert "j/k" not in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)

            app.action_memory_enter()
            await pilot.pause(0.1)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert "记忆候选详情" in _all_text(app)
            assert "返回列表" in _all_text(app)

    asyncio.run(run())

def test_tui_memory_candidate_event_shows_notice(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[_memory_candidate()],
            extra_events=[
                MemoryNoticeEvent(
                    session_id="session-test",
                    turn_index=1,
                    count=1,
                    message="发现 1 条可记忆候选，已放入候选队列，等待你确认。",
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "记住我的爱好"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "发现 1 条可记忆候选，已放入候选队列，等待你确认。" in conversation
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation
            footer = _text(app, "#footer-bar")
            assert "[a/y/r]" in footer
            assert footer.count("[") <= 4

    asyncio.run(run())

def test_tui_memory_panel_lists_and_shows_candidate_details(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            conversation = _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation
            assert "用户身份与爱好" in conversation
            assert "[Enter]详情" in footer

            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "来源摘要：用户明确要求记住自己的名字和爱好。" in conversation
            assert "依据：用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in conversation
            assert "分类理由：这是跨 workspace 可复用的用户偏好和身份信息。" in conversation

    asyncio.run(run())

def test_tui_memory_navigation_selects_second_candidate_for_confirm_and_reject(tmp_path: Path) -> None:
    async def confirm_run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "> cand_second" in conversation
            assert "  cand_first" in conversation
            await pilot.press("a")
            await pilot.pause(0.1)
            assert service.confirmed_candidate_ids == ["cand_second"]

    async def reject_run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")
            await pilot.press("r")
            await pilot.pause(0.1)
            assert service.rejected_candidate_ids == [("cand_second", "rejected from TUI")]

    asyncio.run(confirm_run())
    asyncio.run(reject_run())

def test_tui_memory_navigation_supports_home_end_and_keeps_selection_after_detail(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_middle", "中间偏好"),
                _memory_candidate("cand_last", "最后偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("G")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#conversation")

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "候选编号：cand_last" in _text(app, "#conversation")
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#conversation")

            await pilot.press("g")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "[↑/↓]移动" in footer
            assert "j/k" not in footer
            assert "[g/G]" not in footer
            assert footer.count("[") <= 4

    asyncio.run(run())

def test_tui_memory_navigation_moves_one_candidate_per_keypress(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
                _memory_candidate("cand_third", "第三条偏好"),
                _memory_candidate("cand_fourth", "第四条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_third" in _text(app, "#conversation")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_j_k_do_not_move_memory_selection(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("j")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")
            await pilot.press("k")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_memory_mode_is_readable_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(90, 30)) as pilot:
            await _open_memory_panel(app, pilot)
            conversation = _text(app, "#conversation")
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation

            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "记忆候选详情" in conversation
            assert "依据：用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in conversation

    asyncio.run(run())

def test_tui_memory_confirm_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        ui_thread_id = threading.get_ident()
        confirm_thread_ids: list[int] = []
        original_confirm = service.memory.confirm_candidate

        def recording_confirm(candidate_id: str):
            confirm_thread_ids.append(threading.get_ident())
            return original_confirm(candidate_id)

        service.memory.confirm_candidate = recording_confirm
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("a")
            await pilot.pause(0.1)
            assert service.confirmed_candidate_ids == ["cand_abc123"]
            assert confirm_thread_ids and confirm_thread_ids[0] != ui_thread_id
            assert "已确认记忆候选：cand_abc123" in _text(app, "#conversation")
            assert "暂无待确认候选" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_memory_reject_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("r")
            await pilot.pause(0.1)
            assert service.rejected_candidate_ids == [("cand_abc123", "rejected from TUI")]
            assert "已拒绝记忆候选：cand_abc123" in _text(app, "#conversation")
            assert "暂无待确认候选" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_memory_panel_shows_empty_and_load_errors(tmp_path: Path) -> None:
    async def empty_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            assert "暂无待确认候选" in _text(app, "#conversation")

    async def error_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_error=RuntimeError("queue broken"))
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            assert "记忆候选不可用：queue broken" in _text(app, "#conversation")

    asyncio.run(empty_run())
    asyncio.run(error_run())

