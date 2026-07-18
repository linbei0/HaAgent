"""
tests/tui/test_status_theme.py - HaAgent TUI status_theme 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from haagent.runtime.events import ContextUsageEvent
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.design.failures import failure_from_payload, failure_next_steps
from haagent.tui.design.keys import footer_text, help_body, key_help_lines
from haagent.tui.design.copy import MODAL_TITLES, PANEL_TITLES
from haagent.tui.design.renderers import context_usage_line, status_line
from haagent.tui.state import ResponsiveLayout, layout_for_size
from haagent.tui.design.theme import (
    TuiThemeMode,
    no_color_enabled,
    select_theme,
)
from haagent.tui.widgets import ContextUsageLine, ConversationTimeline, PromptInput
from textual.widgets import TextArea

from tests.tui.support import FakeAssistantService, _all_text, _memory_candidate, _text, _tool_event

def test_tui_status_line_renderer_truncates_to_terminal_width(tmp_path: Path) -> None:
    status = FakeAssistantService(
        workspace_root=tmp_path / "very-long-workspace-name-for-status-rendering",
        model="very-long-model-name-for-status-rendering",
        current_session_id="session-abcdefghijklmnopqrstuvwxyz",
    ).workspace.status()

    line_80 = status_line(status, ui_state="waiting approval", width=80)
    line_120 = status_line(status, ui_state="running", width=120)

    assert isinstance(line_80, Text)
    assert line_80.cell_len <= 80
    assert line_120.cell_len <= 120
    assert "工作区" in line_80.plain
    assert "模型" in line_80.plain
    assert "待确认" in line_80.plain
    assert "正在工作" in line_120.plain
    for internal_label in ("ws:", "profile", "perm:", "sandbox:", "sid:", "turn:", "state:", "key:"):
        assert internal_label not in line_80.plain
        assert internal_label not in line_120.plain

def test_tui_status_renderers_show_explicit_web_state(tmp_path: Path) -> None:
    offline = FakeAssistantService(workspace_root=tmp_path / "offline").workspace.status()
    online = FakeAssistantService(workspace_root=tmp_path / "online", enable_web=True).workspace.status()

    assert "联网已关" in status_line(offline, ui_state="idle", width=120).plain
    assert "联网已开" in status_line(online, ui_state="idle", width=120).plain


def test_context_usage_renderer_adapts_to_width_and_known_limit() -> None:
    assert context_usage_line(116_200, 500_000, terminal_width=120).plain == "116.2K（23%）"
    assert context_usage_line(116_200, 500_000, terminal_width=119).plain == "23%"
    assert context_usage_line(116_200, None, terminal_width=80).plain == "116.2K"
    assert context_usage_line(600_000, 500_000, terminal_width=120).plain == "600K（120%）"
    assert context_usage_line(0, 500_000, terminal_width=120).plain == ""


def test_context_usage_widget_is_hidden_until_real_usage_and_clears_with_session(tmp_path: Path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            widget = app.query_one("#context-usage", ContextUsageLine)
            assert widget.display is False

            app.update_context_usage(
                ContextUsageEvent(
                    session_id="session-1",
                    turn_index=1,
                    model_turn=1,
                    input_tokens=116_200,
                    input_window_tokens=500_000,
                ),
            )
            await pilot.pause()

            assert widget.display is True
            assert widget._Static__content.plain == "116.2K（23%）"

            app._update_responsive_layout(width=119, height=40)
            assert widget._Static__content.plain == "23%"

            app._update_responsive_layout(width=79, height=23)
            assert widget.display is False

            app._update_responsive_layout(width=120, height=40)
            assert widget.display is True
            assert widget._Static__content.plain == "116.2K（23%）"

            app.session_flow.clear_conversation_for_new_session()
            assert widget.display is False
            assert widget._Static__content == ""

    asyncio.run(run())


def test_tui_status_line_uses_widget_content_width_without_clipping_work_state(tmp_path: Path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)):
            status_bar = app.query_one("#status-bar")
            initial = status_bar._Static__content

            assert initial.cell_len == status_bar.content_region.width
            assert initial.plain.endswith("空闲")

            app._state = "running"
            app._refresh()
            rendered = status_bar._Static__content

            assert rendered.cell_len == status_bar.content_region.width
            assert rendered.plain.endswith("正在工作")

    asyncio.run(run())

def test_tui_keymap_help_and_footer_share_context_definitions() -> None:
    for context in ("chat", "running", "memory_list", "memory_detail", "pending_input", "approval", "too_small"):
        footer = footer_text(context)
        help_text = help_body(context)
        for key, _description in key_help_lines(context, include_footer_only=False):
            assert key in help_text
        for key, _description in key_help_lines(context, footer_only=True):
            assert key in footer
    assert footer_text("chat") == "[/]命令 [Ctrl+F]搜索 [?]帮助 [Ctrl+Q]退出"
    assert footer_text("running") == "[Ctrl+X]取消任务 [Ctrl+F]搜索 [?]帮助 [Ctrl+Q]退出"
    assert "Enter" not in footer_text("chat")
    assert "Shift+Enter" not in footer_text("chat")
    assert all(len(key_help_lines(context, footer_only=True)) <= 4 for context in (
        "chat", "running", "memory_list", "memory_detail", "pending_input", "approval", "edit_diff",
    ))
    assert "切换主题" in help_body("chat")
    assert "End" not in footer_text("chat")
    assert "End" in help_body("chat")
    assert "回到底部" in help_body("chat")

def test_tui_theme_selection_respects_env_and_no_color(monkeypatch) -> None:
    assert not no_color_enabled({})
    assert no_color_enabled({"NO_COLOR": "1"})
    assert no_color_enabled({"NO_COLOR": ""})

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)
    assert select_theme().mode is TuiThemeMode.DARK
    assert select_theme("light").mode is TuiThemeMode.LIGHT
    assert select_theme("monochrome").mode is TuiThemeMode.MONOCHROME

    monkeypatch.setenv("HAAGENT_TUI_THEME", "light")
    assert select_theme().mode is TuiThemeMode.LIGHT
    monkeypatch.setenv("NO_COLOR", "1")
    assert select_theme().mode is TuiThemeMode.MONOCHROME


def test_tui_copy_titles_are_chinese_and_keep_protocol_names() -> None:
    assert PANEL_TITLES["conversation"] == "对话"
    assert PANEL_TITLES["sessions"] == "会话"
    assert PANEL_TITLES["memory"] == "记忆候选"
    assert PANEL_TITLES["search"] == "搜索"
    assert MODAL_TITLES["approval"] == "工具审批"
    assert "workbench" not in PANEL_TITLES
    assert "tools" not in PANEL_TITLES
    assert "tool_details" not in MODAL_TITLES

def test_tui_responsive_layout_state_is_testable_without_widgets() -> None:
    assert layout_for_size(79, 24) == ResponsiveLayout(too_small=True)
    assert layout_for_size(80, 23) == ResponsiveLayout(too_small=True)
    assert layout_for_size(80, 24) == ResponsiveLayout(too_small=False)
    assert layout_for_size(120, 24) == ResponsiveLayout(too_small=False)

def test_tui_failure_next_steps_are_conservative() -> None:
    steps = failure_next_steps(
        failed_stage="executing",
        failure_category="Tool Argument Failure",
        reason="path does not exist",
        episode_path=".runs/episode-failed",
    )
    joined = "\n".join(steps)

    assert "重试" in joined
    assert "调整请求" in joined
    assert ".runs/episode-failed" in joined
    assert "/tools" not in joined
    assert "工具详情" not in joined
    assert "工具时间线" not in joined
    assert "修复完成" not in joined
    assert "一定是" not in joined

def test_tui_failure_payload_reports_missing_fields_explicitly() -> None:
    view = failure_from_payload({"status": "failed"}, fallback_message="")

    assert view.failed_stage == "缺少字段: failed_stage"
    assert view.failure_category == "缺少字段: failure_category"
    assert view.reason == "缺少字段: reason"
    assert view.episode_path == "缺少字段: episode_path"
    assert "unknown" not in view.block_text()

def test_tui_app_starts_and_shows_status(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "工作区" in status
            assert str(tmp_path) not in status
            assert "模型 deepseek-chat" in status
            assert "联网已关" in status
            assert "空闲" in status
            assert "profile" not in status
            assert "openai-chat" not in status
            assert "key:" not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert list(app.query("#side-bar")) == []
            assert "Ctrl+Enter 换行" in conversation
            assert "/ 打开命令" in conversation
            assert "Enter 发送" not in conversation
            assert "Ready." not in conversation
            footer = _text(app, "#footer-bar")
            assert "[Ctrl+Q]退出" in footer
            assert "[q]退出" not in footer
            assert "[Enter]发送" not in str(app.query_one("#footer-bar").render())
            assert "[Ctrl+Enter]换行" not in str(app.query_one("#footer-bar").render())
            assert isinstance(app.query_one("#prompt-input"), TextArea)

    asyncio.run(run())


def test_tui_prompt_focus_uses_only_a_thin_left_edge(tmp_path: Path) -> None:
    async def run() -> None:
        app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            prompt = app.query_one("#prompt-input", PromptInput)

            assert prompt.has_focus
            assert prompt.styles.border_left[0] == "solid"
            assert not prompt.styles.border_top[0]
            assert not prompt.styles.border_right[0]
            assert not prompt.styles.border_bottom[0]

    asyncio.run(run())

def test_tui_default_theme_applies_local_status_styles_and_chinese_titles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status_widget = app.query_one("#status-bar")

            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            assert not any(name.startswith("status-") for name in status_widget.classes)
            assert "空闲" in _text(app, "#status-bar")
            assert list(app.query("#side-bar")) == []
            conversation = _text(app, "#conversation")
            assert "Ctrl+Enter 换行" in conversation

    asyncio.run(run())

def test_tui_light_theme_can_be_enabled_with_local_status_styles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("HAAGENT_TUI_THEME", "light")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert app.theme == "haagent-light"
            assert app.screen.has_class("theme-light")
            assert not any(name.startswith("status-") for name in app.query_one("#status-bar").classes)
            assert "空闲" in _text(app, "#status-bar")
            assert list(app.query("#side-bar")) == []

    asyncio.run(run())

def test_tui_theme_can_be_cycled_with_keyboard_shortcut(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            conversation_before = _text(app, "#conversation")

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-light"
            assert app.screen.has_class("theme-light")
            assert _text(app, "#conversation") == conversation_before

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert _text(app, "#conversation") == conversation_before

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            assert _text(app, "#conversation") == conversation_before

    asyncio.run(run())

def test_tui_no_color_prevents_keyboard_theme_cycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-monochrome"
            conversation_before = _text(app, "#conversation")

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert _text(app, "#conversation") == conversation_before

    asyncio.run(run())

def test_tui_no_color_mode_keeps_symbols_text_and_selection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    candidates = [_memory_candidate("cand_first", "第一条"), _memory_candidate("cand_second", "第二条")]

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=candidates)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert "空闲" in _text(app, "#status-bar")

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")

            assert list(app.query("#side-bar")) == []
            assert "记忆候选" in conversation
            assert "> cand_second" in conversation
            assert "确认" in _text(app, "#footer-bar")

    asyncio.run(run())

def test_tui_responsive_minimum_size_and_layout_breakpoints(tmp_path: Path) -> None:
    async def run_too_small() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(70, 20)):
            assert "终端尺寸过小" in _all_text(app)
            assert "请调整到至少 80x24" in _all_text(app)
            assert app.query_one("#main").has_class("hidden")
            assert app.query_one("#input-panel").has_class("hidden")

    async def run_80() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            assert app.query_one("#resize-message").has_class("hidden")
            assert not app.query_one("#main").has_class("hidden")
            assert list(app.query("#side-bar")) == []
            assert "[Ctrl+Q]退出" in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert app.query_one("#resize-message").has_class("hidden")
            assert not app.query_one("#main").has_class("hidden")
            assert list(app.query("#side-bar")) == []
            assert "Ctrl+Enter 换行" in _text(app, "#conversation")

    asyncio.run(run_too_small())
    asyncio.run(run_80())
    asyncio.run(run_120())

def test_tui_profile_missing_shows_setup_message(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            profile_name=None,
            provider=None,
            model=None,
            api_key_env=None,
            api_key_available=False,
            profile_error="未找到默认模型配置，请运行 haagent 后在 TUI 内输入 /connect 配置供应商",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = _text(app, "#conversation")
            assert "未找到默认模型配置" in conversation
            assert "/connect" in conversation
            assert "uv run haagent setup" not in conversation

    asyncio.run(run())

def test_tui_api_key_missing_shows_env_name(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            api_key_available=False,
            credential_source_used=None,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "key:" not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert "DEEPSEEK_API_KEY" in conversation
            assert "/connect" in conversation
            assert "不会在 TUI 中输入" not in conversation

    asyncio.run(run())

def test_tui_api_key_available_via_env(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            credential_source_used="env",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "key:" not in status
            assert "Ctrl+Enter 换行" in conversation

    asyncio.run(run())

def test_tui_keyring_unavailable_shows_reason(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            api_key_available=False,
            credential_source_used=None,
            credential_store_available=False,
            credential_store_error="backend unavailable",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = _text(app, "#conversation")
            assert "系统凭据库不可用：backend unavailable" in conversation
            assert "/connect" in conversation
            assert "uv run haagent setup" not in conversation

    asyncio.run(run())

def test_tui_ctrl_q_exits_even_when_prompt_input_is_focused(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#prompt-input").has_focus
            await pilot.press("ctrl+q")
            await pilot.pause(0.1)
            assert not app.is_running

    asyncio.run(run())

def test_tui_q_does_not_exit_while_prompt_input_is_focused(tmp_path: Path) -> None:
    async def run_empty_input() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            assert input_widget.has_focus
            await pilot.press("q")
            await pilot.pause(0.1)
            assert app.is_running
            assert input_widget.value == "q"

    async def run_existing_input() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "abc"
            await pilot.press("q")
            await pilot.pause(0.1)
            assert app.is_running
            assert input_widget.value == "abcq"

    asyncio.run(run_empty_input())
    asyncio.run(run_existing_input())

def test_tui_text_area_ctrl_enter_inserts_newline_and_enter_submits_prompt(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Summarize this folder"
            await pilot.press("ctrl+enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert input_widget.value == "Summarize this folder\n"
            input_widget.value = "Summarize this folder\nwith constraints"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts == ["Summarize this folder\nwith constraints"]
            assert input_widget.value == ""
            conversation = _text(app, "#conversation")
            assert "[你]" not in conversation
            assert "Summarize this folder" in conversation
            assert "with constraints" in conversation
            assert "assistant: Summarize this folder\nwith constraints" in conversation

    asyncio.run(run())


def test_tui_text_area_blank_input_does_not_submit(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "  \n  "
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert input_widget.value == "  \n  "

    asyncio.run(run())

def test_tui_running_task_rejects_plain_submit_and_keeps_input(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            input_widget.value = "Second task"
            await pilot.press("enter")
            await pilot.pause(0.2)

            try:
                assert service.prompts == ["Long task"]
                assert input_widget.value == "Second task"
                conversation = _text(app, "#conversation")
                assert "当前任务仍在运行，请等待或使用 /cancel" not in conversation
                assert "[命令]" not in conversation
                assert "Second task" not in conversation
            finally:
                service.release.set()
                await pilot.pause(0.2)

    asyncio.run(run())

def test_tui_layout_sizes_do_not_render_side_bar(tmp_path: Path) -> None:
    async def run_80() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            assert list(app.query("#side-bar")) == []
            assert "/tools" not in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert list(app.query("#side-bar")) == []
            assert not app.query_one("#main").has_class("hidden")

    async def run_200() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(200, 60)):
            assert list(app.query("#side-bar")) == []
            assert "Ctrl+Enter 换行" in _text(app, "#conversation")

    asyncio.run(run_80())
    asyncio.run(run_120())
    asyncio.run(run_200())

def test_tui_no_color_timeline_keeps_structure_and_failure_labels(tmp_path: Path, monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[_tool_event("tool_failed", 1, "web_fetch", message="404")],
            assistant_content="已尝试获取网页。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "获取网页"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "[你]" not in conversation
            assert "[HaAgent]" not in conversation
            assert "已完成" in conversation and "步" in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            conversation = _text(app, "#conversation")
            assert "! 读取网页失败" in conversation
            assert "web_fetch" not in conversation
            assert "失败" in conversation

    asyncio.run(run())

def test_tui_running_state_does_not_block_ui(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            assert "正在工作" in _text(app, "#status-bar")
            assert app.query_one("#prompt-input") is input_widget
            service.release.set()
            await pilot.pause(0.2)
            assert "空闲" in _text(app, "#status-bar")

    asyncio.run(run())

