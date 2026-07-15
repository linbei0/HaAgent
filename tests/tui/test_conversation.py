"""
tests/tui/test_conversation.py - HaAgent TUI conversation 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from haagent.runtime.events import FailureNoticeEvent
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.commands import command_registry
from haagent.tui.state.search import ConversationSearchState
from haagent.tui.widgets import ConversationTimeline, PromptInput
from haagent.tui.typography.wrap import is_textual_line_breaking_installed
from textual.widgets import Markdown

from tests.tui.support import (
    FakeAssistantService,
    _all_text,
    _approval_request,
    _assistant_event,
    _runtime_event,
    _session_summary,
    _text,
    _tool_event,
    _wait_for_conversation_bottom,
)

def test_tui_installs_unicode_line_breaking_for_assistant_markdown(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        assistant_content="各地举行建国 250 周年庆祝活动，约 400 架无人机参与。",
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(42, 24)) as pilot:
            assert is_textual_line_breaking_installed()
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "看看新闻"
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert "建国 250 周年" in _text(app, "#conversation")
            rendered = app.query_one(Markdown).render()
            rendered_text = getattr(rendered, "plain", str(rendered))
            assert "250\n周年" not in rendered_text
            assert "400\n架" not in rendered_text

    asyncio.run(run())

def test_tui_conversation_search_state_tracks_matches_and_navigation() -> None:
    state = ConversationSearchState(["You\n  inspect docs", "Tool file_read done", "Assistant\n  docs ready"])

    matched = state.update_query("docs")
    second = state.next_match()
    first = state.previous_match()
    empty = state.update_query("missing")

    assert matched.count == 2
    assert matched.current_line == 0
    assert second.current_line == 2
    assert first.current_line == 0
    assert empty.count == 0
    assert empty.status_text == "无匹配：missing"

def test_tui_help_uses_modal_without_polluting_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            before = _text(app, "#conversation")
            await pilot.press("?")
            await pilot.pause(0.1)
            after = _text(app, "#conversation")
            rendered = _all_text(app)
            assert after == before
            assert "HaAgent 帮助" in rendered
            assert "聊天模式" in rendered
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "HaAgent 帮助" not in _all_text(app)

    asyncio.run(run())

def test_tui_search_overlay_finds_conversation_and_does_not_pollute_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            app._conversation.append_block("Assistant", "Alpha docs\nBeta docs")
            app._conversation.append_line("Tool file_read done")
            app._refresh_conversation()
            await pilot.pause()
            before = _text(app, "#conversation")

            await pilot.press("ctrl+f")
            await pilot.pause(0.1)
            await pilot.press("d", "o", "c", "s")
            await pilot.pause(0.1)
            assert "范围: conversation" in _all_text(app)
            assert "1/2" in _all_text(app)
            await pilot.press("n")
            await pilot.pause(0.1)
            assert "2/2" in _all_text(app)
            await pilot.press("shift+n")
            await pilot.pause(0.1)
            assert "1/2" in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert _text(app, "#conversation") == before

            await pilot.press("ctrl+f")
            await pilot.press("x", "x", "x")
            await pilot.pause(0.1)
            assert "无匹配" in _all_text(app)

    asyncio.run(run())

def test_tui_slash_command_suggestions_filter_execute_and_do_not_pollute_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, sessions=[_session_summary(tmp_path, "session-old", "继续任务")])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            before = _text(app, "#conversation")

            await pilot.press("/")
            await pilot.pause(0.1)
            assert "快捷命令" in _all_text(app)
            assert "/help" in _all_text(app)
            suggestions = app.query_one("#command-suggestions-dialog").state.visible_commands
            assert any(command.token == "/memory" for command in suggestions)
            assert "/resume" not in _all_text(app)
            assert app.command_suggestions_is_open()
            assert app.query_one("#command-suggestions-dialog").parent.id == "input-panel"
            assert app.query_one("#prompt-input").value == "/"
            await pilot.press("h", "e")
            await pilot.pause(0.1)
            assert "过滤: /he" in _all_text(app)
            assert "/help" in _all_text(app)
            assert "/resume" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "HaAgent 帮助" in _all_text(app)
            assert _text(app, "#conversation") == before
            await pilot.press("escape")
            await pilot.pause(0.1)

            await pilot.press("/")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" in _all_text(app)
            assert service.prompts == []
            await pilot.press("escape")
            await pilot.pause(0.1)

            input_widget = app.query_one("#prompt-input")
            input_widget.value = "整理 /tmp"
            await pilot.press("/")
            await pilot.pause(0.1)
            assert not app.command_suggestions_is_open()
            assert app.query_one("#prompt-input").value == "整理 /tmp/"

            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/new"
            await pilot.press("enter")
            await pilot.pause(0.1)
            input_widget.value = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.1)

            input_widget.value = "/unknown"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert service.created_sessions == ["session-new-1"]
            assert service.continued_latest_count == 1
            assert "未知命令：/unknown" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_slash_command_suggestions_scroll_visible_window() -> None:
    from haagent.tui.commands.suggestions import CommandSuggestionState

    state = CommandSuggestionState(commands=command_registry().commands())
    for _ in range(8):
        state = state.move(1)

    rendered = state.render()

    assert "> " in rendered
    assert any(command.name in rendered for command in command_registry().commands())
    assert "/help" not in rendered
    # 可见窗口随 slash 命令总数变化；只断言焦点行在窗口内且唯一
    assert rendered.count("> ") == 1
    focused = next(
        (line for line in rendered.splitlines() if line.startswith("> ")),
        "",
    )
    assert focused.startswith("> /")

def test_tui_tool_events_and_failure_stay_visible_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            interaction_request=_approval_request({"command": "uv run pytest -q", "cwd": ".", "timeout_seconds": 30}),
            extra_events=[
                _runtime_event(
                    "tool_started",
                    1,
                    tool_name="file_write",
                    args={"path": "notes.md", "mode": "create"},
                ),
                _runtime_event(
                    "tool_finished",
                    1,
                    tool_name="file_write",
                    result={"status": "success", "path": str(tmp_path / "notes.md"), "mode": "create", "bytes_written": 12, "created": True},
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "写一个 notes"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("n")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert list(app.query("#side-bar")) == []
            # 审批状态保持可见；侧效与工具失败进过程组，折叠后显示步骤数与耗时。
            assert "已完成" in conversation and "步" in conversation
            assert "已写入文件" not in conversation
            assert "运行命令失败" not in conversation
            assert "需要确认：运行命令" in conversation
            assert "已拒绝：运行命令" in conversation
            assert "file_write" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert app.query(".timeline-effect")
            assert "已写入文件" in conversation
            assert "文件已写入" in conversation
            assert "运行命令失败" in conversation or "写入文件" in conversation
            assert "步骤" not in conversation
            assert "过程" not in conversation
            assert "工具 1 项" not in conversation
            assert "1 失败" not in conversation
            assert "file_write" not in conversation
            assert "shell" not in conversation
            assert "需要确认：运行命令" in conversation
            assert "已拒绝：运行命令" in conversation
            assert "查看工具详情" not in conversation
            assert "任务工作台" not in conversation
            assert "工具时间线" not in conversation

    asyncio.run(run())

def test_tui_conversation_auto_scrolls_to_latest_content(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._conversation.append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            assert conversation.max_scroll_y > 0
            assert conversation.scroll_y == conversation.max_scroll_y
            assert "line 29" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_conversation_does_not_auto_scroll_when_user_reads_history(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._conversation.append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            assert conversation.max_scroll_y > 0

            conversation.scroll_to(y=0, animate=False, force=True)
            await pilot.pause()
            app._conversation.append_block("Assistant", "new line while reading")
            app._refresh_conversation()
            await pilot.pause()

            assert conversation.scroll_y == 0

    asyncio.run(run())

def test_tui_end_key_scrolls_conversation_to_bottom_when_prompt_is_empty(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._conversation.append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            conversation.scroll_to(y=0, animate=False, force=True)
            await pilot.pause()

            await pilot.press("end")
            await pilot.pause()

            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())

def test_tui_end_key_keeps_prompt_cursor_behavior_when_prompt_has_text(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._conversation.append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            conversation.scroll_to(y=0, animate=False, force=True)
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "abc"
            input_widget.cursor_location = (0, 0)
            await pilot.pause()

            await pilot.press("end")
            await pilot.pause()

            assert conversation.scroll_y == 0
            assert input_widget.cursor_location == (0, 3)

    asyncio.run(run())

def test_tui_conversation_wraps_long_messages_for_scroll_height(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            conversation = app.query_one("#conversation")
            long_reply = (
                "# 我能做什么？ 我是 **HaAgent**，一个运行在当前工作目录下的本地个人 AI 助手。"
                "我可以读取文件、整理内容、编辑文档、分析项目、运行命令，并支持多轮对话。"
            ) * 8
            app._conversation.append_block("Assistant", long_reply)
            app._refresh_conversation()
            await pilot.pause()
            answer = conversation.query_one(".timeline-body")
            assert answer.region.width <= conversation.content_size.width
            assert conversation.virtual_size.height > len(app._conversation.lines)
            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())

def test_tui_assistant_body_renders_markdown_without_opening_links(tmp_path: Path) -> None:
    async def run() -> None:
        markdown_reply = (
            "# 结果\n\n"
            "- **重点**\n"
            "- `inline code`\n\n"
            "| 文件 | 状态 |\n"
            "| --- | --- |\n"
            "| README.md | 已读 |\n\n"
            "```python\nprint('ok')\n```\n\n"
            "[资料](https://example.com)"
        )
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content=markdown_reply)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "总结资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body")
            assert isinstance(assistant_body, Markdown)
            assert getattr(assistant_body, "_open_links") is False
            table_cells = [
                widget
                for widget in assistant_body.walk_children()
                if widget.has_class("cell") or widget.has_class("header")
            ]
            assert table_cells
            assert all(widget.tooltip is None for widget in table_cells)
            assert markdown_reply in conversation.plain_text

    asyncio.run(run())

def test_tui_assistant_final_message_replaces_streamed_markdown(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _assistant_event("assistant_delta", 1, "# 草稿"),
            ],
            assistant_content="# 最终\n\n- 完成",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "生成清单"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            assert assistant_body.source == "# 最终\n\n- 完成"
            assert "# 草稿" not in conversation.plain_text
            assert "# 最终\n\n- 完成" in conversation.plain_text

    asyncio.run(run())

def test_tui_streamed_markdown_table_cells_do_not_show_tooltips(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def _run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| 来源 | 链接 |\n"))
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| --- | --- |\n"))
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| TASS | tass.com |\n"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "生成表格"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)
            await pilot.pause(0.3)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            table_cells = [
                widget
                for widget in assistant_body.walk_children()
                if widget.has_class("cell") or widget.has_class("header")
            ]
            assert table_cells
            assert all(widget.tooltip is None for widget in table_cells)
            release_event.set()
            await pilot.pause(0.2)

    asyncio.run(run())

def test_tui_merges_assistant_delta_events_into_single_response_block(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _assistant_event("assistant_delta", 1, "Ha"),
                _assistant_event("assistant_delta", 1, "Agent"),
            ],
            assistant_content="HaAgent",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍能力"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "[HaAgent]" not in conversation
            assert conversation.count("HaAgent") == 1

    asyncio.run(run())

def test_tui_assistant_delta_updates_current_turn_without_overwriting_previous_turn(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="第一轮完整回答")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "第一问"
            await pilot.press("enter")
            await pilot.pause(0.2)

            app._handle_chat_event(_assistant_event("assistant_delta", 2, "第二"))
            app._handle_chat_event(_assistant_event("assistant_delta", 2, "轮"))
            app._handle_chat_event(_assistant_event("assistant_message", 2, "第二轮完整回答"))
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "第一轮完整回答" in conversation
            assert "第二轮完整回答" in conversation
            assert conversation.index("第一轮完整回答") < conversation.index("第二轮完整回答")

    asyncio.run(run())

def test_tui_final_message_on_new_turn_closes_previous_streaming_turn(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            app._handle_chat_event(_assistant_event("assistant_delta", 2, "worker 已启动，请稍候"))
            app._handle_chat_event(_assistant_event("assistant_message", 3, "worker 已完成，结论如下"))
            await asyncio.sleep(0)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_items = [item for item in conversation._items if item.role == "assistant"]
            assert [(item.turn_index, item.status) for item in assistant_items] == [
                (2, "done"),
                (3, "done"),
            ]

    asyncio.run(run())

def test_tui_streaming_turn_keeps_existing_widget_instances_stable(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def _run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "正在"))
                event_sink(_tool_event("tool_started", len(self.prompts), "file_read"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            if event_sink is not None:
                event_sink(_assistant_event("assistant_message", len(self.prompts), "最终回答"))
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_blocks = list(conversation.query(".timeline-assistant"))
            assert assistant_blocks
            assistant_block = assistant_blocks[-1]
            children_before = list(conversation.children)

            rendered = _text(app, "#conversation")
            assert "正在阅读文件..." in _text(app, "#progress-status")
            assert "file_read" not in rendered
            assert "生成中" not in rendered
            assert list(conversation.query(".timeline-assistant"))[-1] is assistant_block
            assert list(conversation.children) == children_before

            release_event.set()
            await pilot.pause(0.2)
            assert "最终回答" in _text(app, "#conversation")

    asyncio.run(run())

def test_tui_streaming_assistant_shows_rotating_cursor_and_cleans_up(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def _run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "正在整理"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            if event_sink is not None:
                event_sink(_assistant_event("assistant_message", len(self.prompts), "最终回答"))
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            active = conversation.query_one(".timeline-assistant .timeline-active")
            first_frame = str(active.content)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            assert first_frame in {"|", "/", "-", "\\"}
            assert assistant_body.source == "正在整理"
            assert "生成中" not in conversation.plain_text

            await pilot.pause(0.3)
            second_frame = str(active.content)
            assert second_frame in {"|", "/", "-", "\\"}
            assert second_frame != first_frame
            assert assistant_body.source == "正在整理"
            assert len(list(conversation.query(".timeline-assistant"))) == 1

            release_event.set()
            await pilot.pause(0.2)
            assert str(active.content) == ""
            assert active.display is False
            assert assistant_body.source == "最终回答"

    asyncio.run(run())

def test_tui_streaming_cursor_keeps_text_semantics_in_no_color_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def _run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "处理中"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            return SimpleNamespace(status="cancelled")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            active = conversation.query_one(".timeline-assistant .timeline-active")
            frame, label = str(active.content).split(" ", maxsplit=1)
            assert frame in {"|", "/", "-", "\\"}
            assert label == "生成中"
            assert "处理中" in conversation.plain_text
            assert "生成中" not in conversation.plain_text

            release_event.set()
            await pilot.pause(0.2)
            assert str(active.content) == ""
            assert active.display is False

    asyncio.run(run())

def test_tui_user_and_assistant_blocks_keep_spacing_and_alignment(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="已经完成。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "帮我总结"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            user = conversation.query_one(".timeline-user")
            assistant = conversation.query_one(".timeline-assistant")
            assert user.region.x <= conversation.region.x + 4
            assert assistant.region.x <= conversation.region.x + 4
            assert assistant.region.y >= user.region.y + user.region.height
            assert assistant.region.y > user.region.y

    asyncio.run(run())


def test_tui_conversation_uses_asymmetric_quiet_layout(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="这是回答。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "这是问题"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            user = conversation.query_one(".timeline-user")
            assistant = conversation.query_one(".timeline-assistant")

            assert user.query_one(".timeline-header").display is False
            assert assistant.query_one(".timeline-header").display is False
            assert user.styles.background != assistant.styles.background
            assert assistant.styles.background == conversation.styles.background
            assert user.styles.border_left[0] == "solid"
            assert not assistant.styles.border_left[0]
            assert user.region.x == assistant.region.x
            assert user.region.width == assistant.region.width
            assert input_widget.region.x == user.region.x
            assert input_widget.region.width == user.region.width
            assert input_widget.content_region.x == user.content_region.x

    asyncio.run(run())

def test_tui_conversation_scrollbar_states_do_not_use_black_track(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="已经完成。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "帮我总结"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assert conversation.styles.scrollbar_background == conversation.styles.background
            assert conversation.styles.scrollbar_background_hover == conversation.styles.background
            assert conversation.styles.scrollbar_background_active == conversation.styles.background
            assert conversation.styles.scrollbar_size_vertical == 0
            assert conversation.styles.scrollbar_size_horizontal == 0
            assert conversation.styles.scrollbar_visibility == "hidden"

    asyncio.run(run())

def test_tui_renders_full_long_assistant_message_from_event_sink(tmp_path: Path) -> None:
    async def run() -> None:
        long_reply = ("HaAgent 可以读取文件、整理内容、编辑文档、分析项目。" * 40) + "完整结尾"
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content=long_reply)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍能力"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "[truncated]" not in conversation
            assert "完整结尾" in conversation
            await _wait_for_conversation_bottom(app, pilot)
            assert app.query_one("#conversation").scroll_y == app.query_one("#conversation").max_scroll_y

    asyncio.run(run())

def test_tui_conversation_text_is_read_only_and_selectable(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = app.query_one("#conversation")
            assert isinstance(conversation, ConversationTimeline)
            app._conversation.append_block("Assistant", "这段回复应该可以被选中复制")
            app._refresh_conversation()
            assert "这段回复应该可以被选中复制" in conversation.plain_text

    asyncio.run(run())

def test_tui_failure_event_shows_reason_episode_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        episode_path = tmp_path / ".runs" / "episode-failed"
        service = FakeAssistantService(
            workspace_root=tmp_path,
            failure_event=FailureNoticeEvent(
                session_id="session-test",
                turn_index=1,
                status="failed",
                failed_stage="executing",
                failure_category="Loop Limit Failure",
                reason="exceeded max_turns=20",
                episode_path=str(episode_path),
            ),
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍一下项目"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "本轮没有完成：模型连续调用工具但没有给出最终回答。" in conversation
            assert "stage=executing" in conversation
            assert "category=Loop Limit Failure" in conversation
            assert "reason=exceeded max_turns=20" in conversation
            assert str(episode_path) in conversation.replace("\n", "")
            status = _text(app, "#status-bar")
            assert "失败" in status
            assert "state:" not in status
            assert list(app.query("#side-bar")) == []
            assistant = next(item for item in app._timeline()._items if item.role == "assistant")
            assert assistant.status == "done"

    asyncio.run(run())

def test_tui_shows_assistant_placeholder_immediately_after_submit(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Search slowly"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            conversation = _text(app, "#conversation")
            assert "Search slowly" in conversation
            assert "[你]" not in conversation
            assert "[HaAgent]" not in conversation
            assert len(list(app.query(".timeline-user"))) == 1
            assert len(list(app.query(".timeline-assistant"))) == 1

            service.release.set()
            await pilot.pause(0.2)

    asyncio.run(run())

