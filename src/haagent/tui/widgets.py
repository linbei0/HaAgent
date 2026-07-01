"""
haagent/tui/widgets.py - TUI 基础组件

封装输入、状态栏、对话区、footer 和尺寸提示等稳定 UI 区域。
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static, TextArea


class PromptInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit_from_input", "发送", priority=True),
        Binding("shift+enter", "insert_newline_from_input", "换行", priority=True),
        Binding("ctrl+f", "open_search_from_input", "搜索", priority=True),
        Binding("ctrl+x", "cancel_current_task_from_input", "取消任务", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, input: PromptInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(_end_location(text))

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if getattr(app, "command_suggestions_is_open", lambda: False)() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_command_suggestion_key(event)
            return
        if getattr(app, "file_reference_is_open", lambda: False)() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_file_ref_key(event)
            return
        if getattr(app, "_memory_mode", False) and getattr(app, "_pending_interaction", None) is None:
            handled = False
            if event.key == "enter":
                app.action_memory_enter()
                handled = True
            elif event.key in {"a", "y"}:
                app.action_confirm_memory()
                handled = True
            elif event.key == "r":
                app.action_reject_memory()
                handled = True
            else:
                handled = app._handle_memory_key(event.key)
            if handled:
                event.stop()
                event.prevent_default()
                return
        if event.key == "@" or event.character == "@":
            event.stop()
            event.prevent_default()
            self.insert("@")
            app.action_open_file_refs()
            return
        if event.key in {"/", "slash"} or event.character == "/":
            event.stop()
            event.prevent_default()
            self.insert("/")
            if self.value == "/":
                app.action_open_command_suggestions()
            return
        if self.value:
            return
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            app.action_help()
            return

    def action_open_search_from_input(self) -> None:
        self.app.action_open_search()

    def action_cancel_current_task_from_input(self) -> None:
        self.app.action_cancel_current_task()

    def action_submit_from_input(self) -> None:
        if getattr(self.app, "command_suggestions_is_open", lambda: False)():
            self.app.action_accept_command_suggestion()
            return
        if getattr(self.app, "file_reference_is_open", lambda: False)():
            self.app.action_accept_file_ref()
            return
        if getattr(self.app, "_memory_mode", False):
            self.app.action_memory_enter()
            return
        self.app.action_submit_prompt()

    def action_insert_newline_from_input(self) -> None:
        self.insert("\n")


class StatusBar(Static):
    def update_status(self, text: str) -> None:
        self.update(text)


class ConversationView(TextArea):
    def __init__(self, *args, **kwargs) -> None:
        wrap = kwargs.pop("wrap", None)
        kwargs.pop("auto_scroll", None)
        if wrap is not None:
            kwargs.setdefault("soft_wrap", wrap)
        kwargs.setdefault("read_only", True)
        kwargs.setdefault("show_cursor", False)
        kwargs.setdefault("show_line_numbers", False)
        kwargs.setdefault("highlight_cursor_line", False)
        super().__init__(*args, **kwargs)

    def show_placeholder(self) -> None:
        self.load_text("Ready. 输入消息后按 Enter 发送；Shift+Enter 换行；Ctrl+Q 退出。")

    def show_memory(self, text: str) -> None:
        self.load_text(text)

    def append_lines(self, lines: list[str], *, start: int) -> None:
        new_text = "\n".join(lines[start:])
        if not new_text:
            return
        prefix = self.text
        if prefix:
            new_text = f"{prefix}\n{new_text}"
        self.load_text(new_text)


class FooterBar(Static):
    def update_footer(self, text: str) -> None:
        self.update(Text(text))


class ResizeMessage(Static):
    pass


def _end_location(text: str) -> tuple[int, int]:
    lines = text.split("\n")
    return (len(lines) - 1, len(lines[-1]))
    class Submitted(events.Message):
        def __init__(self, input: PromptInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()
