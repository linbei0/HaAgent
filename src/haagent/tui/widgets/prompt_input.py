"""
src/haagent/tui/widgets/prompt_input.py - TUI 提示输入组件

封装多行提示输入框和光标定位辅助函数，处理 slash 命令、文件引用和记忆快捷键。
"""

from __future__ import annotations

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit_from_input", "发送", priority=True),
        Binding("shift+enter", "insert_newline_from_input", "换行", priority=True),
        Binding("ctrl+f", "open_search_from_input", "搜索", priority=True),
        Binding("ctrl+x", "cancel_current_task_from_input", "取消任务", priority=True),
        Binding("ctrl+v", "paste_image_from_input", "粘贴图片", priority=True),
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
        memory_flow = getattr(app, "memory_flow", None)
        if memory_flow is not None and memory_flow.mode and getattr(app, "_pending_interaction", None) is None:
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
                handled = app.memory_flow.handle_key(event.key)
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

    def action_paste_image_from_input(self) -> None:
        self.app.action_paste_image_from_input()

    def action_submit_from_input(self) -> None:
        if getattr(self.app, "command_suggestions_is_open", lambda: False)():
            self.app.action_accept_command_suggestion()
            return
        if getattr(self.app, "file_reference_is_open", lambda: False)():
            self.app.action_accept_file_ref()
            return
        memory_flow = getattr(self.app, "memory_flow", None)
        if memory_flow is not None and memory_flow.mode:
            self.app.action_memory_enter()
            return
        self.app.action_submit_prompt()

    def action_insert_newline_from_input(self) -> None:
        self.insert("\n")


def _end_location(text: str) -> tuple[int, int]:
    lines = text.split("\n")
    return (len(lines) - 1, len(lines[-1]))
