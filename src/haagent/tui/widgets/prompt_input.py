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
        Binding("ctrl+enter", "insert_newline_from_input", "换行", priority=True),
        Binding("ctrl+j", "insert_newline_from_input", "换行", priority=True),
        Binding("ctrl+f", "open_search_from_input", "搜索", priority=True),
        Binding("ctrl+x", "cancel_current_task_from_input", "取消任务", priority=True),
        Binding("ctrl+v", "paste_image_from_input", "粘贴图片", priority=True),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._request_history: list[str] = []
        self._request_history_index: int | None = None
        self._recalled_request: str | None = None

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
        self._reset_request_history_navigation()
        self.load_text(text)
        self.move_cursor(_end_location(text))

    def set_request_history(self, requests: list[str]) -> None:
        self._request_history = list(requests)
        self._reset_request_history_navigation()

    def append_request_history(self, request: str) -> None:
        if not request:
            return
        self._request_history.append(request)
        self._reset_request_history_navigation()

    def clear_request_history(self) -> None:
        self._request_history.clear()
        self._reset_request_history_navigation()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is not self or self._request_history_index is None:
            return
        if self.value != self._recalled_request:
            self._reset_request_history_navigation()

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if app.command_suggestions_is_open() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_command_suggestion_key(event)
            return
        if app.file_reference_is_open() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_file_ref_key(event)
            return
        memory_flow = app.memory_flow
        if memory_flow.mode and app._pending_interaction is None:
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
        if (
            app._pending_interaction is None
            and event.key in {"up", "down"}
            and self._navigate_request_history(event.key)
        ):
            # 补全与交互模式优先；普通输入仅从空值启动历史，避免覆盖多行光标移动。
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
        if self.app.command_suggestions_is_open():
            self.app.action_accept_command_suggestion()
            return
        if self.app.file_reference_is_open():
            self.app.action_accept_file_ref()
            return
        if self.app.memory_flow.mode:
            self.app.action_memory_enter()
            return
        self.app.action_submit_prompt()

    def action_insert_newline_from_input(self) -> None:
        self.insert("\n")

    def _navigate_request_history(self, key: str) -> bool:
        if not self._request_history:
            return False
        if self._request_history_index is None:
            if key == "down" or self.value:
                return False
            self._show_request_history(len(self._request_history) - 1)
            return True
        if key == "up":
            self._show_request_history(max(0, self._request_history_index - 1))
            return True
        next_index = self._request_history_index + 1
        if next_index < len(self._request_history):
            self._show_request_history(next_index)
        else:
            self._reset_request_history_navigation()
            self.load_text("")
            self.move_cursor((0, 0))
        return True

    def _show_request_history(self, index: int) -> None:
        request = self._request_history[index]
        self._request_history_index = index
        self._recalled_request = request
        self.load_text(request)
        self.move_cursor(_end_location(request))

    def _reset_request_history_navigation(self) -> None:
        self._request_history_index = None
        self._recalled_request = None


def _end_location(text: str) -> tuple[int, int]:
    lines = text.split("\n")
    return (len(lines) - 1, len(lines[-1]))
