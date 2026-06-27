"""
haagent/tui/widgets.py - TUI 基础组件

封装输入、状态栏、对话区、侧栏、footer 和尺寸提示等稳定 UI 区域。
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.widgets import Input, RichLog, Static


class PromptInput(Input):
    def on_key(self, event: events.Key) -> None:
        if self.value:
            return
        app = self.app
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            app.action_help()
            return
        if event.key != "m" or getattr(app, "_pending_interaction", None) is not None:
            return
        event.stop()
        app.action_toggle_memory()


class StatusBar(Static):
    def update_status(self, text: str) -> None:
        self.update(text)


class ConversationView(RichLog):
    def show_placeholder(self) -> None:
        self.clear()
        self.write(Text("Ready. 输入 prompt 后按 Enter 发送；Ctrl+Q 退出。"), scroll_end=True, animate=False)

    def show_memory(self, text: str) -> None:
        self.clear()
        self.write(Text(text), scroll_end=True, animate=False)

    def append_lines(self, lines: list[str], *, start: int) -> None:
        for line in lines[start:]:
            self.write(Text(line), scroll_end=True, animate=False)


class SideBar(Static):
    def update_content(self, text: str) -> None:
        self.update(text)


class FooterBar(Static):
    def update_footer(self, text: str) -> None:
        self.update(Text(text))


class ResizeMessage(Static):
    pass
