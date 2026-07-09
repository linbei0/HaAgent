"""
src/haagent/tui/widgets/status.py - TUI 状态栏与页脚组件

封装顶部状态栏、进度状态行、底部页脚和终端尺寸提示。
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class StatusBar(Static):
    """顶部状态栏，展示 workspace/profile/model/session/运行状态。"""

    def update_status(self, text: str) -> None:
        self.update(text)


class ProgressStatusLine(Static):
    """输入区上方的进度状态行，有内容时显示，无内容时隐藏。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = False

    def update_status(self, text: str, *, severity: str = "info") -> None:
        self.display = bool(text)
        self.set_class(severity == "warning", "progress-warning")
        self.set_class(severity == "error", "progress-error")
        self.update(text)

    def clear(self) -> None:
        self.display = False
        self.update("")


class FooterBar(Static):
    """底部快捷键 footer，按上下文切换显示内容。"""

    def update_footer(self, text: str) -> None:
        self.update(Text(text))


class ResizeMessage(Static):
    """终端过小时展示的全屏提示，要求用户调大终端。"""

    pass
