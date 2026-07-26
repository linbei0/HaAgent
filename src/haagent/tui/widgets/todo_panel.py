"""
src/haagent/tui/widgets/todo_panel.py - 只读 Todo 面板

在输入区原位展示 session 级清单，不向 timeline 写入每次更新。
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static


class TodoPanel(Static):
    can_focus = True
    DEFAULT_CSS = """
    TodoPanel {
        height: auto;
        color: $text-muted;
        background: $surface;
        border-left: thick $secondary;
        padding: 0 1;
    }
    TodoPanel:hover { background: $surface-lighten-1; }
    TodoPanel:focus { border-left: thick $primary; }
    """

    class ExpandedChanged(Message):
        """Todo 展开状态改变后，通知输入区重新计算高度。"""

        def __init__(self, expanded: bool) -> None:
            super().__init__()
            self.expanded = expanded

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.items: tuple[dict[str, object], ...] = ()
        self.expanded = False
        self.display = False

    def update_state(self, items: tuple[dict[str, object], ...]) -> None:
        self.items = tuple(dict(item) for item in items)
        self.display = any(item.get("status") in {"pending", "in_progress"} for item in self.items)
        if not self.display:
            self.expanded = False
        self.refresh(layout=True)

    def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "space"} and self.display:
            event.stop()
            event.prevent_default()
            self.toggle()

    def on_click(self, event: events.Click) -> None:
        if self.display:
            event.stop()
            self.toggle()

    def toggle(self) -> None:
        self.expanded = not self.expanded
        self.refresh(layout=True)
        self.post_message(self.ExpandedChanged(self.expanded))

    def layout_height(self) -> int:
        if not self.display:
            return 0
        if not self.expanded:
            return 2
        # 展开后必须保留完整清单；不能用“还有 N 项”代替用户看不到的任务。
        return len(self.items) + 2

    def render(self) -> Text:
        completed = sum(item.get("status") == "completed" for item in self.items)
        total = len(self.items)
        active = next((item for item in self.items if item.get("status") == "in_progress"), None)
        active_text = _compact_text(str(active.get("content", "")) if active is not None else "等待下一项", self._content_limit())
        text = Text("任务清单 ", style="bold")
        text.append(f"已完成 {completed}/{total}", style="dim")
        text.append(f" · 当前：{active_text}")
        text.append("  收起 ▴" if self.expanded else "  点击展开 ▾", style="dim")
        if not self.expanded:
            return text
        markers = {"pending": "○", "in_progress": "▶", "completed": "✓", "cancelled": "—"}
        labels = {"pending": "待处理", "in_progress": "进行中", "completed": "已完成", "cancelled": "已取消"}
        for item in self.items:
            status = str(item.get("status", "pending"))
            text.append(f"\n{markers.get(status, '?')} {_compact_text(str(item.get('content', '')), self._content_limit())}")
            text.append(f" · {labels.get(status, '未知')}", style="dim")
        return text

    def _content_limit(self) -> int:
        return max(24, self.size.width - 20)


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
