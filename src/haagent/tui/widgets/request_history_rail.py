"""
src/haagent/tui/widgets/request_history_rail.py - 用户请求历史轨道

把当前会话的用户请求压缩映射为可悬浮、可点击和可键盘访问的导航刻度。
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static


@dataclass(frozen=True)
class RequestHistoryEntry:
    turn_index: int
    request_summary: str
    answer_summary: str


@dataclass(frozen=True)
class RequestHistoryGroup:
    row: int
    entries: tuple[RequestHistoryEntry, ...]


class RequestHistoryPreview(Static):
    """在主对话层上显示当前悬浮请求的紧凑预览。"""


def group_request_history_entries(
    entries: list[RequestHistoryEntry],
    *,
    height: int,
) -> list[RequestHistoryGroup]:
    """将全部请求等比映射到有限轨道高度，同一行自然聚合。"""

    if not entries or height <= 0:
        return []
    rows: dict[int, list[RequestHistoryEntry]] = {}
    track_height = min(len(entries), height)
    row_offset = max(0, (height - track_height) // 2)
    last_row = max(0, track_height - 1)
    denominator = max(1, len(entries) - 1)
    for index, entry in enumerate(entries):
        row = row_offset + round(index * last_row / denominator)
        rows.setdefault(row, []).append(entry)
    return [RequestHistoryGroup(row=row, entries=tuple(group)) for row, group in sorted(rows.items())]


class RequestHistoryRail(Widget):
    can_focus = True
    BINDINGS = [
        Binding("up", "previous_group_entry", "上一条", show=False),
        Binding("down", "next_group_entry", "下一条", show=False),
        Binding("enter", "activate_group_entry", "跳转", show=False),
        Binding("escape", "leave_group", "返回输入框", show=False),
    ]

    class Navigate(Message):
        def __init__(self, turn_index: int, *, keep_focus: bool) -> None:
            self.turn_index = turn_index
            self.keep_focus = keep_focus
            super().__init__()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._entries: list[RequestHistoryEntry] = []
        self._groups: list[RequestHistoryGroup] = []
        self._active_turn: int | None = None
        self._hovered_group: int | None = None
        self._selected_by_group: dict[int, int] = {}
        self._preview: RequestHistoryPreview | None = None

    def set_entries(self, entries: list[RequestHistoryEntry], *, active_turn: int | None) -> None:
        layout_changed = [entry.turn_index for entry in entries] != [entry.turn_index for entry in self._entries]
        self._entries = list(entries)
        self._active_turn = active_turn
        self.display = len(entries) >= 2
        if layout_changed:
            self._selected_by_group.clear()
        self._rebuild_groups()
        self._update_preview()
        self.refresh(layout=layout_changed)

    def set_active_turn(self, turn_index: int | None) -> None:
        if turn_index == self._active_turn:
            return
        self._active_turn = turn_index
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        self._rebuild_groups()
        self._position_preview()

    def render(self) -> Text:
        height = max(1, self.content_size.height or self.size.height)
        lines = ["" for _ in range(height)]
        for group_index, group in enumerate(self._groups):
            is_active = any(entry.turn_index == self._active_turn for entry in group.entries)
            is_hovered = group_index == self._hovered_group
            if len(group.entries) > 1:
                count_label = str(len(group.entries)) if len(group.entries) < 10 else "+"
                marker = f"━━{count_label}" if is_active or is_hovered else f"━{count_label}"
            elif is_active:
                marker = "━━━━"
            elif is_hovered:
                marker = "━━━"
            else:
                marker = "━"
            style = "bold" if is_active or is_hovered else "dim"
            lines[group.row] = Text(marker, style=style)
        rendered = Text()
        for index, line in enumerate(lines):
            if index:
                rendered.append("\n")
            rendered.append_text(line if isinstance(line, Text) else Text(line))
        return rendered

    def on_mouse_move(self, event: events.MouseMove) -> None:
        group_index = self._group_index_at_row(self._content_row(event.y))
        if group_index == self._hovered_group:
            return
        self._hovered_group = group_index
        self._update_preview()
        self._position_preview()
        self.refresh()

    def on_leave(self, event: events.Leave) -> None:
        if self.has_focus:
            return
        self._hovered_group = None
        self._hide_preview()
        self.refresh()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button != 1 or event.shift:
            return
        group_index = self._group_index_at_row(self._content_row(event.y))
        if group_index is None:
            return
        event.stop()
        self._hovered_group = group_index
        entry = self._selected_entry(group_index)
        is_grouped = len(self._groups[group_index].entries) > 1
        self.post_message(self.Navigate(entry.turn_index, keep_focus=is_grouped))
        if is_grouped:
            self.focus()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self._cycle_hovered_group(-1):
            event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self._cycle_hovered_group(1):
            event.stop()

    def action_previous_group_entry(self) -> None:
        self._cycle_focused_group(-1)

    def action_next_group_entry(self) -> None:
        self._cycle_focused_group(1)

    def action_activate_group_entry(self) -> None:
        if self._hovered_group is None:
            return
        self.post_message(
            self.Navigate(
                self._selected_entry(self._hovered_group).turn_index,
                keep_focus=True,
            ),
        )

    def action_leave_group(self) -> None:
        self._hovered_group = None
        self._hide_preview()
        focus_prompt = getattr(self.app, "focus_prompt_input", None)
        if callable(focus_prompt):
            focus_prompt()
        self.refresh()

    def _rebuild_groups(self) -> None:
        height = max(1, self.content_size.height or self.size.height)
        self._groups = group_request_history_entries(self._entries, height=height)
        if self._hovered_group is not None and self._hovered_group >= len(self._groups):
            self._hovered_group = None

    def _group_index_at_row(self, row: int) -> int | None:
        if not self._groups:
            return None
        return min(range(len(self._groups)), key=lambda index: abs(self._groups[index].row - row))

    def _content_row(self, event_y: float) -> int:
        return max(0, int(event_y) - int(self.styles.padding.top))

    def _selected_entry(self, group_index: int) -> RequestHistoryEntry:
        group = self._groups[group_index]
        selected = min(self._selected_by_group.get(group_index, 0), len(group.entries) - 1)
        return group.entries[selected]

    def _cycle_hovered_group(self, delta: int) -> bool:
        group_index = self._hovered_group
        if group_index is None or len(self._groups[group_index].entries) < 2:
            return False
        self._cycle_group(group_index, delta)
        return True

    def _cycle_focused_group(self, delta: int) -> None:
        if self._hovered_group is None:
            return
        self._cycle_group(self._hovered_group, delta)

    def _cycle_group(self, group_index: int, delta: int) -> None:
        group = self._groups[group_index]
        current = self._selected_by_group.get(group_index, 0)
        self._selected_by_group[group_index] = (current + delta) % len(group.entries)
        self._update_preview()
        self._position_preview()
        self.refresh()

    def _update_preview(self) -> None:
        group_index = self._hovered_group
        if group_index is None or group_index >= len(self._groups):
            self._hide_preview()
            return
        group = self._groups[group_index]
        selected_index = self._selected_by_group.get(group_index, 0) % len(group.entries)
        entry = group.entries[selected_index]
        position = f"{selected_index + 1}/{len(group.entries)}\n" if len(group.entries) > 1 else ""
        preview = self._preview
        if preview is None:
            return
        content = Text()
        if position:
            content.append(position, style="dim")
        content.append(entry.request_summary, style="bold")
        content.append("\n────────\n", style="dim")
        content.append(entry.answer_summary, style="dim")
        preview.update(content)
        preview.display = True

    def _hide_preview(self) -> None:
        if self._preview is not None:
            self._preview.display = False

    def _position_preview(self) -> None:
        preview = self._preview
        if preview is None or self._hovered_group is None or not self._groups:
            return
        preview.styles.offset = (
            4,
            max(0, self._groups[self._hovered_group].row + int(self.styles.padding.top) - 2),
        )

    def on_mount(self) -> None:
        self._bind_preview()

    def _bind_preview(self) -> None:
        self._preview = self.app.query_one("#request-history-preview", RequestHistoryPreview)
        self._preview.display = False
