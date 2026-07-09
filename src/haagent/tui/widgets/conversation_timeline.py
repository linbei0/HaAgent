"""
src/haagent/tui/widgets/conversation_timeline.py - 对话时间线主组件

VerticalScroll 容器，维护 TimelineItem 列表并按需挂载/更新 TimelineBlock。
支持 streaming 批处理、工具活动延迟刷新和鼠标拖拽暂停交互更新。
"""

from __future__ import annotations

from itertools import count
from typing import Any

from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Static

from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem
from haagent.tui.widgets.timeline_models import (
    DETAILS_REFRESH_RECENT_TURNS,
    MARKDOWN_DELTA_FLUSH_INTERVAL_MS,
    SELECTION_RESUME_DELAY_MS,
    TOOL_ACTIVITY_FLUSH_INTERVAL_MS,
    TimelineItem,
    ToolActivity,
)
from haagent.tui.widgets.timeline_rendering import (
    render_timeline_item as _render_timeline_item,
)
from haagent.tui.widgets.timeline_block import (
    TimelineBlock,
    _is_process_group_id,
    _is_process_item,
    _presentation_role,
    _process_group_item,
    _process_group_turn_index,
)
from haagent.tui.widgets.tool_activity import matching_latest_tool_activity, merge_tool_activity


class ConversationTimeline(VerticalScroll):
    _ids = count(1)
    can_focus = True

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("wrap", None)
        kwargs.pop("auto_scroll", None)
        super().__init__(*args, **kwargs)
        self._items: list[TimelineItem] = []
        self._tool_details_enabled = False
        self._plain_text = ""
        self._plain_text_dirty = False
        self._blocks: dict[int, TimelineBlock] = {}
        self._placeholder_widget: Static | None = None
        self._memory_widget: Static | None = None
        self._stick_to_bottom = True
        self._pending_tool_item_ids: set[int] = set()
        self._tool_activity_flush_scheduled = False
        self._pending_assistant_delta_item_ids: set[int] = set()
        self._assistant_delta_flush_scheduled = False
        self._interactive_updates_paused = False
        self._mouse_down_location: tuple[float, float] | None = None
        self._selection_dragging = False
        self._expanded_process_turns: set[int] = set()

    def set_stick_to_bottom(self, enabled: bool) -> None:
        self._stick_to_bottom = enabled

    def scroll_end_if_sticky(self) -> None:
        if self._stick_to_bottom:
            self.scroll_end(animate=False)

    def scroll_to(self, x: float | None = None, y: float | None = None, **kwargs: Any) -> None:
        if y is not None:
            self._stick_to_bottom = y >= self.max_scroll_y - 1
            if not self._stick_to_bottom:
                self.pause_interactive_updates()
        super().scroll_to(x=x, y=y, **kwargs)

    def show_placeholder(self) -> None:
        self._items.clear()
        self._plain_text = "Ready. 输入消息后按 Enter 发送；Shift+Enter 换行；Ctrl+Q 退出。"
        self._plain_text_dirty = False
        self._show_singleton("placeholder", "Ready. 输入消息后按 Enter 发送；Shift+Enter 换行；Ctrl+Q 退出。", "timeline-placeholder")

    def show_memory(self, text: str) -> None:
        self._items.clear()
        self._plain_text = text
        self._plain_text_dirty = False
        self._show_singleton("memory", text, "timeline-memory")

    def append_lines(self, lines: list[str], *, start: int) -> None:
        for line in lines[start:]:
            self.add_system("系统", line)

    @property
    def plain_text(self) -> str:
        if self._plain_text_dirty:
            self._sync_plain_text()
        return self._plain_text

    def clear_timeline(self) -> None:
        self._items.clear()
        self._plain_text = ""
        self._plain_text_dirty = False
        self._pending_tool_item_ids.clear()
        self._tool_activity_flush_scheduled = False
        self._pending_assistant_delta_item_ids.clear()
        self._assistant_delta_flush_scheduled = False
        self._interactive_updates_paused = False
        self._mouse_down_location = None
        self._selection_dragging = False
        self._expanded_process_turns.clear()
        self._remove_singletons()
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()

    def set_tool_details(self, enabled: bool) -> None:
        if self._tool_details_enabled == enabled:
            return
        self._tool_details_enabled = enabled
        recent_assistants = [item for item in self._items if item.role == "assistant"][-DETAILS_REFRESH_RECENT_TURNS:]
        if not recent_assistants:
            self._render_timeline()
            return
        for item in recent_assistants:
            self._sync_block(item)
        self._mark_plain_text_dirty()

    def add_user(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="user", turn_index=turn_index, content=content, title="你"))
        self._render_timeline()

    def add_system(self, title: str, content: str, *, turn_index: int = 0) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="system", turn_index=turn_index, content=content, title=title))
        self._render_timeline()

    def add_notice(
        self,
        title: str,
        content: str,
        *,
        turn_index: int,
        detail_id: str | None = None,
        detail_lines: list[str] | None = None,
    ) -> None:
        item = TimelineItem(
            item_id=next(self._ids),
            role="notice",
            turn_index=turn_index,
            title=title,
            content=content,
            detail_id=detail_id,
            detail_lines=detail_lines or [],
        )
        self._items.append(item)
        self._render_or_mark_dirty()

    def add_effect_summary(
        self,
        title: str,
        content: str,
        *,
        turn_index: int,
        detail_id: str | None = None,
        detail_lines: list[str] | None = None,
    ) -> None:
        item = TimelineItem(
            item_id=next(self._ids),
            role="effect",
            turn_index=turn_index,
            title=title,
            content=content,
            detail_id=detail_id,
            detail_lines=detail_lines or [],
        )
        self._items.append(item)
        self._render_or_mark_dirty()

    def add_presentation_item(
        self,
        item: TimelinePresentationItem,
        details: ExpandableDetail | None,
    ) -> None:
        timeline_item = TimelineItem(
            item_id=next(self._ids),
            role=_presentation_role(item),
            turn_index=item.turn_index,
            title=item.title,
            content=item.summary,
            detail_id=item.detail_id,
            detail_lines=details.lines if details else [],
        )
        self._insert_timeline_item(timeline_item)
        self._render_or_mark_dirty()

    def replace_presentation_item(
        self,
        item: TimelinePresentationItem,
        details: ExpandableDetail | None,
    ) -> bool:
        if not item.detail_id:
            return False
        role = _presentation_role(item)
        for existing in self._items:
            if existing.detail_id != item.detail_id:
                continue
            existing.role = role
            existing.turn_index = item.turn_index
            existing.title = item.title
            existing.content = item.summary
            existing.detail_lines = details.lines if details else []
            if self._move_before_same_turn_assistant(existing):
                self._render_or_mark_dirty()
            else:
                self._sync_block_or_mark_dirty(existing)
            return True
        return False

    def add_failure(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="failure", turn_index=turn_index, content=content, status="failed", title="失败"))
        self._render_timeline()

    def toggle_detail(self, item_id: int) -> bool:
        for item in self._items:
            if item.item_id != item_id:
                continue
            if not item.detail_lines:
                return False
            item.expanded = not item.expanded
            if item.expanded and _is_process_item(item):
                self._expanded_process_turns.add(item.turn_index)
            self._render_or_mark_dirty()
            return True
        return False

    def activate_item(self, item_id: int) -> bool:
        if _is_process_group_id(item_id):
            return self.toggle_process_group(_process_group_turn_index(item_id))
        return self.toggle_detail(item_id)

    def toggle_process_group(self, turn_index: int) -> bool:
        if not any(item.turn_index == turn_index and _is_process_item(item) for item in self._items):
            return False
        if turn_index in self._expanded_process_turns:
            self._expanded_process_turns.remove(turn_index)
            for item in self._items:
                if item.turn_index == turn_index and _is_process_item(item):
                    item.expanded = False
        else:
            self._expanded_process_turns.add(turn_index)
        self._render_or_mark_dirty()
        return True

    def add_assistant_message(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="assistant", turn_index=turn_index, content=content, status="done", title="HaAgent"))
        self._render_timeline()

    def start_assistant_response(self, *, turn_index: int) -> None:
        item = self._assistant_item(turn_index)
        item.status = "streaming"
        self._sync_block(item)
        self._mark_plain_text_dirty()

    def update_assistant_delta(self, turn_index: int, delta: str) -> None:
        if not delta:
            return
        item = self._assistant_item(turn_index)
        item.content += delta
        item.status = "streaming"
        self._queue_assistant_delta_sync(item)
        self._mark_plain_text_dirty()

    def finalize_assistant(self, turn_index: int, content: str) -> None:
        self.flush_pending_assistant_delta()
        item = self._assistant_item(turn_index)
        if content:
            item.content = content
        item.status = "done"
        if self._interactive_updates_paused:
            self._pending_assistant_delta_item_ids.add(item.item_id)
        else:
            self._sync_block(item)
        self._mark_plain_text_dirty()

    def add_tool_activity(self, activity: ToolActivity) -> None:
        item = self._assistant_item(activity.turn_index)
        merge_tool_activity(item.tools, activity)
        self._queue_tool_activity_sync(item)
        self._mark_plain_text_dirty()

    def add_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        item = self._assistant_item(turn_index)
        activity = matching_latest_tool_activity(item.tools, tool_name)
        if activity is None:
            activity = ToolActivity(tool_name=tool_name, status="done", summary="诊断", turn_index=turn_index)
            item.tools.append(activity)
        if message not in activity.diagnostics:
            activity.diagnostics.append(message)
        self._queue_tool_activity_sync(item)
        self._mark_plain_text_dirty()

    def _assistant_item(self, turn_index: int) -> TimelineItem:
        for item in self._items:
            if item.role == "assistant" and item.turn_index == turn_index:
                return item
        item = TimelineItem(item_id=next(self._ids), role="assistant", turn_index=turn_index, content="", status="streaming", title="HaAgent")
        self._items.append(item)
        return item

    def _insert_timeline_item(self, item: TimelineItem) -> None:
        assistant_index = self._same_turn_assistant_index(item.turn_index)
        if item.role in {"activity", "notice", "effect"} and assistant_index is not None:
            self._items.insert(assistant_index, item)
            return
        self._items.append(item)

    def _same_turn_assistant_index(self, turn_index: int, *, exclude_item_id: int | None = None) -> int | None:
        for index, item in enumerate(self._items):
            if item.item_id == exclude_item_id:
                continue
            if item.role == "assistant" and item.turn_index == turn_index:
                return index
        return None

    def _move_before_same_turn_assistant(self, item: TimelineItem) -> bool:
        current_index = next((index for index, existing in enumerate(self._items) if existing.item_id == item.item_id), None)
        if current_index is None:
            return False
        assistant_index = self._same_turn_assistant_index(item.turn_index, exclude_item_id=item.item_id)
        if assistant_index is None or current_index < assistant_index:
            return False
        self._items.pop(current_index)
        self._items.insert(assistant_index, item)
        return True

    def _render_timeline(self) -> None:
        if not self.is_attached:
            self._mark_plain_text_dirty()
            return
        self._sync_blocks()
        self._mark_plain_text_dirty()

    def _render_or_mark_dirty(self) -> None:
        if self.is_attached:
            self._render_timeline()
            return
        self._mark_plain_text_dirty()

    def _sync_block(self, item: TimelineItem) -> None:
        block = self._blocks.get(item.item_id)
        if block is None:
            self._render_timeline()
            return
        block.update_item(item, show_tool_details=self._tool_details_enabled)
        self.set_class(bool(self._items), "timeline-ready")
        if self._stick_to_bottom:
            self.call_after_refresh(self.scroll_end_if_sticky)

    def _sync_block_or_mark_dirty(self, item: TimelineItem) -> None:
        if self.is_attached:
            self._sync_block(item)
            return
        self._mark_plain_text_dirty()

    def _queue_tool_activity_sync(self, item: TimelineItem) -> None:
        self._pending_tool_item_ids.add(item.item_id)
        self._schedule_tool_activity_flush()

    def _schedule_tool_activity_flush(self) -> None:
        if self._tool_activity_flush_scheduled:
            return
        self._tool_activity_flush_scheduled = True
        self.set_timer(
            TOOL_ACTIVITY_FLUSH_INTERVAL_MS / 1000,
            self.flush_pending_tool_activity,
            name="tool-activity-flush",
        )

    def flush_pending_tool_activity(self) -> None:
        if self._interactive_updates_paused:
            self._tool_activity_flush_scheduled = False
            return
        if not self._pending_tool_item_ids:
            self._tool_activity_flush_scheduled = False
            return
        pending_ids = set(self._pending_tool_item_ids)
        self._pending_tool_item_ids.clear()
        self._tool_activity_flush_scheduled = False
        for item in self._items:
            if item.item_id in pending_ids:
                self._sync_block(item)

    def _queue_assistant_delta_sync(self, item: TimelineItem) -> None:
        self._pending_assistant_delta_item_ids.add(item.item_id)
        self._schedule_assistant_delta_flush()

    def _schedule_assistant_delta_flush(self) -> None:
        if self._assistant_delta_flush_scheduled:
            return
        self._assistant_delta_flush_scheduled = True
        self.set_timer(
            MARKDOWN_DELTA_FLUSH_INTERVAL_MS / 1000,
            self.flush_pending_assistant_delta,
            name="assistant-delta-flush",
        )

    def flush_pending_assistant_delta(self) -> None:
        if self._interactive_updates_paused:
            self._assistant_delta_flush_scheduled = False
            return
        if not self._pending_assistant_delta_item_ids:
            self._assistant_delta_flush_scheduled = False
            return
        pending_ids = set(self._pending_assistant_delta_item_ids)
        self._pending_assistant_delta_item_ids.clear()
        self._assistant_delta_flush_scheduled = False
        for item in self._items:
            if item.item_id in pending_ids:
                self._sync_block(item)

    def pause_interactive_updates(self) -> None:
        self._interactive_updates_paused = True

    def resume_interactive_updates(self) -> None:
        self._interactive_updates_paused = False
        self.flush_pending_assistant_delta()
        self.flush_pending_tool_activity()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            self._mouse_down_location = (event.x, event.y)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._mouse_down_location is None or event.button != 1:
            return
        start_x, start_y = self._mouse_down_location
        if abs(event.x - start_x) + abs(event.y - start_y) >= 2:
            self._selection_dragging = True
            self.pause_interactive_updates()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._mouse_down_location = None
        if not self._selection_dragging:
            return
        self._selection_dragging = False
        self.set_timer(
            SELECTION_RESUME_DELAY_MS / 1000,
            self.resume_interactive_updates,
            name="selection-resume",
        )

    def _sync_blocks(self) -> None:
        self._remove_singletons()
        seen: set[int] = set()
        anchor: Any = None
        for item in self._visible_items():
            block = self._blocks.get(item.item_id)
            if block is None:
                block = TimelineBlock(item, show_tool_details=self._tool_details_enabled)
                self._blocks[item.item_id] = block
                if anchor is None:
                    self.mount(block)
                else:
                    self.mount(block, after=anchor)
            else:
                block.update_item(item, show_tool_details=self._tool_details_enabled)
                if block.parent is None:
                    if anchor is None:
                        self.mount(block)
                    else:
                        self.mount(block, after=anchor)
            seen.add(item.item_id)
            anchor = block
        for item_id in list(self._blocks):
            if item_id not in seen:
                block = self._blocks.pop(item_id)
                block.remove()
        self.set_class(bool(self._items), "timeline-ready")
        if self._stick_to_bottom:
            self.call_after_refresh(self.scroll_end_if_sticky)

    def _sync_plain_text(self) -> None:
        self._plain_text = "\n\n".join(
            _render_timeline_item(item, show_tool_details=self._tool_details_enabled) for item in self._visible_items()
        )
        self._plain_text_dirty = False

    def _visible_items(self) -> list[TimelineItem]:
        visible: list[TimelineItem] = []
        process_turns_rendered: set[int] = set()
        for item in self._items:
            if not _is_process_item(item):
                visible.append(item)
                continue
            if item.turn_index in process_turns_rendered:
                continue
            process_turns_rendered.add(item.turn_index)
            process_items = [candidate for candidate in self._items if candidate.turn_index == item.turn_index and _is_process_item(candidate)]
            visible.append(_process_group_item(item.turn_index, len(process_items), expanded=item.turn_index in self._expanded_process_turns))
            if item.turn_index in self._expanded_process_turns:
                visible.extend(process_items)
        return visible

    def _mark_plain_text_dirty(self) -> None:
        self._plain_text_dirty = True

    def _show_singleton(self, kind: str, text: str, classes: str) -> None:
        self._remove_singletons()
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()
        existing = Static(text, classes=classes)
        self.mount(existing)
        if kind == "placeholder":
            self._placeholder_widget = existing
        else:
            self._memory_widget = existing
        self.set_class(False, "timeline-ready")
        self.call_after_refresh(self.scroll_end_if_sticky)

    def _remove_singletons(self) -> None:
        for widget in (self._placeholder_widget, self._memory_widget):
            if widget is not None:
                widget.remove()
        self._placeholder_widget = None
        self._memory_widget = None

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._stick_to_bottom = new_value >= self.max_scroll_y - 1
        if self._stick_to_bottom:
            self.resume_interactive_updates()
        else:
            self.pause_interactive_updates()


# 公共别名：ConversationView 与 ConversationTimeline 等价
ConversationView = ConversationTimeline
