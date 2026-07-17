"""
src/haagent/tui/widgets/conversation_timeline.py - 对话时间线主组件

VerticalScroll 容器，维护 TimelineItem 列表并按需挂载/更新 TimelineBlock。
支持 streaming 批处理、增量索引、长会话挂载窗口和鼠标拖拽暂停交互更新。
"""

from __future__ import annotations

from itertools import count
from time import monotonic
from typing import Any

from textual import events
from textual.containers import VerticalScroll
from textual.timer import Timer
from textual.widgets import Static

from haagent.app.assistant_types import AssistantSessionTurn
from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem
from haagent.tui.widgets.timeline_models import (
    DETAILS_REFRESH_RECENT_TURNS,
    InteractionKey,
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
from haagent.tui.widgets.tool_activity import (
    matching_latest_tool_activity,
    matching_open_tool_activity,
    merge_tool_activity,
)


TIMELINE_WINDOW_SIZE = 200
TIMELINE_WINDOW_STEP = 100
ELAPSED_REFRESH_INTERVAL_SECONDS = 1.0


class ConversationTimeline(VerticalScroll):
    _ids = count(1)
    can_focus = True

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("wrap", None)
        kwargs.pop("auto_scroll", None)
        super().__init__(*args, **kwargs)
        self._items: list[TimelineItem] = []
        # 索引只在统一新增、替换和清空入口维护，热路径无需反复扫描完整会话。
        self._items_by_id: dict[int, TimelineItem] = {}
        self._detail_items_by_id: dict[str, TimelineItem] = {}
        # 只索引尚未回答的人机交互；resolved 事件按结构化 key 原地替换，禁止 requested 条目残留。
        self._pending_interaction_items: dict[InteractionKey, TimelineItem] = {}
        self._assistant_items_by_turn: dict[int, TimelineItem] = {}
        self._process_items_by_turn: dict[int, list[TimelineItem]] = {}
        self._turn_started_at: dict[int, float] = {}
        self._elapsed_timer: Timer | None = None
        self._visible_items_cache: list[TimelineItem] | None = None
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
        self._collapsed_process_turns: set[int] = set()
        # None 表示尾窗；非 None 时是用户正在回看的完整可见投影起点。
        self._window_start: int | None = None
        self._window_shift_in_progress = False

    def set_stick_to_bottom(self, enabled: bool) -> None:
        window_was_away_from_tail = self._window_start is not None
        self._stick_to_bottom = enabled
        if enabled:
            self._window_start = None
            if window_was_away_from_tail and self.is_attached:
                self._replace_window_blocks()

    def scroll_end_if_sticky(self) -> None:
        if self._stick_to_bottom:
            self.scroll_end(animate=False)

    def scroll_to(self, x: float | None = None, y: float | None = None, **kwargs: Any) -> None:
        if y is not None:
            self._stick_to_bottom = self._window_start is None and y >= self.max_scroll_y - 1
            if not self._stick_to_bottom:
                self.pause_interactive_updates()
        super().scroll_to(x=x, y=y, **kwargs)

    def show_placeholder(self) -> None:
        self._clear_items()
        text = "可以开始了。Ctrl+Enter 换行，输入 / 打开命令。"
        self._plain_text = text
        self._plain_text_dirty = False
        self._show_singleton("placeholder", text, "timeline-placeholder")

    def show_memory(self, text: str) -> None:
        # 记忆候选只是临时覆盖层；保留 timeline 数据，Esc 返回时才能恢复原对话。
        self._plain_text = text
        self._plain_text_dirty = False
        self._show_singleton("memory", text, "timeline-memory")

    def hide_memory(self) -> None:
        """关闭记忆候选覆盖层，并从保留的数据模型恢复对话节点。"""
        if self._memory_widget is None:
            return
        self._remove_singletons()
        self._render_timeline()

    @property
    def plain_text(self) -> str:
        if self._plain_text_dirty:
            self._sync_plain_text()
        return self._plain_text

    def clear_timeline(self) -> None:
        self._clear_items()
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
        self._collapsed_process_turns.clear()
        self._window_start = None
        self._window_shift_in_progress = False
        self._remove_singletons()
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()

    def set_tool_details(self, enabled: bool) -> None:
        if self._tool_details_enabled == enabled:
            return
        self._tool_details_enabled = enabled
        process_turns = set(self._process_items_by_turn)
        if process_turns:
            if enabled:
                self._expanded_process_turns.update(process_turns)
                self._collapsed_process_turns.difference_update(process_turns)
            else:
                self._expanded_process_turns.difference_update(process_turns)
                self._collapsed_process_turns.update(process_turns)
                for turn_index in process_turns:
                    for item in self._process_items_by_turn[turn_index]:
                        item.expanded = False
            self._invalidate_visible_items()
            self._render_timeline()
            return
        recent_assistants = [item for item in self._items if item.role == "assistant"][-DETAILS_REFRESH_RECENT_TURNS:]
        if not recent_assistants:
            self._render_timeline()
            return
        for item in recent_assistants:
            self._sync_block(item)
        self._mark_plain_text_dirty()

    def add_user(self, content: str, *, turn_index: int) -> None:
        self._append_item(TimelineItem(item_id=next(self._ids), role="user", turn_index=turn_index, content=content, title="你"))
        self._render_timeline()

    def load_session_history(self, turns: list[AssistantSessionTurn]) -> None:
        """批量装载会话历史，只触发一次 timeline 同步。"""
        from haagent.tui.application.session_flow import session_turn_assistant_text

        self.clear_timeline()
        for turn in turns:
            turn_index = turn.turn_index
            request = turn.request
            assistant_text = session_turn_assistant_text(turn)
            self._append_item(
                TimelineItem(
                    item_id=next(self._ids),
                    role="user",
                    turn_index=turn_index,
                    content=request,
                    title="你",
                ),
            )
            self._append_item(
                TimelineItem(
                    item_id=next(self._ids),
                    role="assistant",
                    turn_index=turn_index,
                    content=assistant_text,
                    status="done",
                    title="HaAgent",
                    is_final_answer=turn.status == "completed",
                ),
            )
            status = turn.status
            if status != "completed":
                self._append_item(
                    TimelineItem(
                        item_id=next(self._ids),
                        role="system",
                        turn_index=turn_index,
                        content=f"状态：{status}",
                        title="状态",
                    ),
                )
        self._render_timeline()

    def add_system(self, title: str, content: str, *, turn_index: int = 0) -> None:
        self._append_item(TimelineItem(item_id=next(self._ids), role="system", turn_index=turn_index, content=content, title=title))
        self._render_timeline()

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
            interaction_key=item.interaction_key,
            requires_attention=item.requires_attention,
            status="failed" if item.severity == "error" else "done",
            pinned=item.severity in {"warning", "error"},
        )
        self._insert_timeline_item(timeline_item)
        self._render_or_mark_dirty()

    def replace_presentation_item(
        self,
        item: TimelinePresentationItem,
        details: ExpandableDetail | None,
    ) -> bool:
        if not item.detail_id and item.interaction_key is None:
            return False
        role = _presentation_role(item)
        existing = (
            self._pending_interaction_items.get(item.interaction_key)
            if item.interaction_key is not None
            else self._detail_items_by_id.get(item.detail_id)
        )
        if existing is None:
            return False
        self._unindex_item(existing)
        existing.role = role
        existing.turn_index = item.turn_index
        existing.title = item.title
        existing.content = item.summary
        existing.detail_id = item.detail_id
        existing.detail_lines = details.lines if details else []
        existing.interaction_key = item.interaction_key
        existing.requires_attention = item.requires_attention
        existing.status = "failed" if item.severity == "error" else "done"
        existing.pinned = item.severity in {"warning", "error"}
        self._index_item(existing)
        self._invalidate_visible_items()
        if self._move_before_same_turn_assistant(existing):
            self._render_or_mark_dirty()
        else:
            self._sync_block_or_mark_dirty(existing)
        return True

    def dismiss_pending_interaction(self, interaction_key: InteractionKey) -> bool:
        item = self._pending_interaction_items.get(interaction_key)
        if item is None:
            return False
        self._unindex_item(item)
        self._items.remove(item)
        self._invalidate_visible_items()
        self._render_or_mark_dirty()
        return True

    def add_failure(self, content: str, *, turn_index: int) -> None:
        self._append_item(TimelineItem(item_id=next(self._ids), role="failure", turn_index=turn_index, content=content, status="failed", title="失败"))
        self._render_timeline()

    def toggle_detail(self, item_id: int) -> bool:
        item = self._items_by_id.get(item_id)
        if item is None or (not item.detail_lines and not item.tools):
            return False
        item.expanded = not item.expanded
        if _is_process_item(item):
            if item.expanded:
                self._expanded_process_turns.add(item.turn_index)
            self._invalidate_visible_items()
        self._render_or_mark_dirty()
        return True

    def activate_item(self, item_id: int) -> bool:
        if _is_process_group_id(item_id):
            return self.toggle_process_group(_process_group_turn_index(item_id))
        return self.toggle_detail(item_id)

    def toggle_process_group(self, turn_index: int) -> bool:
        process_items = self._process_items_by_turn.get(turn_index)
        if not process_items:
            return False
        if turn_index in self._expanded_process_turns:
            self._expanded_process_turns.remove(turn_index)
            self._collapsed_process_turns.add(turn_index)
            for item in process_items:
                item.expanded = False
        else:
            self._expanded_process_turns.add(turn_index)
            self._collapsed_process_turns.discard(turn_index)
        self._invalidate_visible_items()
        self._render_or_mark_dirty()
        return True

    def add_assistant_message(self, content: str, *, turn_index: int) -> None:
        self._append_item(
            TimelineItem(
                item_id=next(self._ids),
                role="assistant",
                turn_index=turn_index,
                content=content,
                status="done",
                title="HaAgent",
                is_final_answer=True,
            ),
        )
        self._collapse_process_turn(turn_index)
        self._render_timeline()

    def start_assistant_response(self, *, turn_index: int) -> None:
        item = self._assistant_item(turn_index)
        self._turn_started_at.setdefault(turn_index, monotonic())
        self._ensure_elapsed_timer()
        item.status = "streaming"
        item.is_final_answer = False
        self._sync_block(item)
        self._mark_plain_text_dirty()

    def update_assistant_delta(self, turn_index: int, delta: str) -> None:
        if not delta:
            return
        item = self._assistant_item(turn_index)
        item.content += delta
        item.status = "streaming"
        item.is_final_answer = False
        self._queue_assistant_delta_sync(item)
        self._mark_plain_text_dirty()

    def finalize_assistant(self, turn_index: int, content: str) -> None:
        self.flush_pending_assistant_delta()
        item = self._assistant_item(turn_index)
        if content:
            item.content = content
        item.status = "done"
        item.is_final_answer = True
        self._finish_turn_timing(item)
        self._move_tools_out_of_final_answer(item)
        process_items = self._collapse_process_turn(turn_index)
        if self._interactive_updates_paused:
            self._pending_assistant_delta_item_ids.add(item.item_id)
        elif process_items:
            self._render_timeline()
        else:
            self._sync_block(item)
        self._mark_plain_text_dirty()

    def finish_assistant_without_final(self, turn_index: int, content: str) -> None:
        """结束流式占位但不触发“最终回答已到达”的过程折叠。"""
        self.flush_pending_assistant_delta()
        item = self._assistant_item(turn_index)
        if content:
            item.content = content
        item.status = "done"
        item.is_final_answer = False
        self._finish_turn_timing(item)
        if self._interactive_updates_paused:
            self._pending_assistant_delta_item_ids.add(item.item_id)
        else:
            self._sync_block(item)
        self._mark_plain_text_dirty()

    def _finish_turn_timing(self, item: TimelineItem) -> None:
        started_at = self._turn_started_at.pop(item.turn_index, None)
        if started_at is not None:
            item.elapsed_seconds = max(0.0, monotonic() - started_at)
        if not self._turn_started_at and self._elapsed_timer is not None:
            self._elapsed_timer.pause()

    def _ensure_elapsed_timer(self) -> None:
        if not self.is_attached or not any(
            turn_index in self._turn_started_at for turn_index in self._process_items_by_turn
        ):
            return
        if self._elapsed_timer is None:
            # 单一低频 timer 只刷新过程组标题，不触发 timeline 全量重绘。
            self._elapsed_timer = self.set_interval(
                ELAPSED_REFRESH_INTERVAL_SECONDS,
                self._refresh_elapsed_process_groups,
            )
            return
        self._elapsed_timer.resume()

    def _refresh_elapsed_process_groups(self) -> None:
        if self._interactive_updates_paused:
            return
        now = monotonic()
        active_turns = [
            (turn_index, started_at)
            for turn_index, started_at in self._turn_started_at.items()
            if self._process_items_by_turn.get(turn_index)
        ]
        if not active_turns:
            if self._elapsed_timer is not None:
                self._elapsed_timer.pause()
            return
        self._invalidate_visible_items()
        for turn_index, started_at in active_turns:
            group_item = _process_group_item(
                turn_index,
                self._process_items_by_turn[turn_index],
                expanded=turn_index in self._expanded_process_turns,
                elapsed_seconds=max(0.0, now - started_at),
            )
            self._sync_block(group_item)
        self._mark_plain_text_dirty()

    def _move_tools_out_of_final_answer(self, item: TimelineItem) -> None:
        if not item.tools:
            return
        # 最终回答已经结束本轮过程；没有结束事件的临时“运行中”状态不再持久化。
        completed_tools = [tool for tool in item.tools if tool.status != "running"]
        item.tools = []
        self._pending_tool_item_ids.discard(item.item_id)
        if not completed_tools:
            return
        existing_process_items = self._process_items_by_turn.get(item.turn_index, [])
        if existing_process_items:
            target = existing_process_items[-1]
            for tool in completed_tools:
                merge_tool_activity(target.tools, tool)
            self._sync_block_or_mark_dirty(target)
            return
        process_item = TimelineItem(
            item_id=next(self._ids),
            role="process",
            turn_index=item.turn_index,
            content="",
            status="done",
            title="",
            tools=completed_tools,
        )
        assistant_index = self._items.index(item)
        self._items.insert(assistant_index, process_item)
        self._index_item(process_item)
        self._invalidate_visible_items()

    def _collapse_process_turn(self, turn_index: int) -> list[TimelineItem]:
        process_items = self._process_items_by_turn.get(turn_index, [])
        self._expanded_process_turns.discard(turn_index)
        self._collapsed_process_turns.add(turn_index)
        for process_item in process_items:
            process_item.expanded = False
        self._invalidate_visible_items()
        return process_items

    def finalize_intermediate(self, turn_index: int, model_turn: int | None, content: str) -> None:
        """把当前 provisional assistant 项固化为可折叠过程，释放最终回复槽位。"""
        self.flush_pending_assistant_delta()
        item = self._assistant_item(turn_index)
        resolved = content or item.content
        self._unindex_item(item)
        item.role = "process"
        item.status = "done"
        # 过程子项不贴「过程」标签；折叠头统一用「已完成 N 项」，正文/工具摘要自解释。
        item.title = ""
        item.content = resolved
        item.detail_id = None
        item.detail_lines = []
        self._index_item(item)
        self._render_or_mark_dirty()

    def add_tool_activity(self, activity: ToolActivity) -> None:
        # 终态必须回写最初的 running/approval 记录；审批或中间输出可能已插入新的过程项。
        target = self._matching_open_tool_target(activity) or self._tool_activity_target(activity.turn_index)
        merge_tool_activity(target.tools, activity)
        self._queue_tool_activity_sync(target)
        self._mark_plain_text_dirty()

    def _matching_open_tool_target(self, activity: ToolActivity) -> TimelineItem | None:
        for item in reversed(self._items):
            if item.turn_index != activity.turn_index:
                continue
            if matching_open_tool_activity(item.tools, activity) is not None:
                return item
        return None

    def add_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        target = self._tool_activity_target(turn_index)
        activity = matching_latest_tool_activity(target.tools, tool_name)
        if activity is None:
            activity = ToolActivity(tool_name=tool_name, status="done", summary="诊断", turn_index=turn_index)
            target.tools.append(activity)
        if message not in activity.diagnostics:
            activity.diagnostics.append(message)
        self._queue_tool_activity_sync(target)
        self._mark_plain_text_dirty()

    def _tool_activity_target(self, turn_index: int) -> TimelineItem:
        """运行中的工具暂挂在 assistant；最终回答到达后再归入过程流。"""

        process_items = self._process_items_by_turn.get(turn_index, [])
        assistant = self._assistant_items_by_turn.get(turn_index)
        if assistant is not None and assistant.status != "done" and not process_items:
            return assistant
        if process_items:
            return process_items[-1]
        if assistant is not None and assistant.status != "done":
            return assistant
        process_item = TimelineItem(
            item_id=next(self._ids),
            role="process",
            turn_index=turn_index,
            content="",
            status="done",
            title="",
        )
        if assistant is not None:
            assistant_index = self._items.index(assistant)
            self._items.insert(assistant_index, process_item)
        else:
            self._items.append(process_item)
        self._index_item(process_item)
        self._invalidate_visible_items()
        return process_item

    def _assistant_item(self, turn_index: int) -> TimelineItem:
        item = self._assistant_items_by_turn.get(turn_index)
        if item is not None:
            return item
        item = TimelineItem(item_id=next(self._ids), role="assistant", turn_index=turn_index, content="", status="streaming", title="HaAgent")
        self._append_item(item)
        return item

    def _append_item(self, item: TimelineItem) -> None:
        self._items.append(item)
        self._index_item(item)
        self._invalidate_visible_items()

    def _insert_timeline_item(self, item: TimelineItem) -> None:
        assistant_index = self._same_turn_assistant_index(item.turn_index)
        if item.role in {"activity", "notice", "effect"} and assistant_index is not None:
            self._items.insert(assistant_index, item)
        else:
            self._items.append(item)
        self._index_item(item)
        self._invalidate_visible_items()

    def _index_item(self, item: TimelineItem) -> None:
        self._items_by_id[item.item_id] = item
        if item.detail_id:
            self._detail_items_by_id[item.detail_id] = item
        if item.requires_attention and item.interaction_key is not None:
            self._pending_interaction_items[item.interaction_key] = item
        if item.role == "assistant":
            self._assistant_items_by_turn[item.turn_index] = item
        if _is_process_item(item):
            process_items = self._process_items_by_turn.setdefault(item.turn_index, [])
            process_items.append(item)
            # 模型过程、失败和任务受阻在运行中保持可见；最终回答到达后统一折叠。
            assistant = self._assistant_items_by_turn.get(item.turn_index)
            has_final_answer = assistant is not None and assistant.is_final_answer
            if has_final_answer:
                self._collapse_process_turn(item.turn_index)
            elif (item.role in {"process", "notice", "failure"} or item.pinned) and item.turn_index not in self._collapsed_process_turns:
                self._expanded_process_turns.add(item.turn_index)
            self._ensure_elapsed_timer()

    def _unindex_item(self, item: TimelineItem) -> None:
        if self._items_by_id.get(item.item_id) is item:
            self._items_by_id.pop(item.item_id)
        if item.detail_id and self._detail_items_by_id.get(item.detail_id) is item:
            self._detail_items_by_id.pop(item.detail_id)
        if item.interaction_key is not None and self._pending_interaction_items.get(item.interaction_key) is item:
            self._pending_interaction_items.pop(item.interaction_key)
        if item.role == "assistant" and self._assistant_items_by_turn.get(item.turn_index) is item:
            self._assistant_items_by_turn.pop(item.turn_index)
        if _is_process_item(item):
            process_items = self._process_items_by_turn.get(item.turn_index, [])
            process_items[:] = [candidate for candidate in process_items if candidate is not item]
            if not process_items:
                self._process_items_by_turn.pop(item.turn_index, None)

    def _clear_items(self) -> None:
        self._items.clear()
        self._items_by_id.clear()
        self._detail_items_by_id.clear()
        self._pending_interaction_items.clear()
        self._assistant_items_by_turn.clear()
        self._process_items_by_turn.clear()
        self._turn_started_at.clear()
        if self._elapsed_timer is not None:
            self._elapsed_timer.pause()
        self._invalidate_visible_items()

    def _invalidate_visible_items(self) -> None:
        self._visible_items_cache = None

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
        block.update_item(item, show_tool_details=self._show_tool_details(item))
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
        # 未挂载 timeline（单元测试直接写）时不能 set_timer，立即同步 plain_text。
        if not self.is_attached:
            self.flush_pending_tool_activity()
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
        for item_id in pending_ids:
            item = self._items_by_id.get(item_id)
            if item is not None:
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
        for item_id in pending_ids:
            item = self._items_by_id.get(item_id)
            if item is not None:
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
        visible_items = self._visible_items()
        for item in self._windowed_items(visible_items):
            block = self._blocks.get(item.item_id)
            if block is None:
                block = TimelineBlock(item, show_tool_details=self._show_tool_details(item))
                self._blocks[item.item_id] = block
                if anchor is None:
                    self.mount(block)
                else:
                    self.mount(block, after=anchor)
            else:
                block.update_item(item, show_tool_details=self._show_tool_details(item))
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
            _render_timeline_item(item, show_tool_details=self._show_tool_details(item))
            for item in self._visible_items()
        )
        self._plain_text_dirty = False

    def _show_tool_details(self, item: TimelineItem) -> bool:
        return self._tool_details_enabled or item.expanded

    def _visible_items(self) -> list[TimelineItem]:
        if self._visible_items_cache is not None:
            return self._visible_items_cache
        visible: list[TimelineItem] = []
        process_turns_rendered: set[int] = set()
        for item in self._items:
            if not _is_process_item(item):
                visible.append(item)
                continue
            if item.turn_index in process_turns_rendered:
                continue
            process_turns_rendered.add(item.turn_index)
            process_items = self._process_items_by_turn[item.turn_index]
            expanded = item.turn_index in self._expanded_process_turns
            elapsed_seconds = self._turn_elapsed_seconds(item.turn_index)
            visible.append(
                _process_group_item(
                    item.turn_index,
                    process_items,
                    expanded=expanded,
                    elapsed_seconds=elapsed_seconds,
                )
            )
            if expanded:
                # 展开后直接露出过程子项：叙述正文 + 工具摘要行，不加步骤 chrome。
                visible.extend(process_items)
        self._visible_items_cache = visible
        return self._visible_items_cache

    def _turn_elapsed_seconds(self, turn_index: int) -> float | None:
        assistant = self._assistant_items_by_turn.get(turn_index)
        if assistant is not None and assistant.elapsed_seconds is not None:
            return assistant.elapsed_seconds
        started_at = self._turn_started_at.get(turn_index)
        if started_at is None:
            return None
        return max(0.0, monotonic() - started_at)

    def _windowed_items(self, visible_items: list[TimelineItem]) -> list[TimelineItem]:
        if len(visible_items) <= TIMELINE_WINDOW_SIZE:
            self._window_start = None
            return visible_items
        tail_start = len(visible_items) - TIMELINE_WINDOW_SIZE
        start = tail_start if self._window_start is None else min(self._window_start, tail_start)
        return visible_items[start : start + TIMELINE_WINDOW_SIZE]

    def _shift_window_earlier(self, visible_count: int) -> bool:
        tail_start = max(0, visible_count - TIMELINE_WINDOW_SIZE)
        current_start = tail_start if self._window_start is None else min(self._window_start, tail_start)
        next_start = max(0, current_start - TIMELINE_WINDOW_STEP)
        if next_start == current_start:
            return False
        self._window_start = next_start
        return True

    def _shift_window_later(self, visible_count: int) -> bool:
        if self._window_start is None:
            return False
        tail_start = max(0, visible_count - TIMELINE_WINDOW_SIZE)
        next_start = min(tail_start, self._window_start + TIMELINE_WINDOW_STEP)
        if next_start == self._window_start:
            return False
        self._window_start = None if next_start == tail_start else next_start
        return True

    def _replace_window_blocks(self) -> None:
        # 翻窗发生频率远低于流式更新；边界切换时重建有界窗口可保证 DOM 顺序稳定。
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()
        self._sync_blocks()

    def _restore_window_anchor(self, item_id: int) -> None:
        block = self._blocks.get(item_id)
        if block is not None and block.parent is not None:
            # 前移窗口后把原首项放回顶部，避免挂载更早消息时阅读位置跳动。
            self.scroll_to_widget(block, top=True, animate=False, immediate=True)
        self._window_shift_in_progress = False

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
        if self._window_shift_in_progress:
            return

        visible_items = self._visible_items()
        window_items = self._windowed_items(visible_items)
        if new_value <= 1 and new_value < old_value and window_items:
            anchor_item_id = window_items[0].item_id
            if self._shift_window_earlier(len(visible_items)):
                self._stick_to_bottom = False
                self.pause_interactive_updates()
                self._window_shift_in_progress = True
                self._replace_window_blocks()
                self.call_after_refresh(self._restore_window_anchor, anchor_item_id)
                return

        at_window_bottom = new_value >= self.max_scroll_y - 1
        if at_window_bottom and new_value > old_value and self._window_start is not None:
            if self._shift_window_later(len(visible_items)):
                self._stick_to_bottom = self._window_start is None
                self._window_shift_in_progress = True
                self._replace_window_blocks()
                if self._stick_to_bottom:
                    self._window_shift_in_progress = False
                    self.resume_interactive_updates()
                else:
                    next_window = self._windowed_items(visible_items)
                    self.call_after_refresh(self._restore_window_anchor, next_window[0].item_id)
                return

        self._stick_to_bottom = self._window_start is None and at_window_bottom
        if self._stick_to_bottom:
            self.resume_interactive_updates()
        else:
            self.pause_interactive_updates()
