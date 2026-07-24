"""
tests/unit/tui/test_timeline_performance.py - TUI timeline 性能合同测试

覆盖工具详情和长对话场景下的渲染热路径，避免后续改动重新引入全量重算。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.app import App, ComposeResult

import haagent.tui.widgets.timeline_models as timeline_module
from haagent.tui.widgets import conversation_timeline as conversation_timeline_module
from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem
from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.timeline_block import TimelineBlock
from haagent.tui.widgets.timeline_models import TimelineItem, ToolActivity
from haagent.tui.widgets.request_history_rail import RequestHistoryRail, group_request_history_entries


class InstrumentedTimeline(ConversationTimeline):
    def __init__(self) -> None:
        super().__init__()
        self.plain_text_sync_count = 0
        self.render_timeline_count = 0
        self.synced_items: list[int] = []
        self.scheduled_flush_count = 0
        self.scheduled_flush_delay: float | None = None
        self.scheduled_delta_flush_count = 0

    def _sync_plain_text(self) -> None:
        self.plain_text_sync_count += 1
        super()._sync_plain_text()

    def _render_timeline(self) -> None:
        self.render_timeline_count += 1

    def _sync_block(self, item: TimelineItem) -> None:
        self.synced_items.append(item.item_id)

    def _schedule_tool_activity_flush(self) -> None:
        if self._tool_activity_flush_scheduled:
            return
        self._tool_activity_flush_scheduled = True
        self.scheduled_flush_count += 1
        self.scheduled_flush_delay = timeline_module.TOOL_ACTIVITY_FLUSH_INTERVAL_MS / 1000

    def _schedule_assistant_delta_flush(self) -> None:
        if self._assistant_delta_flush_scheduled:
            return
        self._assistant_delta_flush_scheduled = True
        self.scheduled_delta_flush_count += 1


class CountingItemList(list[TimelineItem]):
    def __init__(self, items: list[TimelineItem]) -> None:
        super().__init__(items)
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        return super().__iter__()


class LongTimelineApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConversationTimeline(id="conversation")

    def on_mount(self) -> None:
        timeline = self.query_one(ConversationTimeline)
        for turn_index in range(500):
            timeline._append_item(
                TimelineItem(
                    item_id=next(timeline._ids),
                    role="user",
                    turn_index=turn_index,
                    content=f"prompt {turn_index}",
                ),
            )
        timeline._sync_blocks()


def test_sync_blocks_does_not_update_unchanged_history_block(monkeypatch) -> None:
    """再次同步窗口时，未变化的完成 Markdown 不得被重新解析。"""

    timeline = ConversationTimeline()
    item = TimelineItem(
        item_id=1,
        role="assistant",
        turn_index=1,
        content="已完成的历史回答",
        status="done",
    )

    class _Block:
        parent = object()

        def __init__(self) -> None:
            self.update_calls = 0

        def update_item(self, updated: TimelineItem, *, show_tool_details: bool) -> None:
            del updated, show_tool_details
            self.update_calls += 1

        def needs_update(self, updated: TimelineItem, *, show_tool_details: bool) -> bool:
            del updated, show_tool_details
            return False

    block = _Block()
    timeline._items = [item]
    timeline._items_by_id[item.item_id] = item
    timeline._blocks[item.item_id] = block  # type: ignore[assignment]
    timeline._visible_items_cache = [item]
    monkeypatch.setattr(ConversationTimeline, "is_attached", property(lambda self: True))

    timeline._sync_blocks()

    assert block.update_calls == 0


def test_tool_activity_marks_plain_text_dirty_without_eager_full_rebuild() -> None:
    timeline = InstrumentedTimeline()
    timeline._assistant_item(1)

    timeline.add_tool_activity(
        ToolActivity(
            tool_name="web_fetch",
            status="done",
            summary="status=success, result_keys=content",
            turn_index=1,
        ),
    )

    assert timeline.plain_text_sync_count == 0
    assert timeline.synced_items == []
    assert timeline.scheduled_flush_count == 1
    assert timeline.scheduled_flush_delay == 0.05
    assert "已完成 1 项" in timeline.plain_text


def test_elapsed_refresh_updates_only_the_process_group_block(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(conversation_timeline_module, "monotonic", lambda: clock[0])
    timeline = InstrumentedTimeline()
    timeline.start_assistant_response(turn_index=1)
    timeline.finalize_intermediate(1, 1, "正在搜索")
    timeline.synced_items.clear()
    timeline.render_timeline_count = 0

    clock[0] = 88.0
    timeline._refresh_elapsed_process_groups()

    assert "已完成 1 步 · 1分18秒" in timeline.plain_text
    assert timeline.render_timeline_count == 0
    assert len(timeline.synced_items) == 1
    assert timeline.synced_items[0] < 0
    assert timeline.plain_text_sync_count == 1


def test_tool_activity_updates_are_batched_until_flush() -> None:
    timeline = InstrumentedTimeline()

    timeline.add_tool_activity(
        ToolActivity(
            tool_name="web_fetch",
            status="running",
            summary="running",
            turn_index=1,
        ),
    )
    timeline.add_tool_activity(
        ToolActivity(
            tool_name="web_search",
            status="done",
            summary="status=success",
            turn_index=1,
        ),
    )
    timeline.add_tool_diagnostic(1, "web_search", "结果已压缩 20228 -> 931 字符")

    assert timeline.synced_items == []
    assert timeline.scheduled_flush_count == 1

    timeline.flush_pending_tool_activity()

    assert timeline.synced_items == [timeline._items[0].item_id]


def test_tool_activity_flush_waits_while_interaction_is_paused() -> None:
    timeline = InstrumentedTimeline()
    timeline.pause_interactive_updates()
    timeline.add_tool_activity(
        ToolActivity(
            tool_name="web_fetch",
            status="done",
            summary="status=success",
            turn_index=1,
        ),
    )

    timeline.flush_pending_tool_activity()

    assert timeline.synced_items == []
    assert timeline._pending_tool_item_ids

    timeline.resume_interactive_updates()

    assert timeline.synced_items == [timeline._items[0].item_id]


def test_assistant_delta_updates_are_batched_until_flush() -> None:
    timeline = InstrumentedTimeline()

    timeline.update_assistant_delta(1, "Ha")
    timeline.update_assistant_delta(1, "Agent")

    assert timeline.synced_items == []
    assert timeline._items[0].content == "HaAgent"
    assert timeline.scheduled_delta_flush_count == 1

    timeline.flush_pending_assistant_delta()

    assert timeline.synced_items == [timeline._items[0].item_id]


def test_many_assistant_deltas_batch_below_delta_count() -> None:
    """100 个 delta 只调度一次批 flush；周期必须落在 16–50ms。"""
    timeline = InstrumentedTimeline()
    for index in range(100):
        timeline.update_assistant_delta(1, f"d{index}")

    assert timeline.scheduled_delta_flush_count == 1
    assert timeline.synced_items == []
    assert 16 <= timeline_module.MARKDOWN_DELTA_FLUSH_INTERVAL_MS <= 50

    timeline.flush_pending_assistant_delta()
    assert timeline.synced_items == [timeline._items[0].item_id]
    assert len(timeline.synced_items) < 100


def test_assistant_delta_flush_waits_while_interaction_is_paused() -> None:
    timeline = InstrumentedTimeline()
    timeline.pause_interactive_updates()
    timeline.update_assistant_delta(1, "Ha")
    timeline.update_assistant_delta(1, "Agent")

    timeline.flush_pending_assistant_delta()

    assert timeline.synced_items == []
    assert timeline._pending_assistant_delta_item_ids

    timeline.resume_interactive_updates()

    assert timeline.synced_items == [timeline._items[0].item_id]


def test_final_assistant_message_waits_while_interaction_is_paused() -> None:
    timeline = InstrumentedTimeline()
    timeline.pause_interactive_updates()

    timeline.finalize_assistant(1, "最终回答")

    assert timeline.synced_items == []
    assert timeline._pending_assistant_delta_item_ids

    timeline.resume_interactive_updates()

    assert timeline.synced_items == [timeline._items[0].item_id]
    assert timeline._items[0].status == "done"


def test_load_session_history_renders_once_for_many_turns() -> None:
    timeline = InstrumentedTimeline()
    turns = [
        SimpleNamespace(
            turn_index=index,
            request=f"prompt {index}",
            summary=f"summary {index}",
            status="completed",
            assistant_display_text=f"answer {index}",
        )
        for index in range(1, 41)
    ]

    timeline.load_session_history(turns)

    assert timeline.render_timeline_count == 1
    assert len(timeline._items) == 80


def test_request_history_entries_follow_timeline_state_and_redact_summaries() -> None:
    timeline = InstrumentedTimeline()
    timeline.add_user("读取 sk-secret-12345678901234567890 并总结", turn_index=1)
    timeline.start_assistant_response(turn_index=1)
    timeline.add_user("第二个请求", turn_index=2)
    timeline.finalize_assistant(2, "已完成第二个请求")

    entries = timeline.request_history_entries()

    assert [entry.turn_index for entry in entries] == [1, 2]
    assert entries[0].answer_summary == "正在生成回答"
    assert "sk-secret" not in entries[0].request_summary
    assert entries[1].answer_summary == "已完成第二个请求"


def test_request_history_final_answer_wins_over_tool_failure() -> None:
    timeline = InstrumentedTimeline()
    timeline.add_user("解释鲜味", turn_index=1)
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="notice",
            title="运行工具失败",
            summary="网页抓取失败",
            severity="error",
            turn_index=1,
        ),
        None,
    )
    timeline.finalize_assistant(1, "鲜味主要来自谷氨酸和呈味核苷酸。")

    entry = timeline.request_history_entries()[0]

    assert entry.answer_summary == "鲜味主要来自谷氨酸和呈味核苷酸。"


def test_request_history_groups_dense_entries_without_losing_turns() -> None:
    timeline = InstrumentedTimeline()
    for turn_index in range(1, 9):
        timeline.add_user(f"请求 {turn_index}", turn_index=turn_index)
        timeline.finalize_assistant(turn_index, f"回答 {turn_index}")

    groups = group_request_history_entries(timeline.request_history_entries(), height=3)

    assert len(groups) == 3
    assert [entry.turn_index for group in groups for entry in group.entries] == list(range(1, 9))
    assert any(len(group.entries) > 1 for group in groups)


def test_request_history_uses_compact_centered_rows_when_space_is_available() -> None:
    timeline = InstrumentedTimeline()
    for turn_index in range(1, 4):
        timeline.add_user(f"请求 {turn_index}", turn_index=turn_index)

    groups = group_request_history_entries(timeline.request_history_entries(), height=15)

    assert [group.row for group in groups] == [6, 7, 8]


def test_dense_request_history_group_cycles_selection() -> None:
    timeline = InstrumentedTimeline()
    for turn_index in range(1, 5):
        timeline.add_user(f"请求 {turn_index}", turn_index=turn_index)
        timeline.finalize_assistant(turn_index, f"回答 {turn_index}")
    rail = RequestHistoryRail()
    rail._entries = timeline.request_history_entries()
    rail._groups = group_request_history_entries(rail._entries, height=1)
    rail._hovered_group = 0

    first = rail._selected_entry(0)
    rail.action_next_group_entry()
    second = rail._selected_entry(0)

    assert second.turn_index == first.turn_index + 1


def test_request_history_navigation_uses_current_turn_and_respects_boundaries() -> None:
    timeline = InstrumentedTimeline()
    for turn_index in (2, 5, 9):
        timeline.add_user(f"请求 {turn_index}", turn_index=turn_index)

    timeline._current_request_turn = 5

    assert timeline.adjacent_request_turn(-1) == 2
    assert timeline.adjacent_request_turn(1) == 9
    timeline._current_request_turn = 2
    assert timeline.adjacent_request_turn(-1) is None
    timeline._current_request_turn = 9
    assert timeline.adjacent_request_turn(1) is None


def test_reapplying_same_tool_detail_state_does_not_rerender_timeline() -> None:
    timeline = InstrumentedTimeline()

    timeline.set_tool_details(False)

    assert timeline.render_timeline_count == 0


def test_tool_detail_toggle_only_refreshes_recent_assistant_blocks() -> None:
    timeline = InstrumentedTimeline()
    for turn_index in range(1, 7):
        timeline._items.append(
            TimelineItem(
                item_id=turn_index,
                role="assistant",
                turn_index=turn_index,
                content=f"answer {turn_index}",
                status="done",
                title="HaAgent",
                tools=[ToolActivity("web_search", "done", "status=success", turn_index)],
            ),
        )
        timeline._blocks[turn_index] = object()

    timeline.set_tool_details(True)

    assert timeline.synced_items == [4, 5, 6]


def test_visible_projection_scans_item_list_once() -> None:
    timeline = ConversationTimeline()
    for turn_index in range(20):
        detail_id = f"detail-{turn_index}"
        timeline.add_presentation_item(
            TimelinePresentationItem(
                kind="activity",
                title=f"activity {turn_index}",
                summary="",
                severity="info",
                turn_index=turn_index,
                detail_id=detail_id,
            ),
            ExpandableDetail(detail_id, [f"detail line {turn_index}"]),
        )
    timeline._items = CountingItemList(timeline._items)

    visible = timeline._visible_items()

    assert len(visible) == 20
    assert timeline._items.iteration_count == 1


def test_pending_flush_uses_item_index_without_scanning_history() -> None:
    timeline = InstrumentedTimeline()
    timeline.update_assistant_delta(1, "answer")
    timeline._items = CountingItemList(timeline._items)

    timeline.flush_pending_assistant_delta()

    assert timeline._items.iteration_count == 0
    assert timeline.synced_items == [timeline._items[0].item_id]


def test_window_moves_in_bounded_steps_and_plain_text_keeps_full_history() -> None:
    timeline = ConversationTimeline()
    for turn_index in range(500):
        timeline.add_user(f"prompt {turn_index}", turn_index=turn_index)
    visible = timeline._visible_items()

    tail = timeline._windowed_items(visible)
    assert [item.turn_index for item in tail] == list(range(450, 500))

    assert timeline._shift_window_earlier(len(visible)) is True
    earlier = timeline._windowed_items(visible)
    assert [item.turn_index for item in earlier] == list(range(425, 475))

    assert timeline._shift_window_later(len(visible)) is True
    assert [item.turn_index for item in timeline._windowed_items(visible)] == list(range(450, 500))
    assert "prompt 0" in timeline.plain_text
    assert "prompt 499" in timeline.plain_text


def test_long_timeline_mounts_only_the_active_window() -> None:
    async def run_test() -> None:
        app = LongTimelineApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            timeline = app.query_one(ConversationTimeline)

            assert len(timeline._blocks) == 50
            assert len(timeline.query(TimelineBlock)) == 50
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 450
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

            overlapping_blocks = {
                block._item.turn_index: block
                for block in timeline._blocks.values()
                if 450 <= block._item.turn_index <= 474
            }

            timeline.watch_scroll_y(10, 0)
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 425
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 474
            assert timeline._window_start == 425
            # `_restore_window_anchor` 通过 call_after_refresh 执行；CI 的 Python 3.12
            # 可能在首个 Pilot pause 后仍未投递该回调，不能把单次 pause 当作完成信号。
            for _ in range(20):
                if not timeline._window_shift_in_progress:
                    break
                await pilot.pause(0.01)
            assert timeline._window_shift_in_progress is False
            assert all(
                timeline._blocks[item_id] is block
                for item_id, block in (
                    (block._item.item_id, block)
                    for block in overlapping_blocks.values()
                )
            )

            timeline.watch_scroll_y(0, max(timeline.max_scroll_y, 2))
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 450
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

            timeline.watch_scroll_y(10, 0)
            await pilot.pause()
            timeline.set_stick_to_bottom(True)
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 450
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

    asyncio.run(run_test())
