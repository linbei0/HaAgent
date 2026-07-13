"""
tests/unit/tui/test_timeline_performance.py - TUI timeline 性能合同测试

覆盖工具详情和长对话场景下的渲染热路径，避免后续改动重新引入全量重算。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.app import App, ComposeResult

import haagent.tui.widgets.timeline_models as timeline_module
from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem
from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.timeline_block import TimelineBlock
from haagent.tui.widgets.timeline_models import TimelineItem, ToolActivity
from haagent.tui.widgets.timeline_rendering import timeline_render_metrics


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
    assert "工具 1 项" in timeline.plain_text
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
    assert [item.turn_index for item in tail] == list(range(300, 500))

    assert timeline._shift_window_earlier(len(visible)) is True
    earlier = timeline._windowed_items(visible)
    assert [item.turn_index for item in earlier] == list(range(200, 400))

    assert timeline._shift_window_later(len(visible)) is True
    assert [item.turn_index for item in timeline._windowed_items(visible)] == list(range(300, 500))
    assert "prompt 0" in timeline.plain_text
    assert "prompt 499" in timeline.plain_text


def test_long_timeline_mounts_only_the_active_window() -> None:
    async def run_test() -> None:
        app = LongTimelineApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            timeline = app.query_one(ConversationTimeline)

            assert len(timeline._blocks) == 200
            assert len(timeline.query(TimelineBlock)) == 200
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 300
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

            timeline.watch_scroll_y(10, 0)
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 200
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 399
            assert timeline._window_start == 200
            assert timeline._window_shift_in_progress is False

            timeline.watch_scroll_y(0, max(timeline.max_scroll_y, 2))
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 300
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

            timeline.watch_scroll_y(10, 0)
            await pilot.pause()
            timeline.set_stick_to_bottom(True)
            await pilot.pause()
            assert min(block._item.turn_index for block in timeline._blocks.values()) == 300
            assert max(block._item.turn_index for block in timeline._blocks.values()) == 499

    asyncio.run(run_test())


def test_timeline_render_metrics_reports_detail_weight() -> None:
    items = [
        TimelineItem(
            item_id=1,
            role="assistant",
            content="answer",
            status="done",
            turn_index=1,
            tools=[
                ToolActivity(
                    tool_name="web_fetch",
                    status="done",
                    summary="status=success",
                    turn_index=1,
                    diagnostics=["结果已压缩 20228 -> 931 字符"],
                ),
                ToolActivity(
                    tool_name="web_search",
                    status="done",
                    summary="status=success",
                    turn_index=1,
                ),
            ],
        ),
        TimelineItem(
            item_id=2,
            role="user",
            content="prompt",
            status="done",
            turn_index=2,
        ),
    ]

    metrics = timeline_render_metrics(items, show_tool_details=True)

    assert metrics.item_count == 2
    assert metrics.tool_count == 2
    assert metrics.diagnostic_count == 1
    assert metrics.detail_line_count == 4
    assert metrics.rendered_character_count > 0
