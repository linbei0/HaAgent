"""
tests/tui/test_task_progress_events.py - TUI 长任务进度展示测试

验证 task progress 事件不会以 runtime 原始字段污染 conversation timeline。
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from haagent.runtime.events.types import TaskProgressEvent
from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem, present_task_progress
from haagent.tui.widgets.timeline import ConversationTimeline, TimelineItem


class TimelineClickTestApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #conversation {
        width: 1fr;
        height: 1fr;
    }

    .timeline-item {
        height: auto;
    }

    .timeline-header {
        height: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield ConversationTimeline(id="conversation")

    def on_mount(self) -> None:
        timeline = self.query_one("#conversation", ConversationTimeline)
        timeline.add_presentation_item(
            TimelinePresentationItem(
                kind="activity",
                title="web_fetch 失败 2 次，已使用已有上下文继续",
                summary="",
                severity="warning",
                turn_index=1,
                detail_id="detail-1",
            ),
            ExpandableDetail("detail-1", ["失败原因：请求超时"]),
        )
        timeline.add_presentation_item(
            TimelinePresentationItem(
                kind="effect",
                title="2 个文件有变更",
                summary="",
                severity="info",
                turn_index=1,
                detail_id="detail-2",
            ),
            ExpandableDetail("detail-2", ["文件：src/example.py"]),
        )


def test_plain_task_progress_projection_does_not_add_timeline_item() -> None:
    timeline = ConversationTimeline()
    event = TaskProgressEvent(
        session_id="s",
        turn_index=1,
        model_turn=1,
        event_name="task_step_progress",
        step_id="step-001",
        title="你好",
        status="running",
        summary="model turn started",
        owner="main",
        category="model_turn_started",
        suggested_action="",
        evidence_count=0,
        checkpoint_count=0,
        reason_chars=0,
    )

    presentation = present_task_progress(event)
    if presentation.timeline_item is not None:
        timeline.add_presentation_item(presentation.timeline_item, presentation.details)

    assert "任务进度" not in timeline.plain_text
    assert "step-001" not in timeline.plain_text
    assert "model_turn_started" not in timeline.plain_text


def test_timeline_accepts_notice_and_effect_presentation_items() -> None:
    timeline = ConversationTimeline()

    timeline.add_notice(
        "任务遇到问题：验证失败",
        "建议：修复后重新运行测试",
        turn_index=1,
        detail_id="detail-1",
        detail_lines=["步骤：step-001", "类别：verification_failed"],
    )
    timeline.add_effect_summary(
        "已修改文件",
        "2 个文件有变更",
        turn_index=1,
        detail_id="detail-2",
        detail_lines=["工具：apply_patch"],
    )

    text = timeline.plain_text
    assert "任务遇到问题：验证失败" in text
    assert "建议：修复后重新运行测试" in text
    assert "已处理 1 项 >" in text
    assert "已修改文件" not in text
    assert "2 个文件有变更" not in text
    assert "类别：verification_failed" not in text
    assert "工具：apply_patch" not in text

    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert "已修改文件" in text
    assert "2 个文件有变更" in text
    assert "工具：apply_patch" not in text


def test_timeline_adds_projected_presentation_items() -> None:
    timeline = ConversationTimeline()

    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="notice",
            title="任务遇到问题：验证失败",
            summary="建议：修复后重新运行测试",
            severity="error",
            turn_index=1,
            detail_id="detail-1",
        ),
        ExpandableDetail("detail-1", ["类别：verification_failed"]),
    )

    text = timeline.plain_text
    assert "任务遇到问题：验证失败" in text
    assert "建议：修复后重新运行测试" in text
    assert "类别：verification_failed" not in text


def test_timeline_updates_grouped_notice_without_moving_assistant_final() -> None:
    timeline = ConversationTimeline()

    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="notice",
            title="工具运行失败：web_fetch",
            summary="建议：查看错误摘要后重试或调整命令",
            severity="error",
            turn_index=1,
            detail_id="tool:1:web_fetch:failed",
        ),
        ExpandableDetail("tool:1:web_fetch:failed", ["工具：web_fetch", "状态：failed"]),
    )
    timeline._items.append(
        TimelineItem(item_id=next(timeline._ids), role="assistant", turn_index=1, content="最终回答")
    )

    replaced = timeline.replace_presentation_item(
        TimelinePresentationItem(
            kind="notice",
            title="web_fetch 失败 2 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="tool:1:web_fetch:failed",
        ),
        ExpandableDetail("tool:1:web_fetch:failed", ["工具：web_fetch", "失败次数：2"]),
    )

    text = timeline.plain_text
    assert replaced is True
    assert "web_fetch 失败 2 次，已使用已有上下文继续" in text
    assert text.count("web_fetch") == 1
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_late_presentation_item_is_inserted_before_same_turn_assistant_final() -> None:
    timeline = ConversationTimeline()

    timeline._items.append(TimelineItem(item_id=next(timeline._ids), role="user", turn_index=1, content="查一下资料"))
    timeline._items.append(
        TimelineItem(item_id=next(timeline._ids), role="assistant", turn_index=1, content="最终回答")
    )
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="web_fetch 失败 2 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="tool:1:web_fetch:failed",
        ),
        ExpandableDetail("tool:1:web_fetch:failed", ["工具：web_fetch", "失败次数：2"]),
    )

    text = timeline.plain_text
    assert "已处理 1 项 >" in text
    assert "web_fetch 失败 2 次" not in text
    assert text.index("已处理 1 项") < text.index("最终回答")

    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_replacing_late_presentation_item_keeps_it_before_same_turn_assistant_final() -> None:
    timeline = ConversationTimeline()

    timeline._items.append(TimelineItem(item_id=next(timeline._ids), role="user", turn_index=1, content="查一下资料"))
    timeline._items.append(
        TimelineItem(item_id=next(timeline._ids), role="assistant", turn_index=1, content="最终回答")
    )
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="工具运行失败：web_fetch",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="tool:1:web_fetch:failed",
        ),
        ExpandableDetail("tool:1:web_fetch:failed", ["工具：web_fetch"]),
    )

    replaced = timeline.replace_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="web_fetch 失败 2 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="tool:1:web_fetch:failed",
        ),
        ExpandableDetail("tool:1:web_fetch:failed", ["工具：web_fetch", "失败次数：2"]),
    )

    text = timeline.plain_text
    assert replaced is True
    assert "已处理 1 项 >" in text
    assert "web_fetch" not in text
    assert text.index("已处理 1 项") < text.index("最终回答")

    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert text.count("web_fetch") == 1
    assert "工具运行失败：web_fetch" not in text
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_notice_details_are_collapsed_by_default_and_expand_in_place() -> None:
    timeline = ConversationTimeline()
    timeline.add_notice(
        "任务遇到问题：验证失败",
        "建议：修复后重新运行测试",
        turn_index=1,
        detail_id="detail-1",
        detail_lines=["步骤：step-001", "类别：verification_failed", "证据：1"],
    )

    collapsed = timeline.plain_text
    assert "类别：verification_failed" not in collapsed

    item_id = timeline._items[0].item_id
    assert timeline.toggle_detail(item_id) is True

    expanded = timeline.plain_text
    assert "详情：点击收起" in expanded
    assert "类别：verification_failed" in expanded

    assert timeline.toggle_detail(item_id) is True
    assert "类别：verification_failed" not in timeline.plain_text


def test_process_items_are_folded_into_turn_summary_by_default() -> None:
    timeline = ConversationTimeline()
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="web_fetch 失败 2 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="detail-1",
        ),
        ExpandableDetail("detail-1", ["失败原因：请求超时"]),
    )
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="effect",
            title="2 个文件有变更",
            summary="",
            severity="info",
            turn_index=1,
            detail_id="detail-2",
        ),
        ExpandableDetail("detail-2", ["文件：src/example.py"]),
    )
    timeline.add_assistant_message("最终回答", turn_index=1)

    text = timeline.plain_text
    assert "已处理 2 项 >" in text
    assert "web_fetch 失败 2 次" not in text
    assert "2 个文件有变更" not in text
    assert "请求超时" not in text
    assert text.index("已处理 2 项") < text.index("最终回答")


def test_process_group_toggle_expands_and_collapses_all_process_items() -> None:
    timeline = ConversationTimeline()
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="web_fetch 失败 2 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=1,
            detail_id="detail-1",
        ),
        ExpandableDetail("detail-1", ["失败原因：请求超时"]),
    )
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="effect",
            title="2 个文件有变更",
            summary="",
            severity="info",
            turn_index=1,
            detail_id="detail-2",
        ),
        ExpandableDetail("detail-2", ["文件：src/example.py"]),
    )

    assert timeline.toggle_process_group(1) is True
    expanded = timeline.plain_text
    assert "已处理 2 项 v" in expanded
    assert "web_fetch 失败 2 次" in expanded
    assert "2 个文件有变更" in expanded
    assert "请求超时" not in expanded

    assert timeline.toggle_process_group(1) is True
    collapsed = timeline.plain_text
    assert "已处理 2 项 >" in collapsed
    assert "web_fetch 失败 2 次" not in collapsed
    assert "2 个文件有变更" not in collapsed


def test_clicking_process_group_expands_the_folded_items() -> None:
    async def run() -> None:
        app = TimelineClickTestApp()
        async with app.run_test(size=(120, 30)) as pilot:
            timeline = app.query_one("#conversation", ConversationTimeline)
            assert "已处理 2 项 >" in timeline.plain_text
            assert "web_fetch 失败 2 次" not in timeline.plain_text

            await pilot.click(".timeline-process")
            await pilot.pause(0.1)

            assert "已处理 2 项 v" in timeline.plain_text
            assert "web_fetch 失败 2 次" in timeline.plain_text
            assert "2 个文件有变更" in timeline.plain_text

    asyncio.run(run())


def test_expanded_detail_lines_are_bounded() -> None:
    timeline = ConversationTimeline()
    long_detail = "错误摘要：" + ("x" * 500)
    timeline.add_effect_summary(
        "已执行操作",
        "操作已完成",
        turn_index=1,
        detail_id="detail-1",
        detail_lines=[long_detail],
    )

    timeline.toggle_detail(timeline._items[0].item_id)

    text = timeline.plain_text
    assert long_detail not in text
    assert "..." in text


def test_timeline_suppresses_plain_turn_lifecycle_task_progress() -> None:
    timeline = ConversationTimeline()
    started = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_step_started",
        step_id="step-001",
        title="你好",
        status="running",
        summary="started task step step-001: 你好",
        category="none",
        suggested_action="none",
    )
    finished = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_step_finished",
        step_id="step-001",
        title="你好",
        status="completed",
        summary="completed task step step-001: 你好",
        category="none",
        suggested_action="none",
        evidence_count=1,
        checkpoint_count=1,
    )

    for event in (started, finished):
        presentation = present_task_progress(event)
        if presentation.timeline_item is not None:
            timeline.add_presentation_item(presentation.timeline_item, presentation.details)

    assert "任务进度" not in timeline.plain_text
    assert "task_step_started" not in timeline.plain_text
    assert "task_step_finished" not in timeline.plain_text
