"""
tests/tui/test_task_progress_events.py - TUI 长任务进度展示测试

验证 task progress 事件不会以 runtime 原始字段污染 conversation timeline。
"""

from __future__ import annotations

from haagent.runtime.events.types import TaskProgressEvent
from haagent.tui.presentation.progress import ExpandableDetail, TimelinePresentationItem, present_task_progress
from haagent.tui.widgets.timeline import ConversationTimeline


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
        "建议：修复后重新运行测试\n详情：按 Enter 展开",
        turn_index=1,
        detail_id="detail-1",
        detail_lines=["步骤：step-001", "类别：verification_failed"],
    )
    timeline.add_effect_summary(
        "已修改文件",
        "2 个文件有变更\n详情：按 Enter 展开",
        turn_index=1,
        detail_id="detail-2",
        detail_lines=["工具：apply_patch"],
    )

    text = timeline.plain_text
    assert "任务遇到问题：验证失败" in text
    assert "建议：修复后重新运行测试" in text
    assert "已修改文件" in text
    assert "2 个文件有变更" in text
    assert "类别：verification_failed" not in text
    assert "工具：apply_patch" not in text


def test_timeline_adds_projected_presentation_items() -> None:
    timeline = ConversationTimeline()

    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="notice",
            title="任务遇到问题：验证失败",
            summary="建议：修复后重新运行测试\n详情：按 Enter 展开",
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


def test_notice_details_are_collapsed_by_default_and_expand_in_place() -> None:
    timeline = ConversationTimeline()
    timeline.add_notice(
        "任务遇到问题：验证失败",
        "建议：修复后重新运行测试\n详情：按 Enter 展开",
        turn_index=1,
        detail_id="detail-1",
        detail_lines=["步骤：step-001", "类别：verification_failed", "证据：1"],
    )

    collapsed = timeline.plain_text
    assert "类别：verification_failed" not in collapsed

    item_id = timeline._items[0].item_id
    assert timeline.toggle_detail(item_id) is True

    expanded = timeline.plain_text
    assert "详情：按 Enter 收起" in expanded
    assert "类别：verification_failed" in expanded

    assert timeline.toggle_detail(item_id) is True
    assert "类别：verification_failed" not in timeline.plain_text


def test_expanded_detail_lines_are_bounded() -> None:
    timeline = ConversationTimeline()
    long_detail = "错误摘要：" + ("x" * 500)
    timeline.add_effect_summary(
        "已执行操作",
        "操作已完成\n详情：按 Enter 展开",
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
