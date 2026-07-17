"""
tests/tui/test_task_progress_events.py - TUI 长任务进度展示测试

验证 task progress 事件不会以 runtime 原始字段污染 conversation timeline。
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult

from haagent.runtime.events.types import ApprovalStateEvent, TaskProgressEvent
from haagent.tui.presentation.progress import (
    ExpandableDetail,
    TimelinePresentationItem,
    present_approval_state,
    present_task_progress,
)
from haagent.tui.widgets import conversation_timeline as timeline_module
from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.timeline_models import TimelineItem, ToolActivity
from haagent.tui.widgets.timeline_rendering import render_timeline_item
from tests.tui.support import _text


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

    .timeline-body,
    .timeline-tools,
    .timeline-active {
        height: auto;
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
        timeline.add_assistant_message("最终回答", turn_index=1)


class ToolDetailClickTestApp(App[None]):
    CSS = TimelineClickTestApp.CSS

    def compose(self) -> ComposeResult:
        yield ConversationTimeline(id="conversation")

    def on_mount(self) -> None:
        timeline = self.query_one("#conversation", ConversationTimeline)
        timeline.start_assistant_response(turn_index=1)
        timeline.add_tool_activity(ToolActivity("shell", "done", "命令执行完成", 1))
        timeline.finalize_assistant(1, "最终回答")


class NoticeClickTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ConversationTimeline(id="conversation")

    def on_mount(self) -> None:
        timeline = self.query_one("#conversation", ConversationTimeline)
        timeline.add_presentation_item(
            TimelinePresentationItem(
                kind="notice",
                title="任务遇到问题：验证失败",
                summary="建议：修复后重新运行测试",
                severity="error",
                turn_index=1,
                detail_id="notice-detail",
            ),
            ExpandableDetail("notice-detail", ["步骤：step-001", "类别：verification_failed", "证据：1"]),
        )


def test_intermediate_assistant_output_is_retained_but_folded_after_final_message() -> None:
    timeline = ConversationTimeline()

    timeline.start_assistant_response(turn_index=1)
    timeline.finalize_intermediate(1, 1, "完整审查报告")
    timeline.finalize_assistant(1, "最终总结")

    assert "最终总结" in timeline.plain_text
    process_items = [item for item in timeline._items if item.role == "process"]
    assert len(process_items) == 1
    assert process_items[0].content == "完整审查报告"
    assert process_items[0].title == ""
    assert process_items[0].detail_lines == []
    assert process_items[0].detail_id is None
    assert "完整审查报告" not in timeline.plain_text
    assert "过程" not in timeline.plain_text
    assert "已完成 1 步" in timeline.plain_text
    assert "详情：点击" not in timeline.plain_text
    assert timeline.toggle_process_group(1) is True
    assert "完整审查报告" in timeline.plain_text


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
    timeline.add_assistant_message("最终回答", turn_index=1)

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
    assert "已完成 1 步" in text
    assert "web_fetch 失败 2 次，已使用已有上下文继续" not in text
    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert "web_fetch 失败 2 次，已使用已有上下文继续" in text
    assert text.count("web_fetch") == 1
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_late_presentation_item_is_inserted_before_same_turn_assistant_final() -> None:
    timeline = ConversationTimeline()

    timeline.add_user("查一下资料", turn_index=1)
    timeline.add_assistant_message("最终回答", turn_index=1)
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
    assert "已完成 1 步" in text
    assert "web_fetch 失败 2 次" not in text
    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert "web_fetch 失败 2 次" in text
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_replacing_late_presentation_item_keeps_it_before_same_turn_assistant_final() -> None:
    timeline = ConversationTimeline()

    timeline.add_user("查一下资料", turn_index=1)
    timeline.add_assistant_message("最终回答", turn_index=1)
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
    assert "已完成 1 步" in text
    assert "web_fetch 失败 2 次" not in text
    assert timeline.toggle_process_group(1) is True
    text = timeline.plain_text
    assert text.count("web_fetch") == 1
    assert "工具运行失败：web_fetch" not in text
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_notice_details_are_collapsed_by_default_and_expand_in_place() -> None:
    async def run() -> None:
        app = NoticeClickTestApp()
        async with app.run_test(size=(120, 30)) as pilot:
            timeline = app.query_one("#conversation", ConversationTimeline)
            await pilot.pause(0.1)
            assert "类别：verification_failed" not in timeline.plain_text

            await pilot.click(".timeline-notice")
            await pilot.pause(0.1)

            assert "详情：点击收起" in timeline.plain_text
            assert "类别：verification_failed" in timeline.plain_text

            await pilot.click(".timeline-notice")
            await pilot.pause(0.1)
            assert "类别：verification_failed" not in timeline.plain_text

    asyncio.run(run())


def test_process_items_fold_warnings_and_effects_after_final_answer() -> None:
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
    assert "已完成 2 步" in text
    assert "web_fetch 失败 2 次" not in text
    assert "2 个文件有变更" not in text
    assert "请求超时" not in text
    assert text.index("已完成 2 步") < text.index("最终回答")


def test_late_process_item_stays_visible_without_final_answer() -> None:
    timeline = ConversationTimeline()
    timeline.start_assistant_response(turn_index=1)
    timeline.finish_assistant_without_final(1, "")
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="已拒绝：运行命令",
            summary="建议：调整请求或选择其他方案",
            severity="warning",
            turn_index=1,
            detail_id="approval-1",
        ),
        ExpandableDetail("approval-1", ["状态：denied"]),
    )

    assert "已拒绝：运行命令" in timeline.plain_text
    assert "已完成 1 步" in timeline.plain_text
    assert "⌄" in timeline.plain_text


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
    timeline.add_assistant_message("最终回答", turn_index=1)

    assert timeline.toggle_process_group(1) is True
    expanded = timeline.plain_text
    assert "已完成 2 步" in expanded
    assert "web_fetch 失败 2 次" in expanded
    assert "2 个文件有变更" in expanded
    assert "请求超时" not in expanded

    assert timeline.toggle_process_group(1) is True
    collapsed = timeline.plain_text
    assert "已完成 2 步" in collapsed
    assert "web_fetch 失败 2 次" not in collapsed
    assert "2 个文件有变更" not in collapsed


def test_clicking_process_group_expands_the_folded_items() -> None:
    async def run() -> None:
        app = TimelineClickTestApp()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.1)
            timeline = app.query_one("#conversation", ConversationTimeline)
            assert "已完成 2 步" in timeline.plain_text
            assert "web_fetch 失败 2 次" not in timeline.plain_text

            process_block = timeline.query_one(".timeline-process")
            await pilot.click(process_block, offset=(1, 0))
            await pilot.pause(0.1)

            assert "已完成 2 步" in timeline.plain_text
            assert "web_fetch 失败 2 次" in timeline.plain_text
            assert "2 个文件有变更" in timeline.plain_text

    asyncio.run(run())


def test_clicking_process_tool_summary_expands_its_details() -> None:
    async def run() -> None:
        app = ToolDetailClickTestApp()
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause(0.1)
            timeline = app.query_one("#conversation", ConversationTimeline)

            await pilot.click(timeline.query_one(".timeline-process"), offset=(1, 0))
            await pilot.pause(0.1)
            assert "运行命令 ›" in timeline.plain_text

            process_blocks = list(timeline.query(".timeline-process"))
            await pilot.click(process_blocks[-1], offset=(1, 0))
            await pilot.pause(0.1)

            assert "运行命令（shell）成功 · 命令执行完成" in timeline.plain_text

    asyncio.run(run())


def test_terminal_tool_event_updates_running_activity_across_process_items() -> None:
    timeline = ConversationTimeline()
    timeline.start_assistant_response(turn_index=1)
    timeline.add_tool_activity(ToolActivity("shell", "running", "starting tool shell", 1))
    timeline.finalize_intermediate(1, 1, "准备运行命令")
    timeline.add_presentation_item(
        TimelinePresentationItem(
            kind="activity",
            title="其他过程",
            summary="",
            severity="info",
            turn_index=1,
            detail_id="other-process",
        ),
        ExpandableDetail("other-process", ["详情"]),
    )

    timeline.add_tool_activity(ToolActivity("shell", "done", "命令执行完成", 1))

    process_items = timeline._process_items_by_turn[1]
    shell_activities = [
        activity
        for item in process_items
        for activity in item.tools
        if activity.tool_name == "shell"
    ]
    assert shell_activities == [ToolActivity("shell", "done", "命令执行完成", 1)]
    assert process_items[-1].tools == []


def test_approved_tool_flow_renders_as_one_completed_step_without_stale_running_state() -> None:
    timeline = ConversationTimeline()
    approval = {
        "session_id": "session-1",
        "turn_index": 1,
        "model_turn": 1,
        "tool_name": "shell",
        "question": "允许运行命令？",
    }
    timeline.start_assistant_response(turn_index=1)
    timeline.add_tool_activity(ToolActivity("shell", "running", "starting tool shell", 1))
    timeline.finalize_intermediate(1, 1, "准备运行命令")
    requested = present_approval_state(ApprovalStateEvent(**approval, state="requested", approved=None))
    assert requested.timeline_item is not None
    timeline.add_presentation_item(requested.timeline_item, requested.details)
    granted = present_approval_state(ApprovalStateEvent(**approval, state="granted", approved=True))
    assert granted.dismiss_interaction_key is not None
    timeline.dismiss_pending_interaction(granted.dismiss_interaction_key)
    timeline.add_tool_activity(ToolActivity("shell", "done", "命令执行完成", 1))
    timeline.finalize_assistant(1, "最终回答")

    assert "已完成 1 步" in timeline.plain_text
    timeline.toggle_process_group(1)
    expanded = timeline.plain_text
    assert "运行命令 ›" in expanded
    assert "正在运行命令" not in expanded
    assert "starting tool shell" not in expanded
    assert "已允许：运行命令" not in expanded


def test_process_child_items_do_not_show_generic_process_label() -> None:
    from haagent.tui.widgets.timeline_rendering import process_group_title

    items = [
        TimelineItem(1, "process", 1, "让我再试一次，加一些错误处理~", title=""),
        TimelineItem(
            item_id=2,
            role="process",
            turn_index=1,
            content="让我直接搜索~",
            title="",
            tools=[ToolActivity("web_search", "done", "查询汕头逐小时天气", 1)],
        ),
    ]
    rendered = [render_timeline_item(item, show_tool_details=False) for item in items]

    assert rendered[0] == "让我再试一次，加一些错误处理~"
    assert "联网搜索 ›" in rendered[1]
    assert all("过程" not in text and "步骤" not in text for text in rendered)
    assert process_group_title(items, expanded=False) == "已完成 2 步 ›"


def test_completed_process_group_shows_step_count_and_elapsed_time(monkeypatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(timeline_module, "monotonic", lambda: clock[0], raising=False)
    timeline = ConversationTimeline()

    timeline.start_assistant_response(turn_index=1)
    timeline.add_tool_activity(ToolActivity("web_search", "done", "已找到资料", 1))
    clock[0] = 88.0
    timeline.finalize_assistant(1, "整理完成")

    assert "已完成 1 步 · 1分18秒 ›" in timeline.plain_text


def test_running_process_group_refreshes_elapsed_time_without_click(monkeypatch) -> None:
    async def run() -> None:
        clock = [10.0]
        monkeypatch.setattr(timeline_module, "monotonic", lambda: clock[0])
        monkeypatch.setattr(timeline_module, "ELAPSED_REFRESH_INTERVAL_SECONDS", 0.01)
        app = TimelineClickTestApp()
        async with app.run_test(size=(120, 30)) as pilot:
            timeline = app.query_one("#conversation", ConversationTimeline)
            timeline.start_assistant_response(turn_index=2)
            timeline.finalize_intermediate(2, 1, "正在搜索")
            await pilot.pause(0.02)

            assert "已完成 1 步 · 0秒" in _text(app, "#conversation")
            clock[0] = 88.0
            await pilot.pause(0.03)

            assert "已完成 1 步 · 1分18秒" in _text(app, "#conversation")

    asyncio.run(run())


def test_expanded_detail_lines_are_bounded() -> None:
    long_detail = "错误摘要：" + ("x" * 500)
    text = render_timeline_item(
        TimelineItem(
            item_id=1,
            role="effect",
            turn_index=1,
            content="操作已完成",
            title="已执行操作",
            detail_lines=[long_detail],
            expanded=True,
        ),
        show_tool_details=False,
    )
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
