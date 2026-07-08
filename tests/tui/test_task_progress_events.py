"""
tests/tui/test_task_progress_events.py - TUI 长任务进度展示测试

验证 task progress 事件能被渲染为安全、简洁的 timeline 文本。
"""

from __future__ import annotations

from haagent.runtime.events.types import TaskProgressEvent
from haagent.tui.widgets.timeline import ConversationTimeline


def test_timeline_adds_task_progress_without_full_reason() -> None:
    timeline = ConversationTimeline()
    event = TaskProgressEvent(
        session_id="session-1",
        turn_index=2,
        model_turn=None,
        event_name="task_step_blocked",
        step_id="step-002",
        title="恢复 worker 子任务",
        status="blocked",
        summary="blocked task step step-002: category=worker_failure",
        owner="worker",
        category="worker_failure",
        suggested_action="retry_worker_or_take_over",
        reason_chars=2000,
    )

    timeline.add_task_progress(event)

    assert "任务进度" in timeline.plain_text
    assert "step-002" in timeline.plain_text
    assert "blocked" in timeline.plain_text
    assert "retry_worker_or_take_over" in timeline.plain_text
    assert "2000 chars" in timeline.plain_text


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

    timeline.add_task_progress(started)
    timeline.add_task_progress(finished)

    assert "任务进度" not in timeline.plain_text
    assert "task_step_started" not in timeline.plain_text
    assert "task_step_finished" not in timeline.plain_text
