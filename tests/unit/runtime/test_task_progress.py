"""
tests/unit/runtime/test_task_progress.py - 长任务进度与恢复映射测试

验证 task progress 事件只携带有界摘要，并为常见失败生成恢复建议。
"""

from __future__ import annotations

from haagent.runtime.orchestration.task_progress import (
    map_failure_to_recovery,
    task_step_blocked_event,
    task_step_finished_event,
)
from haagent.runtime.events.types import TaskProgressEvent
from haagent.runtime.events.ui_mapper import RuntimeUiEventMapper


def test_task_progress_event_payload_is_bounded_for_ui() -> None:
    secret = "SECRET_FULL_STDOUT_SHOULD_NOT_RENDER"
    event = task_step_blocked_event(
        step_id="step-002",
        title="运行验证",
        category="tool_timeout",
        reason=secret * 20,
        suggested_action="retry_with_narrower_command",
    )

    ui_event = RuntimeUiEventMapper.to_ui_event(event, session_id="session-1", turn_index=3)

    assert isinstance(ui_event, TaskProgressEvent)
    assert ui_event.event_name == "task_step_blocked"
    assert ui_event.step_id == "step-002"
    assert ui_event.status == "blocked"
    assert ui_event.category == "tool_timeout"
    assert ui_event.suggested_action == "retry_with_narrower_command"
    assert secret not in ui_event.summary
    assert ui_event.reason_chars > 0


def test_task_finished_event_maps_to_ui_event() -> None:
    event = task_step_finished_event(
        step_id="step-001",
        title="建立任务账本",
        owner="main",
        evidence_count=2,
        checkpoint_count=1,
    )

    ui_event = RuntimeUiEventMapper.to_ui_event(event, session_id="session-1", turn_index=1)

    assert isinstance(ui_event, TaskProgressEvent)
    assert ui_event.status == "completed"
    assert ui_event.owner == "main"
    assert ui_event.evidence_count == 2
    assert ui_event.checkpoint_count == 1


def test_failure_to_recovery_mapping_handles_common_failures() -> None:
    assert map_failure_to_recovery(
        {
            "event_type": "tool_failed",
            "tool_name": "shell",
            "error": {"type": "timeout", "message": "command timed out"},
        },
    ).suggested_action == "retry_with_narrower_command"

    assert map_failure_to_recovery(
        {
            "event_type": "tool_failed",
            "tool_name": "apply_patch",
            "error": {"type": "invalid_arguments", "message": "bad args"},
        },
    ).suggested_action == "correct_tool_arguments"

    assert map_failure_to_recovery(
        {
            "event_type": "approval_denied",
            "tool_name": "shell",
        },
    ).suggested_action == "wait_for_approval_or_replan"

    assert map_failure_to_recovery(
        {
            "event_type": "worker_failed",
            "agent_id": "worker-a",
            "reason": "worker failed",
        },
    ).suggested_action == "retry_worker_or_take_over"

