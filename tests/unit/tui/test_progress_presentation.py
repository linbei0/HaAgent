"""
tests/unit/tui/test_progress_presentation.py - TUI 进度展示投影测试

验证 runtime task/tool 事件会被转换为用户可理解的状态、摘要和有限详情。
"""

from __future__ import annotations

from haagent.runtime.events.types import TaskProgressEvent, ToolActivityEvent
from haagent.tui.presentation.progress import present_task_progress, present_tool_activity


def _task_event(event_name: str, **overrides: object) -> TaskProgressEvent:
    values = {
        "session_id": "s",
        "turn_index": 1,
        "model_turn": 1,
        "event_name": event_name,
        "step_id": "step-001",
        "title": "你好",
        "status": "running",
        "summary": "",
        "owner": "main",
        "category": "",
        "suggested_action": "",
        "evidence_count": 0,
        "checkpoint_count": 0,
        "reason_chars": 0,
    }
    values.update(overrides)
    return TaskProgressEvent(**values)


def _tool_event(status: str, **overrides: object) -> ToolActivityEvent:
    values = {
        "session_id": "s",
        "turn_index": 1,
        "model_turn": 1,
        "tool_name": "file_read",
        "status": status,
        "summary": "",
        "args_summary": {},
        "result_status": "",
        "error_type": "",
        "error_message": "",
    }
    values.update(overrides)
    return ToolActivityEvent(**values)


def test_plain_task_step_progress_is_not_displayed() -> None:
    presentation = present_task_progress(
        _task_event(
            "task_step_progress",
            summary="model turn started",
            category="model_turn_started",
        )
    )

    assert presentation.status_line is None
    assert presentation.timeline_item is None
    assert presentation.details is None


def test_recovery_suggestion_becomes_actionable_notice() -> None:
    presentation = present_task_progress(
        _task_event(
            "task_recovery_suggested",
            title="运行测试",
            status="blocked",
            category="verification_failed",
            suggested_action="修复后重新运行测试",
            evidence_count=1,
            checkpoint_count=1,
            reason_chars=120,
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "activity"
    assert presentation.timeline_item.title == "任务遇到问题：验证失败"
    assert "修复后重新运行测试" in presentation.timeline_item.summary
    assert presentation.details is not None
    assert "verification_failed" in "\n".join(presentation.details.lines)


def test_started_tools_update_ephemeral_status_only() -> None:
    for tool_name in ("file_read", "web_fetch", "shell", "code_run"):
        presentation = present_tool_activity(_tool_event("started", tool_name=tool_name))

        assert presentation.status_line is not None
        assert presentation.timeline_item is None


def test_apply_patch_success_becomes_effect_summary() -> None:
    presentation = present_tool_activity(
        _tool_event(
            "finished",
            tool_name="apply_patch",
            summary="modified 2 files",
            result_status="success",
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "effect"
    assert presentation.timeline_item.title == "已修改文件"
    assert "2 个文件有变更" in presentation.timeline_item.summary
    assert "modified 2 files" not in presentation.timeline_item.summary
    assert "详情：" not in presentation.timeline_item.summary


def test_file_write_success_becomes_effect_summary() -> None:
    presentation = present_tool_activity(
        _tool_event(
            "finished",
            tool_name="file_write",
            summary="wrote notes.md",
            result_status="success",
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "effect"
    assert presentation.timeline_item.title == "已写入文件"
    assert "文件已写入" in presentation.timeline_item.summary


def test_shell_file_modification_becomes_effect_summary() -> None:
    presentation = present_tool_activity(
        _tool_event(
            "finished",
            tool_name="shell",
            summary="modified 2 files",
            result_status="success",
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "effect"
    assert presentation.timeline_item.title == "已执行操作"
    assert "2 个文件有变更" in presentation.timeline_item.summary


def test_code_run_file_modification_becomes_effect_summary() -> None:
    presentation = present_tool_activity(
        _tool_event(
            "finished",
            tool_name="code_run",
            summary="modified 1 file",
            result_status="success",
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "effect"
    assert "1 个文件有变更" in presentation.timeline_item.summary


def test_shell_success_without_file_effect_does_not_persist() -> None:
    presentation = present_tool_activity(
        _tool_event(
            "finished",
            tool_name="shell",
            summary="tests passed",
            result_status="success",
        )
    )

    assert presentation.status_line is None
    assert presentation.timeline_item is None
    assert presentation.details is None


def test_tool_failure_becomes_actionable_notice_without_full_error() -> None:
    long_error = "x" * 2000
    presentation = present_tool_activity(
        _tool_event(
            "failed",
            tool_name="shell",
            error_type="command_failed",
            error_message=long_error,
        )
    )

    assert presentation.timeline_item is not None
    assert presentation.timeline_item.kind == "activity"
    assert presentation.details is not None
    assert long_error not in "\n".join(presentation.details.lines)
