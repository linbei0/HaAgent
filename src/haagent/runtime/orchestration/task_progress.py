"""
src/haagent/runtime/orchestration/task_progress.py - 长任务进度事件

生成有界 task progress 事件，并把常见失败映射为恢复建议。
"""

from __future__ import annotations

from dataclasses import dataclass

SUMMARY_LIMIT = 180


@dataclass(frozen=True)
class TaskRecoverySuggestion:
    category: str
    reason: str
    suggested_action: str
    step_id: str = ""


def task_plan_created_event(
    *,
    step_id: str,
    title: str,
    owner: str,
    status: str,
    summary: str,
) -> dict[str, object]:
    return {
        "event_type": "task_plan_created",
        "step_id": _bounded(step_id, 80),
        "title": _bounded(title),
        "owner": _bounded(owner, 40),
        "status": _bounded(status, 40),
        "summary": _bounded(summary),
    }


def task_step_progress_event(
    *,
    step_id: str,
    title: str,
    phase: str,
    summary: str,
    owner: str = "main",
    status: str = "running",
    evidence_count: int = 0,
    checkpoint_count: int = 0,
) -> dict[str, object]:
    return {
        "event_type": "task_step_progress",
        "step_id": _bounded(step_id, 80),
        "title": _bounded(title),
        "owner": _bounded(owner, 40),
        "status": _bounded(status, 40),
        "category": _bounded(phase, 80),
        "summary": _bounded(summary),
        "evidence_count": max(0, int(evidence_count)),
        "checkpoint_count": max(0, int(checkpoint_count)),
    }


def task_step_finished_event(
    *,
    step_id: str,
    title: str,
    owner: str,
    evidence_count: int,
    checkpoint_count: int,
) -> dict[str, object]:
    return {
        "event_type": "task_step_finished",
        "step_id": step_id,
        "title": _bounded(title),
        "owner": owner,
        "status": "completed",
        "summary": f"completed task step {step_id}: {_bounded(title)}",
        "evidence_count": max(0, evidence_count),
        "checkpoint_count": max(0, checkpoint_count),
    }


def task_step_blocked_event(
    *,
    step_id: str,
    title: str,
    category: str,
    reason: str,
    suggested_action: str,
    owner: str = "main",
) -> dict[str, object]:
    return {
        "event_type": "task_step_blocked",
        "step_id": step_id,
        "title": _bounded(title),
        "owner": owner,
        "status": "blocked",
        "category": _bounded(category, 80),
        "reason_chars": len(reason),
        "suggested_action": _bounded(suggested_action, 120),
        "summary": f"blocked task step {step_id}: category={_bounded(category, 80)}",
    }


def task_checkpoint_saved_event(
    *,
    step_id: str,
    title: str,
    status: str,
    evidence_count: int,
    checkpoint_count: int,
    owner: str = "main",
) -> dict[str, object]:
    return {
        "event_type": "task_checkpoint_saved",
        "step_id": _bounded(step_id, 80),
        "title": _bounded(title),
        "owner": _bounded(owner, 40),
        "status": _bounded(status, 40),
        "summary": _bounded(f"checkpoint saved for {step_id}: {status}"),
        "evidence_count": max(0, int(evidence_count)),
        "checkpoint_count": max(0, int(checkpoint_count)),
    }


def task_recovery_suggested_event(
    *,
    step_id: str,
    title: str,
    category: str,
    reason: str,
    suggested_action: str,
    owner: str = "main",
) -> dict[str, object]:
    return {
        "event_type": "task_recovery_suggested",
        "step_id": _bounded(step_id, 80),
        "title": _bounded(title),
        "owner": _bounded(owner, 40),
        "status": "blocked",
        "category": _bounded(category, 80),
        "summary": _bounded(f"{category}: reason_chars={len(reason)}"),
        "suggested_action": _bounded(suggested_action, 120),
        "reason_chars": len(reason),
    }


def task_budget_warning_event(
    *,
    step_id: str,
    title: str,
    category: str,
    reason: str,
    suggested_action: str,
    owner: str = "main",
) -> dict[str, object]:
    return {
        "event_type": "task_budget_warning",
        "step_id": _bounded(step_id, 80),
        "title": _bounded(title),
        "owner": _bounded(owner, 40),
        "status": "running",
        "category": _bounded(category, 80),
        "summary": _bounded(f"{category}: reason_chars={len(reason)}"),
        "suggested_action": _bounded(suggested_action, 120),
        "reason_chars": len(reason),
    }


def map_failure_to_recovery(event_or_result: dict[str, object]) -> TaskRecoverySuggestion | None:
    event_type = str(event_or_result.get("event_type", ""))
    if event_type == "approval_denied":
        return TaskRecoverySuggestion(
            category="policy_denied",
            reason="approval denied",
            suggested_action="wait_for_approval_or_replan",
        )
    if event_type == "worker_failed":
        return TaskRecoverySuggestion(
            category="worker_failure",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="retry_worker_or_take_over",
        )
    if event_type == "verification_failed":
        return TaskRecoverySuggestion(
            category="verification_failed",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="repair_and_rerun_verification",
        )
    if event_type == "context_build_failed":
        return TaskRecoverySuggestion(
            category="context_error",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="reduce_context_or_fix_task_spec",
        )
    if event_type == "loop_limit":
        return TaskRecoverySuggestion(
            category="loop_limit",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="checkpoint_and_resume",
        )
    if event_type == "model_failed":
        return TaskRecoverySuggestion(
            category="model_error",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="retry_or_switch_model",
        )
    if event_type == "run_cancelled":
        return TaskRecoverySuggestion(
            category="cancelled",
            reason=_bounded(str(event_or_result.get("reason", ""))),
            suggested_action="resume_when_ready",
        )
    if event_type != "tool_failed":
        return None
    error = event_or_result.get("error") if isinstance(event_or_result.get("error"), dict) else {}
    error_type = str(error.get("type", "unknown")).lower()
    message = str(error.get("message", ""))
    if event_or_result.get("execution_state") == "unknown":
        return TaskRecoverySuggestion(
            category="tool_execution_unknown",
            reason=_bounded(message),
            suggested_action="inspect_state_before_retry",
        )
    if "timeout" in error_type or "timed out" in message.lower():
        return TaskRecoverySuggestion(
            category="tool_timeout",
            reason=_bounded(message),
            suggested_action="retry_with_narrower_command",
        )
    if error_type in {"invalid_arguments", "schema_error", "argument_error", "tool_argument_invalid"}:
        return TaskRecoverySuggestion(
            category="model_format_error",
            reason=_bounded(message),
            suggested_action="correct_tool_arguments",
        )
    return TaskRecoverySuggestion(
        category="tool_failure",
        reason=_bounded(message),
        suggested_action="inspect_failure_and_replan",
    )


def _bounded(value: str, limit: int = SUMMARY_LIMIT) -> str:
    text = value.replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
