"""
src/haagent/tui/widgets/task_progress_visibility.py - TUI 任务进度展示规则

集中判断哪些长任务进度适合进入主对话时间线，避免普通对话生命周期事件污染界面。
"""

from __future__ import annotations

from haagent.runtime.events.types import TaskProgressEvent


def should_show_task_progress(event: TaskProgressEvent) -> bool:
    category = meaningful_task_progress_value(event.category)
    suggested_action = meaningful_task_progress_value(event.suggested_action)
    prominent_events = {
        "task_step_progress",
        "task_step_blocked",
        "task_checkpoint_saved",
        "task_recovery_suggested",
        "task_budget_warning",
    }
    if event.event_name in prominent_events:
        return True
    if event.owner == "worker":
        return True
    if category or suggested_action or event.reason_chars:
        return True
    if event.event_name == "task_step_finished":
        return event.evidence_count > 1 or event.checkpoint_count > 1
    if event.event_name == "task_step_started":
        return event.evidence_count > 0 or event.checkpoint_count > 0
    return False


def meaningful_task_progress_value(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "n/a"}:
        return ""
    return value.strip()
