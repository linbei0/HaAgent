"""
src/haagent/runtime/events/factories.py - Runtime UI 事件工厂

封装非 raw runtime dict 来源的 session、memory 与 failure UI 事件创建。
"""

from __future__ import annotations

from typing import Literal

from haagent.runtime.events.formatting import int_value, summary_value
from haagent.runtime.events.types import FailureNoticeEvent, MemoryNoticeEvent, SessionLifecycleEvent, WarningNoticeEvent


def memory_candidates_created_event(
    *,
    session_id: str,
    turn_index: int,
    count: int,
    message: str,
) -> MemoryNoticeEvent:
    return MemoryNoticeEvent(session_id=session_id, turn_index=turn_index, count=count, message=message)


def memory_extraction_warning_event(
    *,
    session_id: str,
    turn_index: int,
    status: str,
    reason: str,
    message: str,
) -> WarningNoticeEvent:
    return WarningNoticeEvent(
        session_id=session_id,
        turn_index=turn_index,
        title="Memory warning",
        message=message,
        details={"status": status, "reason": reason},
    )


def session_lifecycle_event(
    *,
    session_id: str,
    turn_index: int,
    state: Literal["session_started", "session_finished", "turn_started", "turn_finished"],
    message: str,
    details: dict[str, object] | None = None,
) -> SessionLifecycleEvent:
    detail_values = details or {}
    return SessionLifecycleEvent(
        session_id=session_id,
        turn_index=turn_index,
        state=state,
        message=message,
        status=str(detail_values.get("status", "")),
        prompt=str(detail_values.get("prompt", "")),
        episode_path=str(detail_values.get("episode_path", "")),
        runtime_event_count=int_value(detail_values.get("runtime_event_count")),
        details=detail_values,
    )


def failure_notice_event(
    *,
    session_id: str,
    turn_index: int,
    status: str,
    failed_stage: str,
    failure_category: str,
    reason: str,
    episode_path: str,
) -> FailureNoticeEvent:
    return FailureNoticeEvent(
        session_id=session_id,
        turn_index=turn_index,
        status=status,
        failed_stage=summary_value(failed_stage),
        failure_category=summary_value(failure_category),
        reason=summary_value(reason),
        episode_path=episode_path,
    )
