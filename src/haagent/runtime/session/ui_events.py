"""
src/haagent/runtime/session/ui_events.py - Session UI 事件发射

为 AgentSession 封装 RuntimeUiEvent sink、raw runtime event 映射和 session 生命周期事件。
"""

from __future__ import annotations

from collections.abc import Callable

from haagent.runtime.events import (
    RuntimeUiEvent,
    RuntimeUiEventMapper,
    failure_notice_event,
    memory_candidates_created_event,
    memory_extraction_warning_event,
    session_lifecycle_event,
)
from haagent.runtime.events.types import SessionLifecycleEvent


RuntimeUiEventSink = Callable[[RuntimeUiEvent], None] | None


def emit_ui_event(event_sink: RuntimeUiEventSink, event: RuntimeUiEvent) -> None:
    if event_sink is None:
        return
    event_sink(event)


def emit_runtime_ui_event(
    event_sink: RuntimeUiEventSink,
    event: dict[str, object],
    *,
    session_id: str,
    turn_index: int,
) -> None:
    emit_ui_event(
        event_sink,
        RuntimeUiEventMapper.to_ui_event(event, session_id=session_id, turn_index=turn_index),
    )


def session_started_event(
    *,
    session_id: str,
    turn_index: int,
    details: dict[str, object] | None = None,
) -> SessionLifecycleEvent:
    return session_lifecycle_event(
        session_id=session_id,
        turn_index=turn_index,
        state="session_started",
        message="chat session started",
        details=details,
    )


def session_finished_event(
    *,
    session_id: str,
    turn_index: int,
    details: dict[str, object] | None = None,
) -> SessionLifecycleEvent:
    return session_lifecycle_event(
        session_id=session_id,
        turn_index=turn_index,
        state="session_finished",
        message="chat session finished",
        details=details,
    )


def turn_started_event(
    *,
    session_id: str,
    turn_index: int,
    details: dict[str, object] | None = None,
) -> SessionLifecycleEvent:
    return session_lifecycle_event(
        session_id=session_id,
        turn_index=turn_index,
        state="turn_started",
        message="chat turn started",
        details=details,
    )


def turn_finished_event(
    *,
    session_id: str,
    turn_index: int,
    details: dict[str, object] | None = None,
) -> SessionLifecycleEvent:
    return session_lifecycle_event(
        session_id=session_id,
        turn_index=turn_index,
        state="turn_finished",
        message="chat turn finished",
        details=details,
    )


__all__ = [
    "RuntimeUiEventSink",
    "emit_runtime_ui_event",
    "emit_ui_event",
    "failure_notice_event",
    "memory_candidates_created_event",
    "memory_extraction_warning_event",
    "session_finished_event",
    "session_started_event",
    "turn_finished_event",
    "turn_started_event",
]
