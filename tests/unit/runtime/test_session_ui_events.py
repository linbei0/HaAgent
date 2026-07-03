"""
tests/unit/runtime/test_session_ui_events.py - Session UI 事件发射测试

验证 AgentSession 使用的 UI 事件发射 helper 集中封装 runtime raw event 映射。
"""

from haagent.runtime.events import AssistantMessageEvent, SessionLifecycleEvent
from haagent.runtime.session.ui_events import emit_runtime_ui_event, emit_ui_event, session_started_event


def test_emit_runtime_ui_event_maps_raw_event_for_sink() -> None:
    captured = []

    emit_runtime_ui_event(
        captured.append,
        {"event_type": "assistant_message", "turn": 2, "content": "完成"},
        session_id="session-1",
        turn_index=1,
    )

    assert captured == [AssistantMessageEvent("session-1", 1, 2, "完成")]


def test_emit_ui_event_ignores_missing_sink() -> None:
    emit_ui_event(None, AssistantMessageEvent("session-1", 1, 2, "完成"))


def test_session_started_event_uses_typed_lifecycle_event() -> None:
    event = session_started_event(
        session_id="session-1",
        turn_index=1,
        details={"status": "ready"},
    )

    assert event == SessionLifecycleEvent(
        session_id="session-1",
        turn_index=1,
        state="session_started",
        message="chat session started",
        status="ready",
        details={"status": "ready"},
    )
