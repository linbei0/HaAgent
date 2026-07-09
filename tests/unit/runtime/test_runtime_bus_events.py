"""
tests/unit/runtime/test_runtime_bus_events.py - RuntimeBusEvent 协议测试

验证内部证据总线的强类型事件、dict 往返，以及 UI 投影仍走 RuntimeUiEvent。
"""

from __future__ import annotations

from haagent.runtime.events import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    RuntimeUiEventMapper,
    ToolActivityEvent,
)
from haagent.runtime.events.bus import (
    AssistantDeltaBusEvent,
    AssistantMessageBusEvent,
    LegacyRawBusEvent,
    ToolFailedBusEvent,
    ToolFinishedBusEvent,
    ToolStartedBusEvent,
    bus_event_from_dict,
    bus_event_to_dict,
    coerce_bus_event,
)


def test_assistant_delta_bus_event_round_trips_to_dict() -> None:
    event = AssistantDeltaBusEvent(turn=2, delta="hello")
    payload = bus_event_to_dict(event)

    assert payload == {"event_type": "assistant_delta", "turn": 2, "delta": "hello"}
    restored = bus_event_from_dict(payload)
    assert isinstance(restored, AssistantDeltaBusEvent)
    assert restored == event


def test_tool_finished_bus_event_preserves_full_result() -> None:
    result = {"status": "success", "stdout": "FULL_OUTPUT", "exit_code": 0}
    event = ToolFinishedBusEvent(
        turn=1,
        tool_name="shell",
        args={"command": "echo hi"},
        result=result,
    )
    payload = bus_event_to_dict(event)

    assert payload["event_type"] == "tool_finished"
    assert payload["result"] == result
    assert payload["args"] == {"command": "echo hi"}
    restored = bus_event_from_dict(payload)
    assert isinstance(restored, ToolFinishedBusEvent)
    assert restored.result == result


def test_legacy_raw_bus_event_wraps_unknown_dict() -> None:
    raw = {
        "event_type": "task_step_progress",
        "step_id": "step-001",
        "category": "tool_batch_finished",
        "evidence_count": 1,
    }
    event = coerce_bus_event(raw)

    assert isinstance(event, LegacyRawBusEvent)
    assert event.event_type == "task_step_progress"
    assert bus_event_to_dict(event) == raw


def test_slice_event_types_are_not_legacy() -> None:
    assert isinstance(
        coerce_bus_event({"event_type": "tool_started", "turn": 1, "tool_name": "file_read", "args": {}}),
        ToolStartedBusEvent,
    )
    assert isinstance(
        coerce_bus_event({"event_type": "tool_failed", "turn": 1, "tool_name": "shell", "args": {}, "error": {"type": "X"}}),
        ToolFailedBusEvent,
    )
    assert isinstance(
        coerce_bus_event({"event_type": "assistant_message", "turn": 3, "content": "done"}),
        AssistantMessageBusEvent,
    )


def test_bus_event_projects_to_runtime_ui_event_without_full_tool_result() -> None:
    bus_event = ToolFinishedBusEvent(
        turn=3,
        tool_name="file_read",
        args={"path": "a.txt"},
        result={"status": "ok", "content": "hello-secret-full"},
    )
    ui_event = RuntimeUiEventMapper.to_ui_event(
        bus_event_to_dict(bus_event),
        session_id="session-1",
        turn_index=1,
    )

    assert isinstance(ui_event, ToolActivityEvent)
    assert ui_event.status == "finished"
    assert ui_event.tool_name == "file_read"
    assert "hello-secret-full" not in ui_event.summary


def test_assistant_bus_events_project_to_ui() -> None:
    delta_ui = RuntimeUiEventMapper.to_ui_event(
        bus_event_to_dict(AssistantDeltaBusEvent(turn=2, delta="正在整理")),
        session_id="session-1",
        turn_index=1,
    )
    message_ui = RuntimeUiEventMapper.to_ui_event(
        bus_event_to_dict(AssistantMessageBusEvent(turn=2, content="完成")),
        session_id="session-1",
        turn_index=1,
    )

    assert delta_ui == AssistantDeltaEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=2,
        delta="正在整理",
    )
    assert message_ui == AssistantMessageEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=2,
        content="完成",
    )


def test_working_state_consumes_typed_bus_events(tmp_path) -> None:
    from haagent.runtime.session.agent import ChatTurnResult
    from haagent.runtime.session.working_state import empty_working_state, format_working_state_for_model, update_working_state

    result = ChatTurnResult(
        session_id="session-test",
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episode",
        provider="fake",
        final_response="done",
        verification_status="not_run",
    )
    updated = update_working_state(
        empty_working_state(),
        prompt="Inspect",
        result=result,
        runtime_events=[
            ToolFinishedBusEvent(
                turn=1,
                tool_name="shell",
                args={"command": "echo hi"},
                result={"status": "success", "exit_code": 0, "stdout": "SECRET"},
            ),
            AssistantMessageBusEvent(turn=1, content="Read README."),
        ],
    )
    model_text = format_working_state_for_model(updated)
    assert "actor=assistant tool=shell status=success exit_code=0" in model_text
    assert "SECRET" not in model_text
    assert any("Read README" in item for item in updated.key_findings)
