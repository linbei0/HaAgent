"""
src/haagent/tui/runtime_events.py - TUI 运行时事件处理

消费 RuntimeUiEvent 强类型协议并更新 Textual 应用状态。
"""

from __future__ import annotations

from collections.abc import Callable

from haagent.runtime.events import (
    RUNTIME_UI_EVENT_TYPES,
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RuntimeUiEvent,
    SessionLifecycleEvent,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.tui.design.failures import failure_from_payload


RuntimeUiEventHandler = Callable[[object, RuntimeUiEvent], None]


def handle_runtime_ui_event(app, event: RuntimeUiEvent) -> None:
    handler = RUNTIME_UI_EVENT_HANDLERS.get(type(event))
    if handler is None:
        raise TypeError(f"unsupported RuntimeUiEvent: {type(event).__name__}")
    handler(app, event)
    app._refresh()


def _handle_assistant_delta(app, event: RuntimeUiEvent) -> None:
    typed = _as_event(event, AssistantDeltaEvent)
    app._merge_assistant_delta(typed.turn_index, typed.delta)


def _handle_assistant_message(app, event: RuntimeUiEvent) -> None:
    typed = _as_event(event, AssistantMessageEvent)
    app._finalize_assistant_message(typed.turn_index, typed.content)


def _handle_tool_activity(app, event: ToolActivityEvent) -> None:
    status = "running"
    if event.status == "finished":
        status = "done"
    elif event.status == "failed":
        status = "failed"
    app._record_tool_activity(event.turn_index, event.tool_name, status, event.summary or status)


def _handle_approval_state(app, event: ApprovalStateEvent) -> None:
    label = "文件改动" if event.approval_kind == "edit_diff" else "审批"
    if event.state == "requested":
        app._state = "waiting approval"
        summary = "文件改动等待审批" if event.approval_kind == "edit_diff" else "等待审批"
        app._record_tool_activity(event.turn_index, event.tool_name, "approval", summary)
        return
    if event.state == "granted":
        app._state = "running"
        app._record_tool_activity(event.turn_index, event.tool_name, "running", f"{label}已允许")
        app._append_line(f"{label}已允许：{event.tool_name}")
        return
    app._record_tool_activity(event.turn_index, event.tool_name, "failed", f"{label}已拒绝")
    app._append_line(f"{label}已拒绝：{event.tool_name}")


def _handle_user_input_state(app, event: UserInputStateEvent) -> None:
    if event.state == "requested":
        app._state = "waiting input"
        app._set_answer_required(event.question)
        app._append_block("Answer required", event.question)
        return
    app._state = "running"
    if event.approved is False:
        app._append_line(f"回答已取消：{event.tool_name}")
    else:
        app._append_line(f"回答已提交：{event.tool_name}")


def _handle_memory_notice(app, event: MemoryNoticeEvent) -> None:
    message = event.message or "发现可记忆候选，已放入候选队列，等待你确认。"
    app._append_block("Memory", message)
    app._memory_notice = message
    app._memory_mode = True
    app._memory_detail_mode = False
    app._load_memory_candidates(silent=True)


def _handle_failure_notice(app, event: FailureNoticeEvent) -> None:
    app._state = "failed"
    app._last_failure = failure_from_payload(
        {
            "status": event.status,
            "failed_stage": event.failed_stage,
            "failure_category": event.failure_category,
            "reason": event.reason,
            "episode_path": event.episode_path,
        },
        event.reason,
    )
    app._append_block("Failure", app._last_failure.block_text())
    app._finalize_streaming_assistant_if_needed()


def _handle_warning_notice(app, event: WarningNoticeEvent) -> None:
    app._append_block(event.title, event.message)


def _handle_session_lifecycle(app, event: SessionLifecycleEvent) -> None:
    return


def _as_event(event: RuntimeUiEvent, event_type):
    if not isinstance(event, event_type):
        raise TypeError(f"expected {event_type.__name__}, got {type(event).__name__}")
    return event


RUNTIME_UI_EVENT_HANDLERS: dict[type[object], RuntimeUiEventHandler] = {
    AssistantDeltaEvent: _handle_assistant_delta,
    AssistantMessageEvent: _handle_assistant_message,
    ToolActivityEvent: _handle_tool_activity,
    ApprovalStateEvent: _handle_approval_state,
    UserInputStateEvent: _handle_user_input_state,
    MemoryNoticeEvent: _handle_memory_notice,
    WarningNoticeEvent: _handle_warning_notice,
    FailureNoticeEvent: _handle_failure_notice,
    SessionLifecycleEvent: _handle_session_lifecycle,
}

if set(RUNTIME_UI_EVENT_HANDLERS) != set(RUNTIME_UI_EVENT_TYPES):
    missing = set(RUNTIME_UI_EVENT_TYPES) - set(RUNTIME_UI_EVENT_HANDLERS)
    extra = set(RUNTIME_UI_EVENT_HANDLERS) - set(RUNTIME_UI_EVENT_TYPES)
    raise RuntimeError(f"Runtime UI event handler registry mismatch: missing={missing}, extra={extra}")
