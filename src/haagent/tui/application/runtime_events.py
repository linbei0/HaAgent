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
    AssistantIntermediateEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RuntimeUiEvent,
    SessionLifecycleEvent,
    TaskProgressEvent,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.tui.design.failures import failure_from_payload
from haagent.tui.presentation.progress import (
    ProgressPresentation,
    present_approval_state,
    present_grouped_task_problem,
    present_grouped_tool_failure,
    present_task_progress,
    present_tool_activity,
    present_user_input_state,
)
from haagent.tui.widgets.conversation_timeline import ConversationTimeline


RuntimeUiEventHandler = Callable[[object, RuntimeUiEvent], None]


def handle_runtime_ui_event(app, event: RuntimeUiEvent) -> None:
    handler = RUNTIME_UI_EVENT_HANDLERS.get(type(event))
    if handler is None:
        raise TypeError(f"unsupported RuntimeUiEvent: {type(event).__name__}")
    handler(app, event)
    app._refresh()


def _handle_assistant_delta(app, event: RuntimeUiEvent) -> None:
    typed = _as_event(event, AssistantDeltaEvent)
    app._conversation.merge_assistant_delta(typed.turn_index, typed.model_turn, typed.delta)


def _handle_assistant_message(app, event: RuntimeUiEvent) -> None:
    typed = _as_event(event, AssistantMessageEvent)
    app._conversation.finalize_assistant_message(typed.turn_index, typed.model_turn, typed.content)
    app.clear_progress_status()


def _handle_assistant_intermediate(app, event: RuntimeUiEvent) -> None:
    typed = _as_event(event, AssistantIntermediateEvent)
    app._conversation.finalize_intermediate_message(
        typed.turn_index,
        typed.model_turn,
        typed.content,
    )


def _handle_tool_activity(app, event: ToolActivityEvent) -> None:
    presentation = present_tool_activity(event)
    if event.status == "failed":
        presentation = _aggregate_tool_failure(app, event, presentation)
    _apply_progress_presentation(app, presentation)


def _handle_approval_state(app, event: ApprovalStateEvent) -> None:
    if event.state == "requested":
        app.clear_progress_status()
        app._state = "waiting approval"
        _apply_progress_presentation(app, present_approval_state(event))
        return
    if event.state == "granted":
        app._state = "running"
        return
    _apply_progress_presentation(app, present_approval_state(event))


def _handle_user_input_state(app, event: UserInputStateEvent) -> None:
    if event.state == "requested":
        app.clear_progress_status()
        app._state = "waiting input"
        app._set_answer_required(event.question)
        _apply_progress_presentation(app, present_user_input_state(event))
        return
    app._state = "running"
    _apply_progress_presentation(app, present_user_input_state(event))


def _handle_memory_notice(app, event: MemoryNoticeEvent) -> None:
    message = event.message or "发现可记忆候选，已放入候选队列，等待你确认。"
    app._conversation.append_block("Memory", message)
    app.memory_flow.notice = message
    app.memory_flow.mode = True
    app.memory_flow.detail_mode = False
    app.memory_flow.load_candidates(silent=True)


def _handle_failure_notice(app, event: FailureNoticeEvent) -> None:
    app.clear_progress_status()
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
    app._conversation.append_block("Failure", app._last_failure.block_text())
    app._conversation.finalize_streaming_if_needed()


def _handle_task_progress(app, event: TaskProgressEvent) -> None:
    presentation = present_task_progress(event)
    if presentation.timeline_item is not None and event.event_name in {"task_recovery_suggested", "task_step_blocked"}:
        presentation = _aggregate_task_problem(app, event, presentation)
    _apply_progress_presentation(app, presentation)


def _handle_warning_notice(app, event: WarningNoticeEvent) -> None:
    if event.surface == "hidden":
        return
    if event.surface == "tool_detail":
        app._conversation.record_tool_diagnostic(
            event.turn_index,
            _warning_tool_name(event),
            _warning_detail_message(event),
        )
        return
    app._conversation.append_block(event.title, event.message)


def _handle_session_lifecycle(app, event: SessionLifecycleEvent) -> None:
    if event.state in {"turn_finished", "session_finished"}:
        app.clear_progress_status()
    sandbox = event.details.get("sandbox")
    if not isinstance(sandbox, dict):
        return
    backend = sandbox.get("backend")
    availability = sandbox.get("availability", {})
    if not isinstance(availability, dict):
        availability = {}
    if isinstance(backend, str) and backend:
        app._sandbox_status = {
            "backend": backend,
            "degraded": availability.get("degraded") is True,
            "reason": availability.get("reason") if isinstance(availability.get("reason"), str) else "",
        }
    return


def _as_event(event: RuntimeUiEvent, event_type):
    if not isinstance(event, event_type):
        raise TypeError(f"expected {event_type.__name__}, got {type(event).__name__}")
    return event


def _warning_tool_name(event: WarningNoticeEvent) -> str:
    value = event.details.get("tool_name")
    if isinstance(value, str) and value:
        return value
    subject = event.details.get("subject")
    if isinstance(subject, str) and subject:
        return subject
    return "unknown_tool"


def _warning_detail_message(event: WarningNoticeEvent) -> str:
    if event.notice_kind == "compression_diagnostic":
        return event.message
    return event.message


def _apply_progress_presentation(app, presentation: ProgressPresentation) -> None:
    if presentation.status_line is not None:
        app.set_progress_status(presentation.status_line)
    if presentation.timeline_item is None:
        return
    app.clear_progress_status()
    timeline = app.query_one("#conversation", ConversationTimeline)
    if timeline.replace_presentation_item(presentation.timeline_item, presentation.details):
        return
    timeline.add_presentation_item(presentation.timeline_item, presentation.details)


def _aggregate_tool_failure(
    app,
    event: ToolActivityEvent,
    presentation: ProgressPresentation,
) -> ProgressPresentation:
    if presentation.timeline_item is None:
        return presentation
    groups = getattr(app, "_tui_tool_failure_groups", None)
    if groups is None:
        groups = {}
        setattr(app, "_tui_tool_failure_groups", groups)
    key = (
        event.turn_index,
        event.tool_name,
        event.error_type or event.result_status or event.status,
    )
    count = groups.get(key, 0) + 1
    groups[key] = count
    if count <= 1:
        return presentation
    return present_grouped_tool_failure(event, count=count)


def _aggregate_task_problem(
    app,
    event: TaskProgressEvent,
    presentation: ProgressPresentation,
) -> ProgressPresentation:
    if presentation.timeline_item is None:
        return presentation
    groups = getattr(app, "_tui_task_problem_groups", None)
    if groups is None:
        groups = {}
        setattr(app, "_tui_task_problem_groups", groups)
    group = groups.setdefault(
        event.turn_index,
        {"count": 0, "labels": [], "actions": []},
    )
    group["count"] += 1
    label = _task_problem_label(presentation.timeline_item.title)
    if label and label not in group["labels"]:
        group["labels"].append(label)
    action = _task_problem_action(event.suggested_action)
    if action and action not in group["actions"]:
        group["actions"].append(action)
    return present_grouped_task_problem(
        event,
        count=group["count"],
        labels=group["labels"] or [label or "任务受阻"],
        actions=group["actions"],
    )


def _task_problem_label(title: str) -> str:
    prefix = "任务遇到问题："
    if title.startswith(prefix):
        return title[len(prefix) :].strip()
    return title.strip()


def _task_problem_action(value: str) -> str:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"none", "null", "n/a"}:
        return ""
    return stripped


RUNTIME_UI_EVENT_HANDLERS: dict[type[object], RuntimeUiEventHandler] = {
    AssistantDeltaEvent: _handle_assistant_delta,
    AssistantIntermediateEvent: _handle_assistant_intermediate,
    AssistantMessageEvent: _handle_assistant_message,
    ToolActivityEvent: _handle_tool_activity,
    ApprovalStateEvent: _handle_approval_state,
    UserInputStateEvent: _handle_user_input_state,
    MemoryNoticeEvent: _handle_memory_notice,
    WarningNoticeEvent: _handle_warning_notice,
    FailureNoticeEvent: _handle_failure_notice,
    TaskProgressEvent: _handle_task_progress,
    SessionLifecycleEvent: _handle_session_lifecycle,
}

if set(RUNTIME_UI_EVENT_HANDLERS) != set(RUNTIME_UI_EVENT_TYPES):
    missing = set(RUNTIME_UI_EVENT_TYPES) - set(RUNTIME_UI_EVENT_HANDLERS)
    extra = set(RUNTIME_UI_EVENT_HANDLERS) - set(RUNTIME_UI_EVENT_TYPES)
    raise RuntimeError(f"Runtime UI event handler registry mismatch: missing={missing}, extra={extra}")
