"""
src/haagent/runtime/events/registry.py - Runtime raw event 到 UI event 的注册表

维护 raw runtime event type 的单一事实源，并把映射实现藏在注册项后面。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from haagent.runtime.events.formatting import (
    model_turn,
    optional_summary,
    summary_text,
    summary_value,
    tool_name,
    without_event_type,
)
from haagent.runtime.events.types import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    RuntimeUiEvent,
    RuntimeUiEventType,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.tools.presentation import summarize_tool_args, summarize_tool_result


@dataclass(frozen=True)
class RawRuntimeUiEventContext:
    session_id: str
    turn_index: int
    model_turn: int | None


RawRuntimeUiEventBuilder = Callable[[dict[str, object], RawRuntimeUiEventContext], RuntimeUiEvent]


@dataclass(frozen=True)
class RawRuntimeUiEventSpec:
    event_type: str
    ui_event_type: RuntimeUiEventType
    build: RawRuntimeUiEventBuilder


def build_registered_runtime_ui_event(
    event: dict[str, object],
    *,
    session_id: str,
    turn_index: int,
) -> RuntimeUiEvent | None:
    event_type = str(event.get("event_type", "unknown"))
    spec = RAW_RUNTIME_UI_EVENT_REGISTRY.get(event_type)
    if spec is None:
        return None
    context = RawRuntimeUiEventContext(
        session_id=session_id,
        turn_index=turn_index,
        model_turn=model_turn(event),
    )
    return spec.build(event, context)


def unknown_runtime_ui_event(
    event: dict[str, object],
    *,
    session_id: str,
    turn_index: int,
) -> WarningNoticeEvent:
    event_type = str(event.get("event_type", "unknown"))
    return WarningNoticeEvent(
        session_id=session_id,
        turn_index=turn_index,
        title="Runtime warning",
        message=f"Unknown runtime event: {event_type}",
        notice_kind="runtime_warning",
        surface="timeline",
        details=without_event_type(event),
    )


def _assistant_delta_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> AssistantDeltaEvent:
    return AssistantDeltaEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        delta=str(event.get("delta", "")),
    )


def _assistant_message_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> AssistantMessageEvent:
    return AssistantMessageEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        content=str(event.get("content", "")),
    )


def _tool_started_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ToolActivityEvent:
    name = tool_name(event)
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    return ToolActivityEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=name,
        status="started",
        summary=f"starting tool {name}",
        args_summary=summarize_tool_args(name, args),
    )


def _tool_finished_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ToolActivityEvent:
    name = tool_name(event)
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    if name == "start_memory_update":
        summary = summary_value(str(result.get("reason", "")), 240)
    else:
        summary = summary_text(summarize_tool_result(name, result))
    return ToolActivityEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=name,
        status="finished",
        summary=summary,
        result_status=str(result.get("status", "")),
    )


def _tool_failed_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ToolActivityEvent:
    error = event.get("error") if isinstance(event.get("error"), dict) else {}
    message = summary_value(str(error.get("message", "")))
    return ToolActivityEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=tool_name(event),
        status="failed",
        summary=message,
        error_type=str(error.get("type", "unknown")),
        error_message=message,
    )


def _worker_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ToolActivityEvent:
    event_type = str(event.get("event_type", "worker_started"))
    agent_id = summary_value(str(event.get("agent_id", "unknown")), 80)
    status = summary_value(str(event.get("status", "")), 80)
    description = summary_value(str(event.get("description", "")), 160)
    if event_type == "worker_started":
        activity_status: Literal["started", "finished", "failed"] = "started"
        label = "started"
    elif event_type == "worker_failed":
        activity_status = "failed"
        label = "failed"
    else:
        activity_status = "finished"
        label = "stopped" if event_type == "worker_stopped" else "completed"
    summary = f"worker {label}: {description}" if description and description != "none" else f"worker {label}"
    return ToolActivityEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=f"agent:{agent_id}",
        status=activity_status,
        summary=summary,
        args_summary={
            "agent_id": agent_id,
            "task_id": summary_value(str(event.get("task_id", "")), 80),
            "team_id": summary_value(str(event.get("team_id", "")), 80),
            "subagent_type": summary_value(str(event.get("subagent_type", "")), 80),
        },
        result_status=status,
    )


def _tool_approval_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ApprovalStateEvent:
    return _approval_event(event, context, approval_kind="tool")


def _edit_diff_approval_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> ApprovalStateEvent:
    return _approval_event(event, context, approval_kind="edit_diff")


def _approval_event(
    event: dict[str, object],
    context: RawRuntimeUiEventContext,
    *,
    approval_kind: Literal["tool", "edit_diff"],
) -> ApprovalStateEvent:
    event_type = str(event.get("event_type", "approval_requested"))
    state: Literal["requested", "granted", "denied"] = "requested"
    if event_type.endswith("_granted"):
        state = "granted"
    elif event_type.endswith("_denied"):
        state = "denied"
    args_summary = event.get("args_summary") if isinstance(event.get("args_summary"), dict) else {}
    return ApprovalStateEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=tool_name(event),
        state=state,
        question=summary_value(str(event.get("question", "")), 240),
        approved=event.get("approved"),
        answer=optional_summary(event.get("answer"), 80),
        args_summary=args_summary,
        approval_kind=approval_kind,
    )


def _user_input_requested_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> UserInputStateEvent:
    return UserInputStateEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=tool_name(event),
        state="requested",
        question=summary_value(str(event.get("question", "")), 240),
        reason=summary_value(str(event.get("reason", "")), 240),
    )


def _user_input_received_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> UserInputStateEvent:
    return UserInputStateEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        model_turn=context.model_turn,
        tool_name=tool_name(event),
        state="received",
        question=summary_value(str(event.get("question", "")), 240),
        answer_chars=event.get("answer_chars"),
        approved=event.get("approved"),
    )


def _guardrail_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> WarningNoticeEvent:
    return WarningNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        title="Guardrail",
        message=summary_value(str(event.get("message", "guardrail triggered"))),
        notice_kind="guardrail",
        surface="timeline",
        details=without_event_type(event),
    )


def _compression_diagnostic_event(
    event: dict[str, object],
    context: RawRuntimeUiEventContext,
) -> WarningNoticeEvent:
    subject = _compression_subject(event)
    return WarningNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        title="压缩诊断",
        message=_compression_message(event, subject),
        notice_kind="compression_diagnostic",
        surface="tool_detail",
        details=without_event_type(event),
    )


def _compression_message(event: dict[str, object], subject: str) -> str:
    label = _compression_stage_label(str(event.get("stage", "")))
    original_chars = event.get("original_chars")
    final_chars = event.get("final_chars")
    if isinstance(original_chars, int) and isinstance(final_chars, int):
        return f"{label}：{subject} {original_chars} chars -> {final_chars} chars"
    original_tokens = event.get("original_tokens")
    final_tokens = event.get("final_tokens")
    if isinstance(original_tokens, int) and isinstance(final_tokens, int):
        return f"{label}：{subject} {original_tokens} tokens -> {final_tokens} tokens"
    return f"{label}：{subject}"


def _compression_stage_label(stage: str) -> str:
    return {
        "tool_output_artifact": "工具输出落盘",
        "historical_tool_message": "旧工具消息降级",
        "context_section": "上下文 section 折叠",
        "session_memory": "会话记忆压缩",
        "full_compact": "自动 full compact",
    }.get(stage, "压缩诊断")


def _compression_subject(event: dict[str, object]) -> str:
    subject = event.get("subject")
    if isinstance(subject, str) and subject:
        return subject
    name = event.get("tool_name")
    if isinstance(name, str) and name:
        return name
    return "unknown"


def _loop_suggestion_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> WarningNoticeEvent:
    return WarningNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        title="Loop guidance",
        message=summary_value(str(event.get("message", ""))),
        notice_kind="loop_guidance",
        surface="hidden",
        details=without_event_type(event),
    )


def _safety_abort_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> WarningNoticeEvent:
    return WarningNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        title="Safety abort",
        message=summary_value(str(event.get("message", ""))),
        notice_kind="safety_abort",
        surface="timeline",
        details=without_event_type(event),
    )


def _interaction_reused_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> WarningNoticeEvent:
    interaction_type = str(event.get("interaction_type", "interaction"))
    name = tool_name(event)
    resolved_turn = event.get("resolved_turn", "")
    return WarningNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        title="Interaction reused",
        message=f"{interaction_type} for {name} reused from turn {resolved_turn}",
        details=without_event_type(event),
    )


def _failure_event(event: dict[str, object], context: RawRuntimeUiEventContext) -> FailureNoticeEvent:
    return FailureNoticeEvent(
        session_id=context.session_id,
        turn_index=context.turn_index,
        status=str(event.get("status", "failed")),
        failed_stage=summary_value(str(event.get("failed_stage", "unknown"))),
        failure_category=summary_value(str(event.get("failure_category", "unknown"))),
        reason=summary_value(str(event.get("reason", ""))),
        episode_path=summary_value(str(event.get("episode_path", "")), 300),
    )


def _spec(
    event_type: str,
    ui_event_type: RuntimeUiEventType,
    build: RawRuntimeUiEventBuilder,
) -> RawRuntimeUiEventSpec:
    return RawRuntimeUiEventSpec(event_type=event_type, ui_event_type=ui_event_type, build=build)


_RAW_RUNTIME_UI_EVENT_SPECS: tuple[RawRuntimeUiEventSpec, ...] = (
    _spec("assistant_delta", AssistantDeltaEvent, _assistant_delta_event),
    _spec("assistant_message", AssistantMessageEvent, _assistant_message_event),
    _spec("tool_started", ToolActivityEvent, _tool_started_event),
    _spec("tool_finished", ToolActivityEvent, _tool_finished_event),
    _spec("tool_failed", ToolActivityEvent, _tool_failed_event),
    _spec("approval_requested", ApprovalStateEvent, _tool_approval_event),
    _spec("approval_granted", ApprovalStateEvent, _tool_approval_event),
    _spec("approval_denied", ApprovalStateEvent, _tool_approval_event),
    _spec("edit_diff_requested", ApprovalStateEvent, _edit_diff_approval_event),
    _spec("edit_diff_granted", ApprovalStateEvent, _edit_diff_approval_event),
    _spec("edit_diff_denied", ApprovalStateEvent, _edit_diff_approval_event),
    _spec("user_input_requested", UserInputStateEvent, _user_input_requested_event),
    _spec("user_input_received", UserInputStateEvent, _user_input_received_event),
    _spec("guardrail_triggered", WarningNoticeEvent, _guardrail_event),
    _spec("compression_diagnostic", WarningNoticeEvent, _compression_diagnostic_event),
    _spec("loop_suggestion_added", WarningNoticeEvent, _loop_suggestion_event),
    _spec("safety_abort", WarningNoticeEvent, _safety_abort_event),
    _spec("interaction_reused", WarningNoticeEvent, _interaction_reused_event),
    _spec("failure", FailureNoticeEvent, _failure_event),
    _spec("worker_started", ToolActivityEvent, _worker_event),
    _spec("worker_completed", ToolActivityEvent, _worker_event),
    _spec("worker_failed", ToolActivityEvent, _worker_event),
    _spec("worker_stopped", ToolActivityEvent, _worker_event),
)

RAW_RUNTIME_UI_EVENT_REGISTRY: dict[str, RawRuntimeUiEventSpec] = {
    spec.event_type: spec for spec in _RAW_RUNTIME_UI_EVENT_SPECS
}
