"""
src/haagent/runtime/events/types.py - Runtime UI 事件类型

定义 TUI 可直接消费的强类型事件，不暴露原始 runtime dict 结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

NoticeSurface: TypeAlias = Literal["timeline", "tool_detail", "hidden"]


@dataclass(frozen=True)
class AssistantDeltaEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    delta: str


@dataclass(frozen=True)
class AssistantMessageEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    content: str


@dataclass(frozen=True)
class AssistantIntermediateEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    content: str


@dataclass(frozen=True)
class ContextUsageEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    input_tokens: int
    input_window_tokens: int | None = None


@dataclass(frozen=True)
class ToolActivityEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    tool_name: str
    status: Literal["started", "finished", "failed"]
    summary: str
    args_summary: dict[str, object] = field(default_factory=dict)
    result_status: str = ""
    error_type: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class ApprovalStateEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    tool_name: str
    state: Literal["requested", "granted", "denied"]
    question: str
    approved: object
    answer: str = ""
    args_summary: dict[str, object] = field(default_factory=dict)
    approval_kind: Literal["tool", "edit_diff"] = "tool"


@dataclass(frozen=True)
class UserInputStateEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    tool_name: str
    state: Literal["requested", "received"]
    question: str
    reason: str = ""
    answer_chars: object = None
    approved: object = None


@dataclass(frozen=True)
class MemoryNoticeEvent:
    session_id: str
    turn_index: int
    message: str
    count: int = 0
    status: str = ""
    reason: str = ""


@dataclass(frozen=True)
class WarningNoticeEvent:
    session_id: str
    turn_index: int
    title: str
    message: str
    notice_kind: str = "warning"
    surface: NoticeSurface = "timeline"
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureNoticeEvent:
    session_id: str
    turn_index: int
    status: str
    failed_stage: str
    failure_category: str
    reason: str
    episode_path: str


@dataclass(frozen=True)
class TaskProgressEvent:
    session_id: str
    turn_index: int
    model_turn: int | None
    event_name: str
    step_id: str
    title: str
    status: str
    summary: str
    owner: str = "main"
    category: str = ""
    suggested_action: str = ""
    evidence_count: int = 0
    checkpoint_count: int = 0
    reason_chars: int = 0


@dataclass(frozen=True)
class SessionLifecycleEvent:
    session_id: str
    turn_index: int
    state: Literal["session_started", "session_finished", "turn_started", "turn_finished"]
    message: str
    status: str = ""
    prompt: str = ""
    episode_path: str = ""
    runtime_event_count: int = 0
    details: dict[str, object] = field(default_factory=dict)


RuntimeUiEvent: TypeAlias = (
    AssistantDeltaEvent
    | AssistantIntermediateEvent
    | AssistantMessageEvent
    | ContextUsageEvent
    | ToolActivityEvent
    | ApprovalStateEvent
    | UserInputStateEvent
    | MemoryNoticeEvent
    | WarningNoticeEvent
    | FailureNoticeEvent
    | TaskProgressEvent
    | SessionLifecycleEvent
)

RuntimeUiEventType: TypeAlias = (
    type[AssistantDeltaEvent]
    | type[AssistantIntermediateEvent]
    | type[AssistantMessageEvent]
    | type[ContextUsageEvent]
    | type[ToolActivityEvent]
    | type[ApprovalStateEvent]
    | type[UserInputStateEvent]
    | type[MemoryNoticeEvent]
    | type[WarningNoticeEvent]
    | type[FailureNoticeEvent]
    | type[TaskProgressEvent]
    | type[SessionLifecycleEvent]
)

RUNTIME_UI_EVENT_TYPES: tuple[RuntimeUiEventType, ...] = (
    AssistantDeltaEvent,
    AssistantIntermediateEvent,
    AssistantMessageEvent,
    ContextUsageEvent,
    ToolActivityEvent,
    ApprovalStateEvent,
    UserInputStateEvent,
    MemoryNoticeEvent,
    WarningNoticeEvent,
    FailureNoticeEvent,
    TaskProgressEvent,
    SessionLifecycleEvent,
)
