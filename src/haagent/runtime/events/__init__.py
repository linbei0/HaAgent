"""
src/haagent/runtime/events/__init__.py - Runtime 事件协议包

集中放置 runtime 对外输出的事件协议与事件适配器。
"""

from haagent.runtime.events.factories import (
    failure_notice_event,
    memory_candidates_created_event,
    memory_extraction_warning_event,
    session_lifecycle_event,
)
from haagent.runtime.events.registry import RAW_RUNTIME_UI_EVENT_REGISTRY, RawRuntimeUiEventSpec
from haagent.runtime.events.types import (
    RUNTIME_UI_EVENT_TYPES,
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RuntimeUiEvent,
    RuntimeUiEventType,
    SessionLifecycleEvent,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.runtime.events.ui_mapper import RuntimeUiEventMapper

__all__ = [
    "RAW_RUNTIME_UI_EVENT_REGISTRY",
    "RUNTIME_UI_EVENT_TYPES",
    "ApprovalStateEvent",
    "AssistantDeltaEvent",
    "AssistantMessageEvent",
    "FailureNoticeEvent",
    "MemoryNoticeEvent",
    "RawRuntimeUiEventSpec",
    "RuntimeUiEvent",
    "RuntimeUiEventMapper",
    "RuntimeUiEventType",
    "SessionLifecycleEvent",
    "ToolActivityEvent",
    "UserInputStateEvent",
    "WarningNoticeEvent",
    "failure_notice_event",
    "memory_candidates_created_event",
    "memory_extraction_warning_event",
    "session_lifecycle_event",
]
