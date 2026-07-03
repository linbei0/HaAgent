"""
src/haagent/runtime/events/ui.py - Runtime UI 事件公开入口

向现有调用方重新导出事件类型、注册表、mapper 与工厂，隐藏包内拆分结构。
"""

from __future__ import annotations

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
