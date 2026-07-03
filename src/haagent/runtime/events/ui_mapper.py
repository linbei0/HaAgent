"""
src/haagent/runtime/events/ui_mapper.py - Runtime UI 事件映射入口

提供 runtime raw dict 到 RuntimeUiEvent 的窄 Interface。
"""

from __future__ import annotations

from haagent.runtime.events.registry import build_registered_runtime_ui_event, unknown_runtime_ui_event
from haagent.runtime.events.types import RuntimeUiEvent


class RuntimeUiEventMapper:
    """把 runtime 原始 dict 事件映射为 TUI 行为事件。"""

    @staticmethod
    def to_ui_event(event: dict[str, object], *, session_id: str, turn_index: int) -> RuntimeUiEvent:
        registered = build_registered_runtime_ui_event(event, session_id=session_id, turn_index=turn_index)
        if registered is not None:
            return registered
        return unknown_runtime_ui_event(event, session_id=session_id, turn_index=turn_index)
