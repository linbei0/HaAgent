"""
src/haagent/runtime/events/bus.py - Runtime 内部证据总线事件

定义 RuntimeBusEvent：保留完整工具 args/result 的强类型证据事件，
以及 LegacyRawBusEvent 过渡包装。Episode/UI 仍经 to_dict 边界投影。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(frozen=True)
class AssistantDeltaBusEvent:
    turn: int
    delta: str
    event_type: str = field(default="assistant_delta", init=False)


@dataclass(frozen=True)
class AssistantMessageBusEvent:
    turn: int
    content: str
    event_type: str = field(default="assistant_message", init=False)


@dataclass(frozen=True)
class ToolStartedBusEvent:
    turn: int
    tool_name: str
    args: dict[str, Any]
    event_type: str = field(default="tool_started", init=False)


@dataclass(frozen=True)
class ToolFinishedBusEvent:
    turn: int
    tool_name: str
    args: dict[str, Any]
    result: dict[str, Any]
    event_type: str = field(default="tool_finished", init=False)


@dataclass(frozen=True)
class ToolFailedBusEvent:
    turn: int
    tool_name: str
    args: dict[str, Any]
    error: dict[str, Any]
    event_type: str = field(default="tool_failed", init=False)


@dataclass(frozen=True)
class LegacyRawBusEvent:
    """未切片的 raw dict 事件包装；to_dict 原样返回。"""

    payload: dict[str, object]

    @property
    def event_type(self) -> str:
        return str(self.payload.get("event_type", ""))


RuntimeBusEvent: TypeAlias = (
    AssistantDeltaBusEvent
    | AssistantMessageBusEvent
    | ToolStartedBusEvent
    | ToolFinishedBusEvent
    | ToolFailedBusEvent
    | LegacyRawBusEvent
)

_SLICE_EVENT_TYPES = frozenset(
    {
        "assistant_delta",
        "assistant_message",
        "tool_started",
        "tool_finished",
        "tool_failed",
    }
)


def bus_event_to_dict(event: RuntimeBusEvent | dict[str, object]) -> dict[str, object]:
    """边界序列化：episode / UI mapper / 兼容消费者使用 dict 形态。"""
    if isinstance(event, dict):
        return dict(event)
    if isinstance(event, LegacyRawBusEvent):
        return dict(event.payload)
    if isinstance(event, AssistantDeltaBusEvent):
        return {"event_type": "assistant_delta", "turn": event.turn, "delta": event.delta}
    if isinstance(event, AssistantMessageBusEvent):
        return {"event_type": "assistant_message", "turn": event.turn, "content": event.content}
    if isinstance(event, ToolStartedBusEvent):
        return {
            "event_type": "tool_started",
            "turn": event.turn,
            "tool_name": event.tool_name,
            "args": dict(event.args),
        }
    if isinstance(event, ToolFinishedBusEvent):
        return {
            "event_type": "tool_finished",
            "turn": event.turn,
            "tool_name": event.tool_name,
            "args": dict(event.args),
            "result": dict(event.result),
        }
    if isinstance(event, ToolFailedBusEvent):
        return {
            "event_type": "tool_failed",
            "turn": event.turn,
            "tool_name": event.tool_name,
            "args": dict(event.args),
            "error": dict(event.error),
        }
    raise TypeError(f"unsupported bus event: {type(event)!r}")


def bus_event_from_dict(payload: dict[str, object]) -> RuntimeBusEvent:
    event_type = str(payload.get("event_type", ""))
    if event_type == "assistant_delta":
        return AssistantDeltaBusEvent(turn=int(payload.get("turn", 0) or 0), delta=str(payload.get("delta", "")))
    if event_type == "assistant_message":
        return AssistantMessageBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            content=str(payload.get("content", "")),
        )
    if event_type == "tool_started":
        args = payload.get("args")
        return ToolStartedBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            tool_name=str(payload.get("tool_name", "")),
            args=dict(args) if isinstance(args, dict) else {},
        )
    if event_type == "tool_finished":
        args = payload.get("args")
        result = payload.get("result")
        return ToolFinishedBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            tool_name=str(payload.get("tool_name", "")),
            args=dict(args) if isinstance(args, dict) else {},
            result=dict(result) if isinstance(result, dict) else {},
        )
    if event_type == "tool_failed":
        args = payload.get("args")
        error = payload.get("error")
        return ToolFailedBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            tool_name=str(payload.get("tool_name", "")),
            args=dict(args) if isinstance(args, dict) else {},
            error=dict(error) if isinstance(error, dict) else {},
        )
    return LegacyRawBusEvent(payload=dict(payload))


def coerce_bus_event(event: RuntimeBusEvent | dict[str, object]) -> RuntimeBusEvent:
    """把 sink 入参统一为 RuntimeBusEvent（dict 走 from_dict）。"""
    if isinstance(event, dict):
        return bus_event_from_dict(event)
    return event


def is_slice_bus_event(event: RuntimeBusEvent | dict[str, object]) -> bool:
    if isinstance(event, dict):
        return str(event.get("event_type", "")) in _SLICE_EVENT_TYPES
    if isinstance(event, LegacyRawBusEvent):
        return False
    return event.event_type in _SLICE_EVENT_TYPES
