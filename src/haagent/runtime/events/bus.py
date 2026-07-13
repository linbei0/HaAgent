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
class AssistantIntermediateBusEvent:
    """带工具调用的模型轮次中，面向用户的普通 assistant 文本。"""

    turn: int
    content: str
    event_type: str = field(default="assistant_intermediate_message", init=False)


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
    execution_state: str = ""
    event_type: str = field(default="tool_failed", init=False)


@dataclass(frozen=True)
class ModelRetryScheduledBusEvent:
    """模型请求已安排下一次安全重放的最小审计事实。"""

    turn: int
    attempt: int
    next_attempt: int
    category: str
    delay_seconds: float
    source: str
    retry_after_ignored: bool = False
    event_type: str = field(default="model_retry_scheduled", init=False)


@dataclass(frozen=True)
class ModelRetryExhaustedBusEvent:
    """模型可重试失败已耗尽本轮重试预算的脱敏事实。"""

    turn: int
    attempt: int
    category: str
    status_code: int | None = None
    request_id: str | None = None
    event_type: str = field(default="model_retry_exhausted", init=False)


@dataclass(frozen=True)
class ModelRouteFallbackBusEvent:
    turn: int
    kind: str
    reason: str
    from_connection: str | None
    from_model: str | None
    from_protocol: str | None
    to_connection: str | None
    to_model: str | None
    to_protocol: str | None
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()

    @property
    def event_type(self) -> str:
        return self.kind


@dataclass(frozen=True)
class LegacyRawBusEvent:
    """未切片的 raw dict 事件包装；to_dict 原样返回。"""

    payload: dict[str, object]

    @property
    def event_type(self) -> str:
        return str(self.payload.get("event_type", ""))


RuntimeBusEvent: TypeAlias = (
    AssistantDeltaBusEvent
    | AssistantIntermediateBusEvent
    | AssistantMessageBusEvent
    | ToolStartedBusEvent
    | ToolFinishedBusEvent
    | ToolFailedBusEvent
    | ModelRetryScheduledBusEvent
    | ModelRetryExhaustedBusEvent
    | ModelRouteFallbackBusEvent
    | LegacyRawBusEvent
)

def bus_event_to_dict(event: RuntimeBusEvent | dict[str, object]) -> dict[str, object]:
    """边界序列化：episode / UI mapper / 兼容消费者使用 dict 形态。"""
    if isinstance(event, dict):
        return dict(event)
    if isinstance(event, LegacyRawBusEvent):
        return dict(event.payload)
    if isinstance(event, AssistantDeltaBusEvent):
        return {"event_type": "assistant_delta", "turn": event.turn, "delta": event.delta}
    if isinstance(event, AssistantIntermediateBusEvent):
        return {"event_type": "assistant_intermediate_message", "turn": event.turn, "content": event.content}
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
        payload = {
            "event_type": "tool_failed",
            "turn": event.turn,
            "tool_name": event.tool_name,
            "args": dict(event.args),
            "error": dict(event.error),
        }
        if event.execution_state:
            payload["execution_state"] = event.execution_state
        return payload
    if isinstance(event, ModelRetryScheduledBusEvent):
        return {
            "event_type": "model_retry_scheduled",
            "turn": event.turn,
            "attempt": event.attempt,
            "next_attempt": event.next_attempt,
            "category": event.category,
            "delay_seconds": event.delay_seconds,
            "source": event.source,
            "retry_after_ignored": event.retry_after_ignored,
        }
    if isinstance(event, ModelRetryExhaustedBusEvent):
        return {
            "event_type": "model_retry_exhausted",
            "turn": event.turn,
            "attempt": event.attempt,
            "category": event.category,
            "status_code": event.status_code,
            "request_id": event.request_id,
        }
    if isinstance(event, ModelRouteFallbackBusEvent):
        return {
            "event_type": event.kind,
            "turn": event.turn,
            "reason": event.reason,
            "from_connection": event.from_connection,
            "from_model": event.from_model,
            "from_protocol": event.from_protocol,
            "to_connection": event.to_connection,
            "to_model": event.to_model,
            "to_protocol": event.to_protocol,
            "required_capabilities": list(event.required_capabilities),
            "missing_capabilities": list(event.missing_capabilities),
        }
    raise TypeError(f"unsupported bus event: {type(event)!r}")


def bus_event_from_dict(payload: dict[str, object]) -> RuntimeBusEvent:
    event_type = str(payload.get("event_type", ""))
    if event_type == "assistant_delta":
        return AssistantDeltaBusEvent(turn=int(payload.get("turn", 0) or 0), delta=str(payload.get("delta", "")))
    if event_type == "assistant_intermediate_message":
        return AssistantIntermediateBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            content=str(payload.get("content", "")),
        )
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
            execution_state=str(payload.get("execution_state", "")),
        )
    if event_type == "model_retry_scheduled":
        return ModelRetryScheduledBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            attempt=int(payload.get("attempt", 0) or 0),
            next_attempt=int(payload.get("next_attempt", 0) or 0),
            category=str(payload.get("category", "")),
            delay_seconds=float(payload.get("delay_seconds", 0) or 0),
            source=str(payload.get("source", "")),
            retry_after_ignored=payload.get("retry_after_ignored") is True,
        )
    if event_type == "model_retry_exhausted":
        status_code = payload.get("status_code")
        return ModelRetryExhaustedBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            attempt=int(payload.get("attempt", 0) or 0),
            category=str(payload.get("category", "")),
            status_code=status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None,
            request_id=str(payload.get("request_id")) if payload.get("request_id") is not None else None,
        )
    if event_type in {"model_protocol_fallback", "model_fallback"}:
        required = payload.get("required_capabilities")
        missing = payload.get("missing_capabilities")
        return ModelRouteFallbackBusEvent(
            turn=int(payload.get("turn", 0) or 0),
            kind=event_type,
            reason=str(payload.get("reason", "")),
            from_connection=_optional_string(payload.get("from_connection")),
            from_model=_optional_string(payload.get("from_model")),
            from_protocol=_optional_string(payload.get("from_protocol")),
            to_connection=_optional_string(payload.get("to_connection")),
            to_model=_optional_string(payload.get("to_model")),
            to_protocol=_optional_string(payload.get("to_protocol")),
            required_capabilities=tuple(str(item) for item in required) if isinstance(required, list) else (),
            missing_capabilities=tuple(str(item) for item in missing) if isinstance(missing, list) else (),
        )
    return LegacyRawBusEvent(payload=dict(payload))


def coerce_bus_event(event: RuntimeBusEvent | dict[str, object]) -> RuntimeBusEvent:
    """把 sink 入参统一为 RuntimeBusEvent（dict 走 from_dict）。"""
    if isinstance(event, dict):
        return bus_event_from_dict(event)
    return event


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
