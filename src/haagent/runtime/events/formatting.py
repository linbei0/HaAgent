"""
src/haagent/runtime/events/formatting.py - Runtime UI 事件格式化

集中处理 runtime 原始字段到 UI 文本与轻量值的转换。
"""

from __future__ import annotations


def model_turn(event: dict[str, object]) -> int | None:
    turn = event.get("turn")
    return turn if isinstance(turn, int) else None


def tool_name(event: dict[str, object]) -> str:
    return str(event.get("tool_name", "unknown"))


def without_event_type(event: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in event.items() if key != "event_type"}


def int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def optional_summary(value: object, limit: int = 300) -> str:
    if value is None:
        return ""
    return summary_value(str(value), limit)


def summary_text(value: object) -> str:
    if isinstance(value, dict):
        parts = [f"{key}={summary_text(item)}" for key, item in value.items() if item is not None and item != ""]
        return ", ".join(parts) or "ok"
    if isinstance(value, list):
        return ", ".join(summary_text(item) for item in value[:5]) or "none"
    return summary_value(str(value), 120)
