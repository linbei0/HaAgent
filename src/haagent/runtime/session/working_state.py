"""
src/haagent/runtime/session/working_state.py - 有界关键发现状态

只保存 TaskLedger 与 session summary 无法表达的短期关键发现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKING_STATE_MODEL_CHAR_LIMIT = 1200
WORKING_STATE_TEXT_FIELD_LIMIT = 240
WORKING_STATE_MAX_ITEMS = 5
TRACE_MARKERS = ("tool-calls.jsonl", "transcript.jsonl", '"event":', '"tool_name"')


class WorkingStateError(RuntimeError):
    """working_state 文件损坏或结构不合法时抛出。"""


@dataclass(frozen=True)
class WorkingState:
    key_findings: list[str]
    last_updated_turn: int

    def to_dict(self) -> dict[str, object]:
        return {
            "key_findings": list(self.key_findings),
            "last_updated_turn": self.last_updated_turn,
        }

    def is_empty(self) -> bool:
        return not self.key_findings and self.last_updated_turn == 0

    def status_summary(self) -> dict[str, object]:
        return {
            "exists": not self.is_empty(),
            "key_findings_count": len(self.key_findings),
            "last_updated_turn": self.last_updated_turn,
        }


def empty_working_state() -> WorkingState:
    return WorkingState(key_findings=[], last_updated_turn=0)


def load_working_state(path: Path) -> WorkingState:
    if not path.exists():
        raise WorkingStateError(f"session package missing required file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkingStateError(f"invalid working_state.json: {path}") from error
    return working_state_from_dict(raw)


def write_working_state(path: Path, state: WorkingState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def working_state_from_dict(raw: object) -> WorkingState:
    if not isinstance(raw, dict):
        raise WorkingStateError("invalid working_state.json: must contain an object")
    if set(raw) != {"key_findings", "last_updated_turn"}:
        raise WorkingStateError("invalid working_state.json: fields do not match current schema")
    findings = raw.get("key_findings")
    if not isinstance(findings, list) or not all(isinstance(item, str) for item in findings):
        raise WorkingStateError("invalid working_state.json: key_findings must be a list of strings")
    updated_turn = raw.get("last_updated_turn")
    if not isinstance(updated_turn, int) or isinstance(updated_turn, bool) or updated_turn < 0:
        raise WorkingStateError("invalid working_state.json: last_updated_turn must be a non-negative integer")
    return WorkingState(key_findings=_bounded_items(findings), last_updated_turn=updated_turn)


def update_working_state(
    state: WorkingState,
    *,
    prompt: str,
    result: Any,
    runtime_events: list[Any],
) -> WorkingState:
    del prompt
    from haagent.runtime.events.bus import AssistantMessageBusEvent, bus_event_to_dict, coerce_bus_event

    findings = list(state.key_findings)
    for raw_event in runtime_events:
        event = coerce_bus_event(raw_event)
        if isinstance(event, AssistantMessageBusEvent):
            candidate = _bounded_text(event.content)
        else:
            payload = bus_event_to_dict(event)
            candidate = _bounded_text(str(payload.get("content", ""))) if payload.get("event_type") == "assistant_message" else ""
        if candidate and candidate != "none":
            findings.append(candidate)
    final_response = _bounded_text(str(getattr(result, "final_response", "")))
    if final_response and not any(item.startswith(final_response[:120]) for item in findings):
        findings.append(final_response)
    return WorkingState(
        key_findings=_bounded_items(findings),
        last_updated_turn=max(0, int(getattr(result, "turn_index", 0))),
    )


def format_working_state_for_model(value: object) -> str:
    state = value if isinstance(value, WorkingState) else working_state_from_dict(value)
    if state.is_empty():
        return ""
    lines = ["key_findings:", *[f"- {item}" for item in state.key_findings], f"last_updated_turn: {state.last_updated_turn}"]
    return "\n".join(lines)[:WORKING_STATE_MODEL_CHAR_LIMIT]


def raw_working_state_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _bounded_items(items: list[str]) -> list[str]:
    selected = [_bounded_text(item) for item in items]
    return [item for item in selected if item and item != "none"][-WORKING_STATE_MAX_ITEMS:]


def _bounded_text(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized or any(marker in normalized for marker in TRACE_MARKERS):
        return ""
    if len(normalized) <= WORKING_STATE_TEXT_FIELD_LIMIT:
        return normalized
    return normalized[: WORKING_STATE_TEXT_FIELD_LIMIT - 1] + "…"
