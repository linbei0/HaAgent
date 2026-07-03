"""
src/haagent/runtime/session/working_state.py - 短期工作状态

维护 chat 会话内的有界 working_state，不保存完整工具输出或 episode trace。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKING_STATE_MODEL_CHAR_LIMIT = 1200
WORKING_STATE_TEXT_FIELD_LIMIT = 240
WORKING_STATE_CURRENT_GOAL_LIMIT = 200
WORKING_STATE_MAX_ITEMS = 5
TRACE_MARKERS = (
    "tool-calls.jsonl",
    "transcript.jsonl",
    '"event":',
    '"tool_name"',
)


class WorkingStateError(RuntimeError):
    """working_state 文件损坏或结构不合法时抛出。"""


@dataclass(frozen=True)
class WorkingState:
    current_goal: str
    key_findings: list[str]
    completed_actions: list[str]
    next_steps: list[str]
    last_updated_turn: int

    def to_dict(self) -> dict[str, object]:
        return {
            "current_goal": self.current_goal,
            "key_findings": list(self.key_findings),
            "completed_actions": list(self.completed_actions),
            "next_steps": list(self.next_steps),
            "last_updated_turn": self.last_updated_turn,
        }

    def is_empty(self) -> bool:
        return (
            not self.current_goal
            and not self.key_findings
            and not self.completed_actions
            and not self.next_steps
            and self.last_updated_turn == 0
        )

    def status_summary(self) -> dict[str, object]:
        return {
            "exists": not self.is_empty(),
            "current_goal": self.current_goal,
            "key_findings_count": len(self.key_findings),
            "completed_actions_count": len(self.completed_actions),
            "next_steps_count": len(self.next_steps),
            "last_updated_turn": self.last_updated_turn,
        }


def empty_working_state() -> WorkingState:
    return WorkingState(
        current_goal="",
        key_findings=[],
        completed_actions=[],
        next_steps=[],
        last_updated_turn=0,
    )


def load_working_state(path: Path) -> WorkingState:
    if not path.exists():
        raise WorkingStateError(f"session package missing required file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkingStateError(f"invalid working_state.json: {path}") from error
    return working_state_from_dict(raw)


def write_working_state(path: Path, state: WorkingState) -> None:
    path.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def working_state_from_dict(raw: object) -> WorkingState:
    if not isinstance(raw, dict):
        raise WorkingStateError("invalid working_state.json: must contain an object")
    required = ["current_goal", "key_findings", "completed_actions", "next_steps", "last_updated_turn"]
    for field_name in required:
        if field_name not in raw:
            raise WorkingStateError(f"invalid working_state.json: missing {field_name}")
    if not isinstance(raw["current_goal"], str):
        raise WorkingStateError("invalid working_state.json: current_goal must be a string")
    for field_name in ["key_findings", "completed_actions", "next_steps"]:
        if not isinstance(raw[field_name], list) or not all(isinstance(item, str) for item in raw[field_name]):
            raise WorkingStateError(f"invalid working_state.json: {field_name} must be a list of strings")
    if not isinstance(raw["last_updated_turn"], int) or isinstance(raw["last_updated_turn"], bool):
        raise WorkingStateError("invalid working_state.json: last_updated_turn must be an integer")
    return WorkingState(
        current_goal=_bounded_text(str(raw["current_goal"]), WORKING_STATE_CURRENT_GOAL_LIMIT),
        key_findings=_bounded_items(raw["key_findings"]),
        completed_actions=_bounded_items(raw["completed_actions"]),
        next_steps=_bounded_items(raw["next_steps"]),
        last_updated_turn=int(raw["last_updated_turn"]),
    )


def update_working_state(
    state: WorkingState,
    *,
    prompt: str,
    result: Any,
    runtime_events: list[dict[str, object]],
) -> WorkingState:
    key_findings = list(state.key_findings)
    completed_actions = list(state.completed_actions)

    for event in runtime_events:
        event_type = str(event.get("event_type", ""))
        if event_type == "tool_finished":
            completed_actions.append(_tool_action_summary(event))
        elif event_type == "tool_failed":
            completed_actions.append(_tool_failure_summary(event))
        elif event_type == "assistant_message":
            finding = _bounded_text(str(event.get("content", "")))
            if finding and finding != "none":
                key_findings.append(finding)

    final_response = _bounded_text(str(getattr(result, "final_response", "")))
    if final_response and final_response != "none":
        key_findings.append(final_response)
    if not completed_actions:
        completed_actions.append(
            f"turn {getattr(result, 'turn_index', 0)} status={getattr(result, 'status', 'unknown')}",
        )

    return WorkingState(
        current_goal=_bounded_text(prompt, WORKING_STATE_CURRENT_GOAL_LIMIT),
        key_findings=_bounded_items(key_findings),
        completed_actions=_bounded_items(completed_actions),
        next_steps=_next_steps_from_result(result),
        last_updated_turn=int(getattr(result, "turn_index", 0)),
    )


def format_working_state_for_model(value: object) -> str:
    state = _state_from_value(value)
    if state.is_empty():
        return ""
    lines = [
        "Working State:",
        f"last_user_request: {state.current_goal or 'none'}",
        "assistant_findings:",
        *_format_list(state.key_findings),
        "assistant_actions:",
        *_format_list(state.completed_actions),
        "next_steps:",
        *_format_list(state.next_steps),
        f"last_updated_turn: {state.last_updated_turn}",
    ]
    text = "\n".join(lines)
    return text[:WORKING_STATE_MODEL_CHAR_LIMIT]


def raw_working_state_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _state_from_value(value: object) -> WorkingState:
    if isinstance(value, WorkingState):
        return value
    return working_state_from_dict(value)


def _tool_action_summary(event: dict[str, object]) -> str:
    tool_name = str(event.get("tool_name", "unknown"))
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    parts = [f"actor=assistant", f"tool={tool_name}", f"status={result.get('status', 'unknown')}"]
    if tool_name in {"shell", "code_run"} and result.get("exit_code") is not None:
        parts.append(f"exit_code={result.get('exit_code')}")
    if tool_name in {"file_read", "file_write", "apply_patch"} and result.get("path"):
        parts.append(f"path={_bounded_text(str(result.get('path')), 120)}")
    if tool_name == "file_write" and result.get("bytes_written") is not None:
        parts.append(f"bytes_written={result.get('bytes_written')}")
    return _bounded_text(" ".join(parts))


def _tool_failure_summary(event: dict[str, object]) -> str:
    tool_name = str(event.get("tool_name", "unknown"))
    error = event.get("error") if isinstance(event.get("error"), dict) else {}
    error_type = str(error.get("type", "unknown"))
    return _bounded_text(f"actor=assistant tool={tool_name} status=failed type={error_type}")


def _next_steps_from_result(result: Any) -> list[str]:
    status = str(getattr(result, "status", "unknown"))
    verification_status = str(getattr(result, "verification_status", "not_run"))
    if status == "completed":
        if verification_status == "success":
            return ["Use the verified result for follow-up work if the user continues."]
        return ["Continue from this completed chat turn if the user asks a follow-up."]
    reason = _bounded_text(str(getattr(result, "reason", "")), 180)
    failure_category = _bounded_text(str(getattr(result, "failure_category", "unknown")), 120)
    return [f"Address failure {failure_category}: {reason}"]


def _bounded_items(items: list[str]) -> list[str]:
    selected: list[str] = []
    for item in items:
        text = _bounded_text(item)
        if text and text != "none" and not _looks_like_trace(text):
            selected.append(text)
    return selected[-WORKING_STATE_MAX_ITEMS:]


def _bounded_text(value: str, limit: int = WORKING_STATE_TEXT_FIELD_LIMIT) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        return ""
    if _looks_like_trace(normalized):
        return ""
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _looks_like_trace(value: str) -> bool:
    return any(marker in value for marker in TRACE_MARKERS)


def _format_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]
