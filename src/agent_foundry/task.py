from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TaskLoadError(ValueError):
    """Raised when task.yaml cannot be loaded as a valid task spec."""


@dataclass(frozen=True)
class TaskSpec:
    goal: str
    constraints: list[str]
    allowed_tools: list[str]
    acceptance_criteria: list[str]
    verification_commands: list[str]


def load_task(path: Path) -> TaskSpec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TaskLoadError("task.yaml must contain a mapping")

    goal = _required_str(raw, "goal")
    return TaskSpec(
        goal=goal,
        constraints=_required_str_list(raw, "constraints"),
        allowed_tools=_required_str_list(raw, "allowed_tools"),
        acceptance_criteria=_required_str_list(raw, "acceptance_criteria"),
        verification_commands=_required_str_list(raw, "verification_commands"),
    )


def _required_str(raw: dict[str, Any], field: str) -> str:
    if field not in raw:
        raise TaskLoadError(f"missing required field: {field}")
    value = raw[field]
    if not isinstance(value, str):
        raise TaskLoadError(f"{field} must be a string")
    return value


def _required_str_list(raw: dict[str, Any], field: str) -> list[str]:
    if field not in raw:
        raise TaskLoadError(f"missing required field: {field}")
    value = raw[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskLoadError(f"{field} must be a list of strings")
    return value
