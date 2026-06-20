"""
agentfoundry/runtime/task_contract.py - task.yaml 加载与校验

把用户提供的 YAML 任务规格转换成运行时使用的 TaskSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentfoundry.tools.registry import TOOL_REGISTRY


class TaskLoadError(ValueError):
    """Raised when task.yaml cannot be loaded as a valid task spec."""


@dataclass(frozen=True)
class TaskSpec:
    goal: str
    constraints: list[str]
    allowed_tools: list[str]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    workspace_root: str | None = None
    policy: dict[str, list[str]] = field(default_factory=lambda: {"approval_allowed_tools": []})


def load_task(path: Path) -> TaskSpec:
    """读取 task.yaml，并校验 MVP 所需的五个必填字段。"""
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
        workspace_root=_optional_str(raw, "workspace_root"),
        policy=_optional_policy(raw),
    )


def resolve_workspace_root(task: TaskSpec, task_path: Path) -> Path:
    """解析 task.yaml 的 workspace_root，失败时显式暴露为 TaskLoadError。"""
    raw_root = task.workspace_root
    candidate = task_path.parent if raw_root is None else Path(raw_root)
    if raw_root is not None and not candidate.is_absolute():
        candidate = task_path.parent / candidate
    workspace_root = candidate.resolve()
    if not workspace_root.exists():
        raise TaskLoadError(f"workspace_root does not exist: {workspace_root}")
    if not workspace_root.is_dir():
        raise TaskLoadError(f"workspace_root must be a directory: {workspace_root}")
    return workspace_root


def _required_str(raw: dict[str, Any], field: str) -> str:
    if field not in raw:
        raise TaskLoadError(f"missing required field: {field}")
    value = raw[field]
    if not isinstance(value, str):
        raise TaskLoadError(f"{field} must be a string")
    return value


def _optional_str(raw: dict[str, Any], field: str) -> str | None:
    if field not in raw:
        return None
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


def _optional_policy(raw: dict[str, Any]) -> dict[str, list[str]]:
    policy = raw.get("policy", {})
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise TaskLoadError("policy must be a mapping")
    approval_allowed_tools = policy.get("approval_allowed_tools", [])
    if not isinstance(approval_allowed_tools, list) or not all(
        isinstance(item, str)
        for item in approval_allowed_tools
    ):
        raise TaskLoadError("policy.approval_allowed_tools must be a list of strings")
    unknown_tools = [tool for tool in approval_allowed_tools if tool not in TOOL_REGISTRY]
    if unknown_tools:
        raise TaskLoadError(
            f"unknown policy.approval_allowed_tools: {', '.join(unknown_tools)}",
        )
    return {"approval_allowed_tools": approval_allowed_tools}
