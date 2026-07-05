"""
src/haagent/runtime/contracts/task.py - task.yaml 加载与校验

把用户提供的 YAML 任务规格转换成运行时使用的 TaskSpec。
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    workspace_root: str | None = None
    path_policy: dict[str, Any] | None = None
    target_paths: list[str] = field(default_factory=list)
    prompt_pack_ids: list[str] = field(default_factory=list)
    policy: dict[str, list[str]] = field(
        default_factory=lambda: {"approval_allowed_tools": [], "approved_tools": []},
    )
    worker_context: dict[str, Any] | None = None


def load_task(path: Path) -> TaskSpec:
    """读取 task.yaml，并校验 MVP 所需的五个必填字段。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TaskLoadError("task.yaml must contain a mapping")

    goal = _required_str(raw, "goal")
    return TaskSpec(
        goal=goal,
        constraints=_required_str_list(raw, "constraints"),
        allowed_tools=(allowed_tools := _required_str_list(raw, "allowed_tools")),
        acceptance_criteria=_required_str_list(raw, "acceptance_criteria"),
        verification_commands=_required_str_list(raw, "verification_commands"),
        workspace_root=_optional_str(raw, "workspace_root"),
        path_policy=_optional_path_policy(raw),
        target_paths=_optional_str_list(raw, "target_paths"),
        prompt_pack_ids=_optional_str_list(raw, "prompt_pack_ids"),
        policy=_optional_policy(raw, allowed_tools),
        worker_context=_optional_mapping(raw, "worker_context"),
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


def _optional_str_list(raw: dict[str, Any], field: str) -> list[str]:
    value = raw.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskLoadError(f"{field} must be a list of strings")
    return value


def _required_str_list(raw: dict[str, Any], field: str) -> list[str]:
    if field not in raw:
        raise TaskLoadError(f"missing required field: {field}")
    value = raw[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskLoadError(f"{field} must be a list of strings")
    return value


def _optional_policy(raw: dict[str, Any], allowed_tools: list[str]) -> dict[str, list[str]]:
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
    approved_tools = policy.get("approved_tools", [])
    if not isinstance(approved_tools, list) or not all(
        isinstance(item, str)
        for item in approved_tools
    ):
        raise TaskLoadError("policy.approved_tools must be a list of strings")
    disallowed_approval_tools = [tool for tool in approval_allowed_tools if tool not in allowed_tools]
    if disallowed_approval_tools:
        raise TaskLoadError(
            "policy.approval_allowed_tools must be included in allowed_tools: "
            + ", ".join(disallowed_approval_tools),
        )
    disallowed_approved_tools = [tool for tool in approved_tools if tool not in allowed_tools]
    if disallowed_approved_tools:
        raise TaskLoadError(
            "policy.approved_tools must be included in allowed_tools: "
            + ", ".join(disallowed_approved_tools),
        )
    not_allowed_tools = [tool for tool in approved_tools if tool not in approval_allowed_tools]
    if not_allowed_tools:
        raise TaskLoadError(
            "approved_tools must also appear in approval_allowed_tools: "
            + ", ".join(not_allowed_tools),
        )
    return {
        "approval_allowed_tools": approval_allowed_tools,
        "approved_tools": approved_tools,
    }


def _optional_path_policy(raw: dict[str, Any]) -> dict[str, Any] | None:
    value = raw.get("path_policy")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TaskLoadError("path_policy must be a mapping")
    return value


def _optional_mapping(raw: dict[str, Any], field: str) -> dict[str, Any] | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TaskLoadError(f"{field} must be a mapping")
    return dict(value)
