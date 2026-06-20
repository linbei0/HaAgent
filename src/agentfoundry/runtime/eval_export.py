"""
agentfoundry/runtime/eval_export.py - Eval Case 导出器

把已校验的 episode package 转换为可审计、可序列化的最小 eval case 字典。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode_validator import load_validated_episode_package
from agentfoundry.runtime.task_contract import load_task


EVAL_CASE_VERSION = "1.0"


def export_eval_case(episode_path: Path) -> dict[str, Any]:
    """导出单个 episode 的 eval case；入口先执行完整 package 校验。"""
    package_view = load_validated_episode_package(episode_path)

    episode_metadata = package_view.episode_metadata
    failure_record = package_view.failure_record
    task = load_task(episode_path / "task.yaml")

    return {
        "eval_case_version": EVAL_CASE_VERSION,
        "episode_version": episode_metadata["episode_version"],
        "task": {
            "goal": task.goal,
            "acceptance_criteria": task.acceptance_criteria,
            "verification_commands": task.verification_commands,
        },
        "workspace_root": episode_metadata["workspace_root"],
        "final_status": episode_metadata["status"],
        "failure": _failure_summary(failure_record),
        "verification": _verification_summary(package_view.verification_commands),
        "tool_names_used": _tool_names_used(package_view.tool_calls),
        "next_actions": _next_actions_summary(episode_path, package_view.context_manifest),
    }


def _failure_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["status"] == "success":
        return None
    failure = record["failure"]
    return {
        "category": failure["category"],
        "stage": failure["stage"],
        "evidence": failure["evidence"],
    }


def _verification_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "command": record["command"],
            "status": record["status"],
            "exit_code": record.get("exit_code"),
            "timeout": bool(record.get("timeout", False)),
        }
        for record in records
    ]


def _tool_names_used(records: list[dict[str, Any]]) -> list[str]:
    names = {str(record["tool_name"]) for record in records}
    return sorted(names)


def _next_actions_summary(episode_path: Path, context_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = context_manifest.get("contexts", [])
    if not isinstance(contexts, list):
        return []
    return [_next_action_summary(episode_path, context) for context in contexts if isinstance(context, dict)]


def _next_action_summary(episode_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    context_id = str(context.get("context_id", "unknown"))
    manifest_path = context.get("manifest_path")
    if not isinstance(manifest_path, str):
        return _missing_next_action(context_id)
    next_action = _read_next_action(episode_path / manifest_path)
    if next_action is None:
        return _missing_next_action(context_id)
    return {
        "context_id": context_id,
        "status": str(next_action.get("status", "missing")),
        "reason": str(next_action.get("reason", "legacy/missing")),
        "based_on_observation_index": next_action.get("based_on_observation_index"),
        "based_on_tool_name": next_action.get("based_on_tool_name"),
    }


def _read_next_action(path: Path) -> dict[str, Any] | None:
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(context, dict):
        return None
    next_action = context.get("next_action")
    if not isinstance(next_action, dict):
        return None
    return next_action


def _missing_next_action(context_id: str) -> dict[str, Any]:
    return {
        "context_id": context_id,
        "status": "missing",
        "reason": "legacy/missing",
        "based_on_observation_index": None,
        "based_on_tool_name": None,
    }
