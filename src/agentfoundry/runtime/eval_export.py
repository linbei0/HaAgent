"""
agentfoundry/runtime/eval_export.py - Eval Case 导出器

把已校验的 episode package 转换为可审计、可序列化的最小 eval case 字典。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode_validator import validate_episode_package
from agentfoundry.runtime.task_contract import load_task


def export_eval_case(episode_path: Path) -> dict[str, Any]:
    """导出单个 episode 的 eval case；入口先执行完整 package 校验。"""
    validate_episode_package(episode_path)

    episode_metadata = _read_json(episode_path / "episode.json")
    failure_record = _read_json(episode_path / "failure.json")
    task = load_task(episode_path / "task.yaml")
    verification_commands = _read_jsonl(episode_path / "verification" / "commands.jsonl")
    tool_calls = _read_jsonl(episode_path / "tool-calls.jsonl")

    return {
        "episode_version": episode_metadata["episode_version"],
        "task": {
            "goal": task.goal,
            "acceptance_criteria": task.acceptance_criteria,
            "verification_commands": task.verification_commands,
        },
        "workspace_root": episode_metadata["workspace_root"],
        "final_status": episode_metadata["status"],
        "failure": _failure_summary(failure_record),
        "verification": _verification_summary(verification_commands),
        "tool_names_used": _tool_names_used(tool_calls),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
