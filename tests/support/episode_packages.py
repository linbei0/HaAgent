"""
tests/support/episode_packages.py - Episode package 测试构造器

提供 validator、inspect 和 eval 测试复用的 episode 文件读写与 trace 构造工具。
"""

from __future__ import annotations

import json
from pathlib import Path

def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_task(path: Path) -> None:
    path.write_text(
        """
goal: Validate package
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Run reaches completed state
verification_commands: []
""".strip(),
        encoding="utf-8",
    )


def valid_policy(tool_name: str = "fake_tool") -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "risk_level": "low",
        "action": "allow",
        "reason": "Allowed by test policy",
        "approval": {
            "required": False,
            "status": "not_required",
            "reason": "low risk",
        },
    }


def write_tool_call(
    episode_path: Path,
    *,
    tool_name: object = "fake_tool",
    status: object = "success",
    policy: object = None,
    include_policy: bool = True,
    error: object | None = None,
) -> None:
    record: dict[str, object] = {"tool_name": tool_name, "status": status}
    if include_policy:
        record["policy"] = valid_policy(str(tool_name)) if policy is None else policy
    if error is not None:
        record["error"] = error
    (episode_path / "tool-calls.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def valid_verification_command() -> dict[str, object]:
    return {
        "command": "uv run pytest",
        "status": "success",
        "exit_code": 0,
        "timeout": False,
        "stdout_excerpt": "",
        "stderr_excerpt": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_length": 0,
        "stderr_original_length": 0,
        "redacted": False,
    }


def write_verification_command(episode_path: Path, **updates: object) -> None:
    record = valid_verification_command()
    record.update(updates)
    (episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
