"""
tests/support/episode_packages.py - Episode package 测试构造器

提供 validator、inspect 和 eval 测试复用的 episode 文件读写与 trace 构造工具。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.runtime.orchestration.orchestrator import RunOrchestrator


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


class EpisodePackageBuilder:
    """用真实 orchestrator 或最小文件集合构造 episode package。"""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.episode_path = tmp_path / "episode-1"

    def create_completed(self) -> Path:
        task_path = self.tmp_path / "task.yaml"
        write_task(task_path)
        result = RunOrchestrator(runs_root=self.tmp_path / ".runs").run(task_path)
        self.episode_path = result.episode_path
        return self.episode_path

    def create_failed(self, stage: str, category: str) -> Path:
        self.episode_path.mkdir(parents=True, exist_ok=True)
        write_json(self.episode_path / "episode.json", valid_episode_json(self.tmp_path, status="failed"))
        write_json(
            self.episode_path / "failure.json",
            {
                "status": "failed",
                "failure": {
                    "category": category,
                    "stage": stage,
                    "evidence": "builder failure",
                },
            },
        )
        return self.episode_path

    def update_json(self, relative_path: str, updates: dict[str, object]) -> None:
        path = self.episode_path / relative_path
        payload = read_json(path)
        payload.update(updates)
        write_json(path, payload)

    def remove(self, relative_path: str) -> None:
        (self.episode_path / relative_path).unlink()
