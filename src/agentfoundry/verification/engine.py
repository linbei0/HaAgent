"""
agentfoundry/verification/engine.py - Verification Engine

执行 task.yaml 中声明的 verification_commands，并写入 verification trace。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentfoundry.runtime.command import run_command
from agentfoundry.runtime.episode import EpisodeWriter


@dataclass(frozen=True)
class VerificationResult:
    status: str
    failed_command: str | None = None
    exit_code: int | None = None
    failure_reason: str | None = None


class VerificationEngine:
    def __init__(
        self,
        episode_writer: EpisodeWriter,
        workspace_root: Path,
        timeout_seconds: float = 60,
    ) -> None:
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root
        self._timeout_seconds = timeout_seconds
        self._commands_log = episode_writer.path / "verification" / "commands.jsonl"
        self._commands_log.parent.mkdir(parents=True, exist_ok=True)
        self._commands_log.write_text("", encoding="utf-8")

    def run(self, commands: list[str]) -> VerificationResult:
        """逐条执行验证命令，任一命令失败即返回 failed。"""
        for command in commands:
            record = self._run_command(command)
            self._append_record(record)
            if record["exit_code"] != 0:
                return VerificationResult(
                    status="failed",
                    failed_command=command,
                    exit_code=record["exit_code"],
                    failure_reason=str(record["status"]),
                )
        return VerificationResult(status="success")

    def _run_command(self, command: str) -> dict[str, object]:
        return run_command(command, self._workspace_root, self._timeout_seconds).to_dict()

    def _append_record(self, record: dict[str, object]) -> None:
        with self._commands_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
