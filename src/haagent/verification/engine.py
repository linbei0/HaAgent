"""
haagent/verification/engine.py - Verification Engine

执行 task.yaml 中声明的 verification_commands，并写入 verification trace。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from haagent.runtime.execution.command import run_command
from haagent.runtime.episodes.writer import EpisodeWriter


EXCERPT_LIMIT = 2000
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
TOKEN_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
OPENAI_API_KEY_PATTERN = re.compile(r"OPENAI_API_KEY=[^\s'\";]+")


@dataclass(frozen=True)
class VerificationResult:
    status: str
    failed_command: str | None = None
    exit_code: int | None = None
    failure_reason: str | None = None
    timeout: bool = False
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""


class VerificationEngine:
    def __init__(
        self,
        episode_writer: EpisodeWriter,
        workspace_root: Path,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
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
            self._append_jsonl(self._commands_log, record)
            if record["exit_code"] != 0:
                return VerificationResult(
                    status="failed",
                    failed_command=str(record["command"]),
                    exit_code=record["exit_code"],
                    failure_reason=str(record["status"]),
                    timeout=bool(record["timeout"]),
                    stdout_excerpt=str(record["stdout_excerpt"]),
                    stderr_excerpt=str(record["stderr_excerpt"]),
                )
        return VerificationResult(status="success")

    def _run_command(self, command: str) -> dict[str, object]:
        command_result = run_command(command, self._workspace_root, self._timeout_seconds)
        redacted_command, command_redacted = _redact(command_result.command)
        safe_stdout, stdout_redacted = _redact(command_result.stdout)
        safe_stderr, stderr_redacted = _redact(command_result.stderr)
        return {
            "command": redacted_command,
            "status": command_result.status,
            "exit_code": command_result.exit_code,
            "timeout": command_result.timeout,
            "stdout_excerpt": safe_stdout[:EXCERPT_LIMIT],
            "stderr_excerpt": safe_stderr[:EXCERPT_LIMIT],
            "stdout_truncated": len(safe_stdout) > EXCERPT_LIMIT,
            "stderr_truncated": len(safe_stderr) > EXCERPT_LIMIT,
            "stdout_original_length": len(command_result.stdout),
            "stderr_original_length": len(command_result.stderr),
            "redacted": command_redacted or stdout_redacted or stderr_redacted or command_result.redacted,
            "duration_seconds": command_result.duration_seconds,
            "timeout_seconds": command_result.timeout_seconds,
        }

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _redact(text: str) -> tuple[str, bool]:
    redacted = OPENAI_API_KEY_PATTERN.sub("OPENAI_API_KEY=[REDACTED]", text)
    redacted = TOKEN_PATTERN.sub("[REDACTED_TOKEN]", redacted)
    return redacted, redacted != text
