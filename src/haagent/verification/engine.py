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
            self._append_record(record)
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
        command_result = run_command(command, self._workspace_root, self._timeout_seconds).to_dict()
        redacted_command, _command_redacted = _redact(str(command_result["command"]))
        stdout = str(command_result["stdout"])
        stderr = str(command_result["stderr"])
        stdout_excerpt = _build_excerpt(stdout)
        stderr_excerpt = _build_excerpt(stderr)
        return {
            **command_result,
            "command": redacted_command,
            "stdout": stdout_excerpt.text,
            "stderr": stderr_excerpt.text,
            "timeout": command_result["status"] == "timeout",
            "stdout_excerpt": stdout_excerpt.excerpt,
            "stderr_excerpt": stderr_excerpt.excerpt,
            "stdout_truncated": stdout_excerpt.truncated,
            "stderr_truncated": stderr_excerpt.truncated,
            "stdout_original_length": stdout_excerpt.original_length,
            "stderr_original_length": stderr_excerpt.original_length,
            "redacted": stdout_excerpt.redacted or stderr_excerpt.redacted,
        }

    def _append_record(self, record: dict[str, object]) -> None:
        with self._commands_log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Excerpt:
    text: str
    excerpt: str
    truncated: bool
    original_length: int
    redacted: bool


def _build_excerpt(text: str) -> Excerpt:
    redacted_text, redacted = _redact(text)
    return Excerpt(
        text=redacted_text,
        excerpt=redacted_text[:EXCERPT_LIMIT],
        truncated=len(redacted_text) > EXCERPT_LIMIT,
        original_length=len(text),
        redacted=redacted,
    )


def _redact(text: str) -> tuple[str, bool]:
    redacted = OPENAI_API_KEY_PATTERN.sub("OPENAI_API_KEY=[REDACTED]", text)
    redacted = TOKEN_PATTERN.sub("[REDACTED_TOKEN]", redacted)
    return redacted, redacted != text
