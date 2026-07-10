"""
haagent/verification/engine.py - Verification Engine

执行 task.yaml 中声明的 verification_commands，并写入 verification trace。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

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
        self._files_log = episode_writer.path / "verification" / "files.jsonl"
        self._files_log.write_text("", encoding="utf-8")

    def run(
        self,
        commands: list[str],
        *,
        changed_files: list[dict[str, object]] | None = None,
    ) -> VerificationResult:
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
        for change in _latest_workspace_changes(changed_files or [], self._workspace_root):
            record = self._file_evidence(change)
            self._append_file_record(record)
            if record["status"] != "success":
                return VerificationResult(
                    status="failed",
                    failure_reason=str(record["reason"]),
                )
        return VerificationResult(status="success")

    def _file_evidence(self, change: dict[str, object]) -> dict[str, object]:
        raw_path = change.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return {
                "path": "<unknown>",
                "change_type": _change_type(change),
                "status": "failed",
                "reason": "file change observation is missing a path",
            }
        path = Path(raw_path).resolve()
        relative_path = path.relative_to(self._workspace_root.resolve()).as_posix()
        if not path.exists():
            return {
                "path": relative_path,
                "change_type": _change_type(change),
                "status": "failed",
                "reason": "changed workspace file is missing",
            }
        if not path.is_file():
            return {
                "path": relative_path,
                "change_type": _change_type(change),
                "status": "failed",
                "reason": "changed workspace path is not a file",
            }
        content = path.read_bytes()
        return {
            "path": relative_path,
            "change_type": _change_type(change),
            "status": "success",
            "size_bytes": len(content),
            "sha256": sha256(content).hexdigest(),
        }

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

    def _append_file_record(self, record: dict[str, object]) -> None:
        with self._files_log.open("a", encoding="utf-8") as file:
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


def _latest_workspace_changes(
    changed_files: list[dict[str, object]],
    workspace_root: Path,
) -> list[dict[str, object]]:
    """仅保留 workspace 内每个文件的最后一次成功改动，避免泄露外部路径。"""
    if not changed_files:
        return []
    workspace = workspace_root.resolve()
    latest: dict[Path, dict[str, object]] = {}
    for change in changed_files:
        raw_path = change.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).resolve()
        try:
            path.relative_to(workspace)
        except ValueError:
            continue
        latest[path] = change
    return list(latest.values())


def _change_type(change: dict[str, object]) -> str:
    value = change.get("change_type")
    return value if isinstance(value, str) and value else "modified"
