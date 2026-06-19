"""
agentfoundry/runtime/command.py - 统一命令执行器

封装 subprocess.run 的成功、非零退出和 timeout 结果。
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timeout_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_command(command: str, cwd: Path, timeout_seconds: float) -> CommandResult:
    """运行 shell 命令，并用统一结构表达执行结果。"""
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return CommandResult(
            command=command,
            status="timeout",
            exit_code=None,
            stdout=_decode_timeout_output(error.stdout),
            stderr=_decode_timeout_output(error.stderr),
            duration_seconds=time.perf_counter() - started,
            timeout_seconds=timeout_seconds,
        )

    return CommandResult(
        command=command,
        status="success" if completed.returncode == 0 else "failed",
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.perf_counter() - started,
        timeout_seconds=timeout_seconds,
    )


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
