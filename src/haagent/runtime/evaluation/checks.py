"""
src/haagent/runtime/evaluation/checks.py - 本地质量门禁编排

复用 eval runner，并可选调用 pytest，生成一键本地自检报告。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from haagent.models.gateway import ModelGateway
from haagent.runtime.evaluation.runner import run_eval_path


OUTPUT_EXCERPT_LIMIT = 4000
DEFAULT_PYTEST_COMMAND = ["uv", "run", "pytest", "-q"]


@dataclass(frozen=True)
class PytestResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        stdout_excerpt, stdout_truncated = _excerpt(self.stdout)
        stderr_excerpt, stderr_truncated = _excerpt(self.stderr)
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout_excerpt": stdout_excerpt,
            "stderr_excerpt": stderr_excerpt,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "summary": pytest_summary(self),
        }


def run_quality_checks(
    *,
    eval_path: Path,
    runs_root: Path,
    model_gateway: ModelGateway | None = None,
    run_pytest: bool = False,
    pytest_command: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    started_at = _now_iso()
    eval_report = run_eval_path(
        eval_path,
        runs_root=runs_root,
        model_gateway=model_gateway,
        max_turns=5,
    )
    checks = [_eval_check_summary(eval_report)]
    pytest_result = None
    if run_pytest:
        command = list(pytest_command or DEFAULT_PYTEST_COMMAND)
        pytest_result = run_pytest_command(command, cwd or Path.cwd())
        checks.append(_pytest_check_summary(pytest_result))
    status = "passed" if all(check["status"] == "passed" for check in checks) else "failed"
    report: dict[str, Any] = {
        "status": status,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "checks": checks,
        "eval_report": eval_report,
    }
    if pytest_result is not None:
        report["pytest"] = pytest_result.to_dict()
    return report


def run_pytest_command(command: Sequence[str], cwd: Path) -> PytestResult:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return PytestResult(
        command=list(command),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def pytest_summary(result: PytestResult) -> str:
    stdout_line = _last_non_empty_line(result.stdout)
    if stdout_line:
        return stdout_line
    stderr_line = _last_non_empty_line(result.stderr)
    return stderr_line or "none"


def _eval_check_summary(eval_report: dict[str, Any]) -> dict[str, Any]:
    failed = int(eval_report["failed_count"])
    errors = int(eval_report["error_count"])
    return {
        "name": "eval",
        "status": "passed" if failed == 0 and errors == 0 else "failed",
        "total": eval_report["total_count"],
        "passed": eval_report["passed_count"],
        "failed": failed,
        "error": errors,
    }


def _pytest_check_summary(result: PytestResult) -> dict[str, Any]:
    return {
        "name": "pytest",
        "status": "passed" if result.exit_code == 0 else "failed",
        "exit_code": result.exit_code,
        "summary": pytest_summary(result),
    }


def _excerpt(text: str) -> tuple[str, bool]:
    truncated = len(text) > OUTPUT_EXCERPT_LIMIT
    return text[:OUTPUT_EXCERPT_LIMIT], truncated


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
