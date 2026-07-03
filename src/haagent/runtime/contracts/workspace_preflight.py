"""
src/haagent/runtime/contracts/workspace_preflight.py - Workspace 安全预检

采集 run 开始前的 workspace 路径和轻量 git 状态，供 episode 与 inspect 复盘。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def build_workspace_preflight(
    workspace_root: Path,
    *,
    modifies_original_workspace: bool = True,
) -> dict[str, Any]:
    """返回 workspace preflight 记录；不改变 workspace 状态。"""
    resolved_root = workspace_root.resolve(strict=False)
    exists = resolved_root.exists()
    summary = _empty_dirty_summary()
    record: dict[str, Any] = {
        "workspace_root": str(resolved_root),
        "exists": exists,
        "is_git_repo": False,
        "git_branch": None,
        "git_dirty": None,
        "git_dirty_summary": summary,
        "git_status": "missing" if not exists else "not_git_repo",
        "modifies_original_workspace": modifies_original_workspace,
    }
    if not exists:
        return record

    if _git(resolved_root, "rev-parse", "--show-toplevel") is None:
        return record

    branch_result = _git(resolved_root, "branch", "--show-current")
    status_result = _git(resolved_root, "status", "--porcelain=v1")
    if branch_result is None or status_result is None:
        return {
            **record,
            "is_git_repo": True,
            "git_status": "unknown",
        }

    summary = _dirty_summary(status_result.stdout.splitlines())
    dirty = summary["total"] > 0
    return {
        **record,
        "is_git_repo": True,
        "git_branch": branch_result.stdout.strip() or None,
        "git_dirty": dirty,
        "git_dirty_summary": summary,
        "git_status": "dirty" if dirty else "clean",
    }


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result


def _dirty_summary(lines: list[str]) -> dict[str, int]:
    summary = _empty_dirty_summary()
    for line in lines:
        if not line:
            continue
        summary["total"] += 1
        code = line[:2]
        if code == "??":
            summary["untracked"] += 1
        elif "D" in code:
            summary["deleted"] += 1
        elif "R" in code:
            summary["renamed"] += 1
        elif "M" in code:
            summary["modified"] += 1
        else:
            summary["other"] += 1
    return summary


def _empty_dirty_summary() -> dict[str, int]:
    return {
        "total": 0,
        "modified": 0,
        "untracked": 0,
        "deleted": 0,
        "renamed": 0,
        "other": 0,
    }
