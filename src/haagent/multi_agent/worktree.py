"""
haagent/multi_agent/worktree.py - worker 隔离工作区管理

提供 Git worktree 创建、校验与清理流程，供代码类 worker 使用隔离工作区。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class WorktreeLease:
    branch_name: str
    worktree_path: Path
    base_repo: Path


def validate_worktree_slug(slug: str) -> str:
    if not slug or slug.startswith("/") or "\\" in slug or ":" in slug:
        raise ValueError("invalid worktree slug")
    parts = slug.split("/")
    for part in parts:
        if part in {"", ".", ".."}:
            raise ValueError("invalid worktree slug")
        if not _SEGMENT_PATTERN.fullmatch(part):
            raise ValueError("invalid worktree slug")
    if len(slug) > 64:
        raise ValueError("invalid worktree slug")
    return slug


def create_worktree(base_repo: Path, *, slug: str, parent_dir: Path | None = None) -> WorktreeLease:
    repo = base_repo.resolve()
    _require_git_repo(repo)
    safe = validate_worktree_slug(slug)
    branch_name = f"haagent/{safe}"
    root = (parent_dir.resolve() if parent_dir is not None else repo.parent / ".haagent-worktrees")
    worktree_path = root / safe
    if worktree_path.exists():
        raise ValueError(f"worktree path already exists: {worktree_path}")
    root.mkdir(parents=True, exist_ok=True)
    _run_git(["worktree", "add", "-b", branch_name, str(worktree_path), "HEAD"], cwd=repo)
    return WorktreeLease(
        branch_name=branch_name,
        worktree_path=worktree_path.resolve(),
        base_repo=repo,
    )


def cleanup_worktree(lease: WorktreeLease) -> None:
    repo = lease.base_repo.resolve()
    if lease.worktree_path.exists():
        _run_git(["worktree", "remove", "--force", str(lease.worktree_path)], cwd=repo)
    _run_git(["branch", "-D", lease.branch_name], cwd=repo, check=False)


def _require_git_repo(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise ValueError(f"base repo must be an existing directory: {path}")
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=path)
    root = Path(result.stdout.strip()).resolve()
    if root != path.resolve():
        raise ValueError(f"base repo must be the git repository root: {path}")


def _run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"git exited with {result.returncode}"
        raise ValueError(detail)
    return result
