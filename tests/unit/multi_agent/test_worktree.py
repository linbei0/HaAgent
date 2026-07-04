"""
tests/unit/multi_agent/test_worktree.py - worker worktree 路径校验测试

验证后续隔离工作区不会接受路径穿越或绝对路径。
"""

import pytest
import subprocess
from pathlib import Path

from haagent.multi_agent.worktree import cleanup_worktree, create_worktree, validate_worktree_slug


def test_validate_worktree_slug_accepts_simple_slug() -> None:
    assert validate_worktree_slug("fix-tests") == "fix-tests"


@pytest.mark.parametrize("slug", ["", ".", "..", "../x", "x/../y", "/tmp/x", "C:\\tmp", "bad name"])
def test_validate_worktree_slug_rejects_unsafe_values(slug: str) -> None:
    with pytest.raises(ValueError):
        validate_worktree_slug(slug)


def test_create_worktree_creates_real_git_worktree(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")

    lease = create_worktree(repo, slug="worker-a")

    assert lease.worktree_path.exists()
    assert (lease.worktree_path / ".git").exists()
    assert lease.branch_name.endswith("worker-a")

    cleanup_worktree(lease)

    assert not lease.worktree_path.exists()


def _init_git_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    (path / "README.md").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, check=True, capture_output=True, text=True)
    return path
