"""
tests/unit/context/test_soul.py - Soul 文件加载测试

验证用户级与可信工作区 Soul 的加载、合并、跳过审计和失败边界。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from haagent.context.soul import (
    MAX_SOUL_FILE_BYTES,
    SoulLoadError,
    load_soul,
)
from haagent.runtime.settings import SoulSettings


def test_load_soul_missing_files_is_noop(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(Path, "home", lambda: home)

    result = load_soul(workspace, SoulSettings())

    assert result.content is None
    assert result.metadata == {"sources": []}
    assert result.skip_reason is None


def test_load_soul_merges_global_before_trusted_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    global_soul = home / ".haagent" / "SOUL.md"
    workspace_soul = workspace / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    workspace.mkdir()
    global_soul.write_text("Be calm.", encoding="utf-8")
    workspace_soul.write_text("Be concise here.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)

    result = load_soul(
        workspace,
        SoulSettings(trusted_workspace_roots=(str(workspace.resolve()),)),
    )

    assert result.content is not None
    assert result.content.index("Be calm.") < result.content.index("Be concise here.")
    assert [source["status"] for source in result.metadata["sources"]] == [
        "loaded",
        "loaded",
    ]
    json.dumps(result.metadata)
    assert result.skip_reason is None


def test_load_soul_does_not_read_untrusted_workspace_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_soul = workspace / "SOUL.md"
    workspace_soul.write_text("untrusted body", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if self == workspace_soul:
            raise AssertionError("untrusted workspace Soul body was read")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    result = load_soul(workspace, SoulSettings())

    assert result.content is None
    assert result.skip_reason == "workspace_untrusted"
    assert result.metadata["sources"] == [
        {
            "scope": "workspace",
            "path": str(workspace_soul),
            "status": "skipped_untrusted",
        },
    ]


def test_load_soul_rejects_workspace_target_outside_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_soul = workspace / "SOUL.md"
    workspace_soul.write_text("present", encoding="utf-8")
    outside_target = tmp_path / "outside-secret.txt"
    monkeypatch.setattr(Path, "home", lambda: home)
    original_resolve = Path.resolve

    def escaped_resolve(self: Path, *args, **kwargs) -> Path:
        if self == workspace_soul:
            return outside_target
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)

    with pytest.raises(SoulLoadError, match="escapes workspace"):
        load_soul(
            workspace,
            SoulSettings(trusted_workspace_roots=(str(workspace),)),
        )


def test_load_soul_rejects_oversized_file_before_reading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_bytes(b"x" * (MAX_SOUL_FILE_BYTES + 1))
    monkeypatch.setattr(Path, "home", lambda: home)
    original_open = Path.open

    class ReadForbiddenFile:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self._handle.close()

        def fileno(self) -> int:
            return self._handle.fileno()

        def read(self, size: int) -> bytes:
            raise AssertionError("oversized Soul body was read")

    def guarded_open(self: Path, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self == global_soul:
            return ReadForbiddenFile(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)

    with pytest.raises(SoulLoadError, match="exceeds 2097152 bytes"):
        load_soul(tmp_path / "workspace", SoulSettings())


def test_load_soul_rejects_growth_after_fstat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_text("present", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    original_open = Path.open
    original_fstat = os.fstat
    fake_fd = 98765

    class GrowingFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def fileno(self) -> int:
            return fake_fd

        def read(self, size: int) -> bytes:
            assert size == MAX_SOUL_FILE_BYTES + 1
            return b"x" * size

    def growing_open(self: Path, *args, **kwargs):
        if self == global_soul:
            return GrowingFile()
        return original_open(self, *args, **kwargs)

    def small_fstat(fd: int):
        if fd == fake_fd:
            return SimpleNamespace(st_size=1)
        return original_fstat(fd)

    monkeypatch.setattr(Path, "open", growing_open)
    monkeypatch.setattr(os, "fstat", small_fstat)

    with pytest.raises(SoulLoadError, match="exceeds 2097152 bytes"):
        load_soul(tmp_path / "workspace", SoulSettings())


def test_load_soul_treats_blank_content_as_noop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)

    result = load_soul(tmp_path / "workspace", SoulSettings())

    assert result.content is None
    assert result.skip_reason is None
    assert result.metadata["sources"][0]["status"] == "empty"


def test_load_soul_exposes_utf8_or_io_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_text("present", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    original_open = Path.open

    def denied_open(self: Path, *args, **kwargs):
        if self == global_soul:
            raise PermissionError("denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)

    with pytest.raises(SoulLoadError, match="cannot read Soul file as UTF-8"):
        load_soul(tmp_path / "workspace", SoulSettings())


def test_load_soul_wraps_existence_probe_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    global_soul = home / ".haagent" / "SOUL.md"
    global_soul.parent.mkdir(parents=True)
    global_soul.write_text("present", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    original_is_file = Path.is_file

    def denied_is_file(self: Path) -> bool:
        if self == global_soul:
            raise PermissionError("denied")
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_file", denied_is_file)

    with pytest.raises(SoulLoadError, match="cannot read Soul file as UTF-8"):
        load_soul(tmp_path / "workspace", SoulSettings())


def test_load_soul_dedupes_workspace_when_same_as_global(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    soul = config_dir / "SOUL.md"
    soul.write_text("ONE-BODY", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)

    result = load_soul(
        config_dir,
        SoulSettings(trusted_workspace_roots=(str(config_dir.resolve()),)),
    )

    assert result.content is not None
    assert result.content.count("ONE-BODY") == 1
    assert result.metadata["sources"] == [
        {
            "scope": "global",
            "path": str(soul),
            "status": "loaded",
            "chars": len("ONE-BODY"),
        },
    ]
    assert result.skip_reason is None
