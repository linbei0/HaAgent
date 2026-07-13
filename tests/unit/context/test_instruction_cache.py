"""
tests/unit/context/test_instruction_cache.py - InstructionCache 合同测试

覆盖 AGENTS.md 元数据命中、删除、变更与 OSError 显式失败。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from haagent.context.builder import ContextBuildError
from haagent.context.instruction_cache import InstructionCache


def test_instruction_cache_hit_reads_once(tmp_path: Path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# rules\nkeep short\n", encoding="utf-8")
    cache = InstructionCache()
    reads = {"count": 0}
    original = Path.read_text

    def counting_read(self, *args, **kwargs):
        if self.name == "AGENTS.md":
            reads["count"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)

    first = cache.load(tmp_path)
    second = cache.load(tmp_path)

    assert first is second
    assert first.content == "# rules\nkeep short\n"
    assert first.source_path == str(agents.resolve())
    assert reads["count"] == 1


def test_instruction_cache_reloads_when_mtime_or_size_changes(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("old\n", encoding="utf-8")
    cache = InstructionCache()
    first = cache.load(tmp_path)
    time.sleep(0.02)
    agents.write_text("new body\n", encoding="utf-8")
    second = cache.load(tmp_path)
    assert first is not second
    assert second.content == "new body\n"


def test_instruction_cache_missing_file_returns_none(tmp_path: Path) -> None:
    cache = InstructionCache()
    loaded = cache.load(tmp_path)
    assert loaded.content is None
    assert loaded.source_path is None
    again = cache.load(tmp_path)
    assert again is loaded


def test_instruction_cache_delete_invalidates_to_none(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("gone soon\n", encoding="utf-8")
    cache = InstructionCache()
    first = cache.load(tmp_path)
    assert first.content == "gone soon\n"
    agents.unlink()
    second = cache.load(tmp_path)
    assert second is not first
    assert second.content is None


def test_instruction_cache_oserror_becomes_context_build_error(tmp_path: Path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("secret\n", encoding="utf-8")
    cache = InstructionCache()
    cache.load(tmp_path)
    agents.write_text("changed content\n", encoding="utf-8")

    def boom(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", boom)
    with pytest.raises(ContextBuildError, match="AGENTS.md"):
        cache.load(tmp_path)


def test_instruction_cache_stat_error_is_not_treated_as_missing(tmp_path: Path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("rules\n", encoding="utf-8")
    original_stat = Path.stat

    def denied_stat(self, *args, **kwargs):
        if self == agents:
            raise PermissionError("metadata denied")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)

    with pytest.raises(ContextBuildError, match="metadata.*AGENTS.md|AGENTS.md.*metadata"):
        InstructionCache().load(tmp_path)
