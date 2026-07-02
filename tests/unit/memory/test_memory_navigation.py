from __future__ import annotations

import json
from pathlib import Path

from haagent.memory.navigation import MemoryNavigationBudget, build_memory_navigation
from haagent.memory.schema import MEMORY_SCHEMA_VERSION


def test_memory_navigation_index_renders_active_entries_without_body(tmp_path: Path) -> None:
    _write_index(
        tmp_path / ".haagent" / "memory" / "index.json",
        source="workspace",
        items=[
            {
                "id": "mem-fact",
                "category": "facts",
                "title": "Python path",
                "summary": "Use uv run for project commands",
                "tags": ["python", "uv"],
                "updated_at": "2026-01-01T00:00:00Z",
                "status": "active",
            },
            {
                "id": "mem-old",
                "category": "facts",
                "title": "Deleted item",
                "summary": "This should not appear",
                "tags": [],
                "updated_at": "2026-01-02T00:00:00Z",
                "status": "deleted",
            },
        ],
    )

    result = build_memory_navigation(workspace_root=tmp_path)

    assert result.entries_count == 1
    assert "Memory/SOP Navigation Index:" in result.content
    assert "scope=workspace category=facts id=mem-fact title=Python path" in result.content
    assert "tags=python, uv" in result.content
    assert "summary=Use uv run for project commands" in result.content
    assert "body=" not in result.content
    assert "This should not appear" not in result.content


def test_memory_navigation_index_prioritizes_workspace_sop_before_other_memory(tmp_path: Path) -> None:
    _write_index(
        tmp_path / ".haagent" / "memory" / "index.json",
        source="workspace",
        items=[
            {
                "id": "mem-fact-new",
                "category": "facts",
                "title": "New fact",
                "summary": "Fact summary",
                "tags": [],
                "updated_at": "2026-02-01T00:00:00Z",
                "status": "active",
            },
            {
                "id": "mem-sop-old",
                "category": "sop",
                "title": "Old SOP",
                "summary": "SOP summary",
                "tags": ["workflow"],
                "updated_at": "2026-01-01T00:00:00Z",
                "status": "active",
            },
        ],
    )

    result = build_memory_navigation(workspace_root=tmp_path)

    assert result.content.index("id=mem-sop-old") < result.content.index("id=mem-fact-new")


def test_empty_memory_index_is_skipped_and_not_in_model_input(tmp_path: Path) -> None:
    _write_index(tmp_path / ".haagent" / "memory" / "index.json", source="workspace", items=[])

    result = build_memory_navigation(workspace_root=tmp_path)

    assert result.content == ""
    assert result.entries_count == 0
    assert result.diagnostics["decision"] == "skipped"
    assert result.diagnostics["reason"] == "empty"


def test_missing_memory_index_records_skipped_reason_in_manifest(tmp_path: Path) -> None:
    result = build_memory_navigation(workspace_root=tmp_path)

    assert result.content == ""
    assert result.entries_count == 0
    assert result.diagnostics["decision"] == "skipped"
    assert result.diagnostics["reason"] == "missing_index"
    assert "workspace" in result.diagnostics["missing_scopes"]


def test_memory_index_budget_truncates_and_records_diagnostics(tmp_path: Path) -> None:
    _write_index(
        tmp_path / ".haagent" / "memory" / "index.json",
        source="workspace",
        items=[
            {
                "id": f"mem-{index}",
                "category": "sop",
                "title": f"SOP {index}",
                "summary": "long summary " * 12,
                "tags": ["tag"],
                "updated_at": f"2026-01-{index + 1:02d}T00:00:00Z",
                "status": "active",
            }
            for index in range(8)
        ],
    )

    result = build_memory_navigation(
        workspace_root=tmp_path,
        budget=MemoryNavigationBudget(max_chars=260),
    )

    assert result.truncated is True
    assert len(result.content) <= 260
    assert result.diagnostics["decision"] == "selected"
    assert result.diagnostics["truncated"] is True
    assert result.diagnostics["rendered_entries"] < result.entries_count


def _write_index(path: Path, *, source: str, items: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": MEMORY_SCHEMA_VERSION,
                "updated_at": "2026-01-01T00:00:00Z",
                "source": source,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
