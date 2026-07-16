"""
haagent/memory/navigation.py - 长期记忆导航目录

从现有 user/workspace memory index 派生薄目录，供模型按需发现长期知识。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from haagent.memory.schema import USER_SCOPE, WORKSPACE_SCOPE
from haagent.models.config.connections import user_config_dir


@dataclass(frozen=True)
class MemoryNavigationBudget:
    max_chars: int = 1200


@dataclass(frozen=True)
class MemoryNavigationEntry:
    scope: str
    category: str
    id: str
    title: str
    summary: str
    tags: list[str]
    updated_at: str


@dataclass(frozen=True)
class MemoryNavigationResult:
    content: str
    entries_count: int
    truncated: bool
    diagnostics: dict[str, Any]


_CATEGORY_ORDER = {
    (WORKSPACE_SCOPE, "sop"): 0,
    (WORKSPACE_SCOPE, "facts"): 1,
    (WORKSPACE_SCOPE, "decisions"): 2,
    (WORKSPACE_SCOPE, "glossary"): 3,
    (USER_SCOPE, "constraints"): 4,
    (USER_SCOPE, "user_preferences"): 5,
    (USER_SCOPE, "habits"): 6,
}


def build_memory_navigation(
    *,
    workspace_root: Path,
    user_memory_root: Path | None = None,
    budget: MemoryNavigationBudget = MemoryNavigationBudget(),
) -> MemoryNavigationResult:
    entries: list[MemoryNavigationEntry] = []
    missing_scopes: list[str] = []
    invalid_scopes: list[str] = []
    seen_index = False

    for scope, root in [
        (WORKSPACE_SCOPE, workspace_root.resolve() / ".haagent" / "memory"),
        (USER_SCOPE, (user_memory_root or user_config_dir() / "memory").resolve()),
    ]:
        path = root / "index.json"
        if not path.exists():
            missing_scopes.append(scope)
            continue
        seen_index = True
        try:
            entries.extend(_read_index_entries(path, scope))
        except (OSError, json.JSONDecodeError, ValueError):
            invalid_scopes.append(scope)

    entries = sorted(entries, key=_entry_sort_key)
    content, rendered_entries, truncated = _render_entries(entries, budget.max_chars)
    if not entries:
        reason = "empty" if seen_index else "missing_index"
        if invalid_scopes:
            reason = "invalid_index"
        return MemoryNavigationResult(
            content="",
            entries_count=0,
            truncated=False,
            diagnostics={
                "decision": "skipped",
                "reason": reason,
                "missing_scopes": missing_scopes,
                "invalid_scopes": invalid_scopes,
                "entries_count": 0,
                "rendered_entries": 0,
                "truncated": False,
            },
        )

    return MemoryNavigationResult(
        content=content,
        entries_count=len(entries),
        truncated=truncated,
        diagnostics={
            "decision": "selected",
            "reason": "confirmed_memory_index_available",
            "missing_scopes": missing_scopes,
            "invalid_scopes": invalid_scopes,
            "entries_count": len(entries),
            "rendered_entries": rendered_entries,
            "truncated": truncated,
            "max_chars": max(0, budget.max_chars),
            "content_chars": len(content),
        },
    )


def _read_index_entries(path: Path, scope: str) -> list[MemoryNavigationEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("memory index must be an object")
    raw_items = raw.get("items", [])
    if not isinstance(raw_items, list):
        raise ValueError("memory index items must be a list")
    entries: list[MemoryNavigationEntry] = []
    for item in raw_items:
        if not isinstance(item, dict) or item.get("status") != "active":
            continue
        entries.append(
            MemoryNavigationEntry(
                scope=scope,
                category=_string(item.get("category")),
                id=_string(item.get("id")),
                title=_string(item.get("title")),
                summary=_string(item.get("summary")),
                tags=_string_list(item.get("tags")),
                updated_at=_string(item.get("updated_at")),
            ),
        )
    return entries


def _render_entries(entries: list[MemoryNavigationEntry], max_chars: int) -> tuple[str, int, bool]:
    if not entries or max_chars <= 0:
        return "", 0, bool(entries)
    lines = [
        "Memory/SOP Navigation Index:",
        "Use this as a directory of confirmed long-term knowledge; read or retrieve details only when needed.",
    ]
    rendered = 0
    truncated = False
    for entry in entries:
        tags = ", ".join(entry.tags) if entry.tags else "none"
        line = (
            f"- scope={entry.scope} category={entry.category} id={entry.id} title={entry.title}; "
            f"tags={tags}; summary={entry.summary}"
        )
        candidate = "\n".join([*lines, line])
        if len(candidate) > max_chars:
            truncated = True
            break
        lines.append(line)
        rendered += 1
    return "\n".join(lines) if rendered else "", rendered, truncated


def _entry_sort_key(entry: MemoryNavigationEntry) -> tuple[int, str]:
    return (_CATEGORY_ORDER.get((entry.scope, entry.category), 100), _reverse_text(entry.updated_at))


def _reverse_text(value: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in value)


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
