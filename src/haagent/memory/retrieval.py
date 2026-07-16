"""
haagent/memory/retrieval.py - 长期记忆检索

从 workspace/user memory 索引进入，hydrate 已确认事实源，并返回有界可审计结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from haagent.memory.schema import (
    MEMORY_CATEGORIES_BY_SCOPE,
    USER_SCOPE,
    WORKSPACE_SCOPE,
    MemoryRecord,
    MemoryTombstone,
)
from haagent.memory.store import FILE_BY_SCOPE_CATEGORY
from haagent.models.config.connections import user_config_dir


DEFAULT_CATEGORY_PRIORITY = {
    "decisions": 0,
    "sop": 1,
    "facts": 2,
    "glossary": 3,
    "constraints": 4,
    "user_preferences": 5,
    "habits": 6,
}
DIAGNOSTIC_FIELDS = [
    "workspace_index_missing",
    "user_index_missing",
    "skipped_inactive",
    "skipped_deleted",
    "skipped_missing",
    "skipped_invalid",
    "skipped_over_budget",
]


@dataclass(frozen=True)
class MemoryRetrievalBudget:
    max_workspace_items: int = 6
    max_user_items: int = 3
    max_workspace_chars: int = 2400
    max_user_chars: int = 900
    max_item_chars: int = 600
    category_priority: dict[str, int] | None = None

    def priority_for(self, category: str) -> int:
        priorities = self.category_priority or DEFAULT_CATEGORY_PRIORITY
        return priorities.get(category, 100)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workspace_items": self.max_workspace_items,
            "max_user_items": self.max_user_items,
            "max_workspace_chars": self.max_workspace_chars,
            "max_user_chars": self.max_user_chars,
            "max_item_chars": self.max_item_chars,
            "category_priority": dict(self.category_priority or DEFAULT_CATEGORY_PRIORITY),
        }


@dataclass(frozen=True)
class MemoryRetrievalRequest:
    query: str
    workspace_root: Path
    user_memory_root: Path | None = None
    budget: MemoryRetrievalBudget = field(default_factory=MemoryRetrievalBudget)
    task_context: str = ""


@dataclass(frozen=True)
class RetrievedMemory:
    memory_id: str
    scope: str
    category: str
    title: str
    body: str
    tags: list[str]
    updated_at: str
    score: float
    char_count: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory_id,
            "scope": self.scope,
            "category": self.category,
            "title": self.title,
            "updated_at": self.updated_at,
            "score": round(self.score, 3),
            "char_count": self.char_count,
        }


@dataclass(frozen=True)
class MemoryRetrievalResult:
    memories: list[RetrievedMemory]
    budget: MemoryRetrievalBudget
    diagnostics: dict[str, Any]

    def to_model_block(self) -> str:
        if not self.memories:
            return ""
        lines = [
            "Relevant Memory:",
            "Current turn, project instructions, session summary, and working_state override these memories.",
        ]
        for memory in self.memories:
            tags = ", ".join(memory.tags) if memory.tags else "none"
            lines.extend(
                [
                    f"- id={memory.memory_id} scope={memory.scope} category={memory.category} title={memory.title}",
                    f"  tags={tags}",
                    f"  body={memory.body}",
                ],
            )
        return "\n".join(lines)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "used_memories": [memory.to_manifest_dict() for memory in self.memories],
            "budget": self.budget.to_dict(),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class _IndexItem:
    memory_id: str
    scope: str
    category: str
    title: str
    summary: str
    tags: list[str]
    updated_at: str
    status: str


class MemoryRetriever:
    def retrieve(self, request: MemoryRetrievalRequest) -> MemoryRetrievalResult:
        tokens = _tokenize(f"{request.query}\n{request.task_context}")
        diagnostics = _empty_diagnostics()
        candidates: list[RetrievedMemory] = []
        for scope, root in [
            (WORKSPACE_SCOPE, request.workspace_root.resolve() / ".haagent" / "memory"),
            (USER_SCOPE, (request.user_memory_root or user_config_dir() / "memory").resolve()),
        ]:
            candidates.extend(self._read_scope(scope, root, tokens, request.budget, diagnostics))

        selected = _apply_budget(_sort_memories(candidates, request.budget), request.budget, diagnostics)
        return MemoryRetrievalResult(memories=selected, budget=request.budget, diagnostics=diagnostics)

    def _read_scope(
        self,
        scope: str,
        root: Path,
        tokens: set[str],
        budget: MemoryRetrievalBudget,
        diagnostics: dict[str, Any],
    ) -> list[RetrievedMemory]:
        index_path = root / "index.json"
        if not index_path.exists():
            diagnostics[f"{scope}_index_missing"] += 1
            return []
        items = _load_index_items(index_path, scope, diagnostics)
        deleted_ids = _deleted_ids(root, diagnostics)
        records = _load_records_by_id(root, scope, diagnostics)
        memories: list[RetrievedMemory] = []
        for item in items:
            if item.status != "active":
                diagnostics["skipped_deleted" if item.status == "deleted" else "skipped_inactive"] += 1
                continue
            if item.memory_id in deleted_ids:
                diagnostics["skipped_deleted"] += 1
                continue
            record = records.get(item.memory_id)
            if record is None:
                diagnostics["skipped_missing"] += 1
                _remember_skip(diagnostics, "missing_ids", item.memory_id)
                continue
            if record.status != "active":
                diagnostics["skipped_deleted" if record.status == "deleted" else "skipped_inactive"] += 1
                continue
            score = _score(item, record, tokens)
            if score <= 0:
                continue
            body = _bounded(record.body, budget.max_item_chars)
            memories.append(
                RetrievedMemory(
                    memory_id=record.memory_id,
                    scope=record.scope,
                    category=record.category,
                    title=record.title,
                    body=body,
                    tags=list(record.tags),
                    updated_at=record.updated_at,
                    score=score,
                    char_count=len(body),
                ),
            )
        return memories


def _load_index_items(path: Path, scope: str, diagnostics: dict[str, Any]) -> list[_IndexItem]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        diagnostics["skipped_invalid"] += 1
        _remember_skip(diagnostics, "invalid_sources", str(path))
        return []
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        diagnostics["skipped_invalid"] += 1
        return []
    items: list[_IndexItem] = []
    for raw_item in raw["items"]:
        if not isinstance(raw_item, dict):
            diagnostics["skipped_invalid"] += 1
            continue
        try:
            memory_id = _required_str(raw_item, "id")
            category = _required_str(raw_item, "category")
            if category not in MEMORY_CATEGORIES_BY_SCOPE[scope]:
                raise ValueError(category)
            tags = raw_item.get("tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags")
            items.append(
                _IndexItem(
                    memory_id=memory_id,
                    scope=scope,
                    category=category,
                    title=_required_str(raw_item, "title"),
                    summary=_required_str(raw_item, "summary"),
                    tags=list(tags),
                    updated_at=_required_str(raw_item, "updated_at"),
                    status=_required_str(raw_item, "status"),
                ),
            )
        except ValueError:
            diagnostics["skipped_invalid"] += 1
    return items


def _load_records_by_id(root: Path, scope: str, diagnostics: dict[str, Any]) -> dict[str, MemoryRecord]:
    records: dict[str, MemoryRecord] = {}
    for category, file_name in FILE_BY_SCOPE_CATEGORY[scope].items():
        path = root / file_name
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = MemoryRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                diagnostics["skipped_invalid"] += 1
                _remember_skip(diagnostics, "invalid_sources", f"{path}:{line_number}")
                continue
            if record.scope == scope and record.category == category:
                records[record.memory_id] = record
    return records


def _deleted_ids(root: Path, diagnostics: dict[str, Any]) -> set[str]:
    path = root / "tombstones.jsonl"
    if not path.exists():
        return set()
    deleted: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            deleted.add(MemoryTombstone.from_dict(json.loads(line)).memory_id)
        except (json.JSONDecodeError, ValueError):
            diagnostics["skipped_invalid"] += 1
            _remember_skip(diagnostics, "invalid_sources", f"{path}:{line_number}")
    return deleted


def _score(item: _IndexItem, record: MemoryRecord, tokens: set[str]) -> float:
    if not tokens:
        return 0.0
    title = _tokenize(item.title)
    tags = set().union(*(_tokenize(tag) for tag in item.tags)) if item.tags else set()
    category = _tokenize(item.category)
    summary = _tokenize(item.summary)
    body = _tokenize(record.body)
    score = (
        len(tokens & title) * 4.0
        + len(tokens & tags) * 3.0
        + len(tokens & category) * 3.0
        + len(tokens & summary) * 2.0
        + len(tokens & body)
    )
    if score <= 0:
        return 0.0
    if item.scope == WORKSPACE_SCOPE:
        score += 0.6
    return score


def _apply_budget(
    memories: list[RetrievedMemory],
    budget: MemoryRetrievalBudget,
    diagnostics: dict[str, Any],
) -> list[RetrievedMemory]:
    selected: list[RetrievedMemory] = []
    counts = {WORKSPACE_SCOPE: 0, USER_SCOPE: 0}
    chars = {WORKSPACE_SCOPE: 0, USER_SCOPE: 0}
    max_items = {WORKSPACE_SCOPE: budget.max_workspace_items, USER_SCOPE: budget.max_user_items}
    max_chars = {WORKSPACE_SCOPE: budget.max_workspace_chars, USER_SCOPE: budget.max_user_chars}
    for memory in memories:
        scope = memory.scope
        if counts[scope] >= max_items[scope] or chars[scope] + memory.char_count > max_chars[scope]:
            diagnostics["skipped_over_budget"] += 1
            continue
        selected.append(memory)
        counts[scope] += 1
        chars[scope] += memory.char_count
    diagnostics["included_workspace"] = counts[WORKSPACE_SCOPE]
    diagnostics["included_user"] = counts[USER_SCOPE]
    diagnostics["included_workspace_chars"] = chars[WORKSPACE_SCOPE]
    diagnostics["included_user_chars"] = chars[USER_SCOPE]
    return selected


def _sort_memories(memories: list[RetrievedMemory], budget: MemoryRetrievalBudget) -> list[RetrievedMemory]:
    return sorted(
        memories,
        key=lambda memory: (
            -memory.score,
            0 if memory.scope == WORKSPACE_SCOPE else 1,
            -_timestamp(memory.updated_at),
            budget.priority_for(memory.category),
            memory.memory_id,
        ),
    )


def _tokenize(text: str) -> set[str]:
    ascii_tokens = {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(token) >= 2}
    han_chars = set(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    return ascii_tokens | han_chars


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _bounded(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _required_str(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ValueError(name)
    return value


def _empty_diagnostics() -> dict[str, Any]:
    return {field: 0 for field in DIAGNOSTIC_FIELDS}


def _remember_skip(diagnostics: dict[str, Any], field: str, value: str) -> None:
    values = diagnostics.setdefault(field, [])
    if isinstance(values, list) and len(values) < 10:
        values.append(value)
