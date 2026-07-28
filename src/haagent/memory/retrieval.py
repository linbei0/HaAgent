"""
haagent/memory/retrieval.py - 长期记忆检索

从 workspace/user memory 索引进入，hydrate 已确认事实源，并返回有界可审计结果。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
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

# BM25 参数（Elasticsearch/Lucene 工业默认值）
_BM25_K1 = 1.2
_BM25_B = 0.75

# 时间衰减：90 天半衰期，下限 0.1 防止老记忆完全消失
_DECAY_HALF_LIFE_DAYS = 90.0
_DECAY_FLOOR = 0.1

# BM25 最低相关阈值，低于此值的候选不入选。
# 注意：小语料库（<50 文档）中 IDF 绝对值天然较低，阈值不宜过高。
_MIN_RELEVANCE_SCORE = 0.1

# 字段乘性加权（title 命中权重最高，body 最低）
_FIELD_BOOSTS: dict[str, float] = {
    "title": 2.0,
    "tags": 1.8,
    "category": 1.8,
    "summary": 1.4,
    "body": 1.0,
}


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
    hit_reasons: list[dict[str, Any]] = field(default_factory=list)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "id": self.memory_id,
            "scope": self.scope,
            "category": self.category,
            "title": self.title,
            "updated_at": self.updated_at,
            "score": round(self.score, 3),
            "char_count": self.char_count,
            "hit_reasons": self.hit_reasons,
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


@dataclass(frozen=True)
class _BM25Index:
    """BM25 倒排索引，在 _cached_scope_sources 中构建并随缓存复用。"""

    # token -> {memory_id: term_frequency}
    tf_by_token: dict[str, dict[str, int]]
    # token -> document_frequency（包含该 token 的文档数）
    df: dict[str, int]
    # memory_id -> 文档总 token 数（所有字段合并）
    doc_lengths: dict[str, int]
    # 平均文档长度
    avg_doc_length: float
    # 文档总数
    corpus_size: int
    # token -> {memory_id: 最大字段 boost}（取该 token 在文档中命中字段的最高权重）
    field_boosts: dict[str, dict[str, float]]


@dataclass(frozen=True)
class _ScopeSources:
    items: tuple[_IndexItem, ...]
    deleted_ids: frozenset[str]
    records: dict[str, MemoryRecord]
    bm25_index: _BM25Index
    normalized_bodies: dict[str, str]
    timestamps: dict[str, float]
    diagnostics: dict[str, Any]


# (index_item, record, final_score, body, timestamp, hit_reasons)
_ScoredMemory = tuple[_IndexItem, MemoryRecord, float, str, float, list[dict[str, Any]]]


_SOURCE_CACHE_LIMIT = 8
_SOURCE_CACHE_LOCK = RLock()
_SOURCE_CACHE: OrderedDict[
    tuple[str, Path],
    tuple[tuple[tuple[str, int, int] | tuple[str, None, None], ...], _ScopeSources],
] = OrderedDict()


class MemoryRetriever:
    def retrieve(self, request: MemoryRetrievalRequest) -> MemoryRetrievalResult:
        query_tokens = _tokenize_text(f"{request.query}\n{request.task_context}")
        diagnostics = _empty_diagnostics()
        candidates: list[_ScoredMemory] = []
        for scope, root in [
            (WORKSPACE_SCOPE, request.workspace_root.resolve() / ".haagent" / "memory"),
            (USER_SCOPE, (request.user_memory_root or user_config_dir() / "memory").resolve()),
        ]:
            candidates.extend(self._read_scope(scope, root, query_tokens, request.budget, diagnostics))

        selected = _apply_budget(_sort_scored_memories(candidates, request.budget), request.budget, diagnostics)
        return MemoryRetrievalResult(memories=selected, budget=request.budget, diagnostics=diagnostics)

    def _read_scope(
        self,
        scope: str,
        root: Path,
        query_tokens: list[str],
        budget: MemoryRetrievalBudget,
        diagnostics: dict[str, Any],
    ) -> list[_ScoredMemory]:
        index_path = root / "index.json"
        if not index_path.exists():
            diagnostics[f"{scope}_index_missing"] += 1
            return []
        sources = _cached_scope_sources(root, scope)
        _merge_diagnostics(diagnostics, sources.diagnostics)
        memories: list[_ScoredMemory] = []
        for item in sources.items:
            if item.status != "active":
                diagnostics["skipped_deleted" if item.status == "deleted" else "skipped_inactive"] += 1
                continue
            if item.memory_id in sources.deleted_ids:
                diagnostics["skipped_deleted"] += 1
                continue
            record = sources.records.get(item.memory_id)
            if record is None:
                diagnostics["skipped_missing"] += 1
                _remember_skip(diagnostics, "missing_ids", item.memory_id)
                continue
            if record.status != "active":
                diagnostics["skipped_deleted" if record.status == "deleted" else "skipped_inactive"] += 1
                continue
            # BM25 评分 + 时间衰减
            bm25_score, hit_reasons = _score_bm25(query_tokens, sources.bm25_index, item.memory_id)
            if bm25_score < _MIN_RELEVANCE_SCORE:
                continue
            decay = _time_decay_factor(item.updated_at)
            final_score = bm25_score * decay
            # workspace 记忆固定加成，保持当前目录上下文优先
            if item.scope == WORKSPACE_SCOPE:
                final_score += 0.6
            body = sources.normalized_bodies[item.memory_id][: budget.max_item_chars]
            memories.append((item, record, final_score, body, sources.timestamps[item.memory_id], hit_reasons))
        return memories


def _cached_scope_sources(root: Path, scope: str) -> _ScopeSources:
    fingerprint = _scope_fingerprint(root, scope)
    cache_key = (scope, root)
    with _SOURCE_CACHE_LOCK:
        cached = _SOURCE_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            _SOURCE_CACHE.move_to_end(cache_key)
            return cached[1]

        # 缓存只复用解析结果；文件指纹变化后仍完整 hydrate 正文、墓碑和诊断。
        source_diagnostics = _empty_diagnostics()
        items = tuple(_load_index_items(root / "index.json", scope, source_diagnostics))
        records = _load_records_by_id(root, scope, source_diagnostics)
        bm25_index = _build_bm25_index(items, records)
        sources = _ScopeSources(
            items=items,
            deleted_ids=frozenset(_deleted_ids(root, source_diagnostics)),
            records=records,
            bm25_index=bm25_index,
            normalized_bodies={memory_id: " ".join(record.body.split()) for memory_id, record in records.items()},
            timestamps={memory_id: _timestamp(record.updated_at) for memory_id, record in records.items()},
            diagnostics=source_diagnostics,
        )
        _SOURCE_CACHE[cache_key] = (fingerprint, sources)
        _SOURCE_CACHE.move_to_end(cache_key)
        while len(_SOURCE_CACHE) > _SOURCE_CACHE_LIMIT:
            _SOURCE_CACHE.popitem(last=False)
        return sources


def _scope_fingerprint(
    root: Path,
    scope: str,
) -> tuple[tuple[str, int, int] | tuple[str, None, None], ...]:
    paths = [
        root / "index.json",
        root / "tombstones.jsonl",
        *(root / file_name for file_name in FILE_BY_SCOPE_CATEGORY[scope].values()),
    ]
    signatures: list[tuple[str, int, int] | tuple[str, None, None]] = []
    for path in paths:
        try:
            stat = path.stat()
        except FileNotFoundError:
            signatures.append((str(path), None, None))
        else:
            signatures.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signatures)


def _merge_diagnostics(target: dict[str, Any], cached: dict[str, Any]) -> None:
    for field, value in cached.items():
        if isinstance(value, int):
            target[field] = int(target.get(field, 0)) + value
        elif isinstance(value, list):
            for item in value:
                _remember_skip(target, field, str(item))


def _build_bm25_index(
    items: tuple[_IndexItem, ...],
    records: dict[str, MemoryRecord],
) -> _BM25Index:
    """构建 BM25 倒排索引：统计 TF、DF、文档长度和字段 boost。"""
    tf_by_token: dict[str, dict[str, int]] = {}
    field_boosts: dict[str, dict[str, float]] = {}
    doc_lengths: dict[str, int] = {}

    for item in items:
        record = records.get(item.memory_id)
        if record is None:
            continue
        # 各字段独立分词，合并为文档级 token 列表（保留 TF）
        field_texts: list[tuple[str, str]] = [
            ("title", item.title),
            ("tags", " ".join(item.tags) if item.tags else ""),
            ("category", item.category),
            ("summary", item.summary),
            ("body", record.body),
        ]
        doc_tokens: list[str] = []
        for field_name, text in field_texts:
            if not text:
                continue
            tokens = _tokenize_text(text)
            doc_tokens.extend(tokens)
            # 记录每个 token 在该文档中的最高字段 boost
            boost = _FIELD_BOOSTS[field_name]
            for token in set(tokens):
                boosts = field_boosts.setdefault(token, {})
                boosts[item.memory_id] = max(boosts.get(item.memory_id, 0.0), boost)

        # 统计文档级 TF（所有字段合并）
        token_counts = Counter(doc_tokens)
        doc_lengths[item.memory_id] = len(doc_tokens)
        for token, count in token_counts.items():
            tf_by_token.setdefault(token, {})[item.memory_id] = count

    # 计算 DF：每个 token 出现在多少文档中
    df: dict[str, int] = {token: len(doc_ids) for token, doc_ids in tf_by_token.items()}
    corpus_size = len(doc_lengths)
    avg_doc_length = sum(doc_lengths.values()) / max(corpus_size, 1)

    return _BM25Index(
        tf_by_token=tf_by_token,
        df=df,
        doc_lengths=doc_lengths,
        avg_doc_length=avg_doc_length,
        corpus_size=corpus_size,
        field_boosts=field_boosts,
    )


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


def _apply_budget(
    memories: list[_ScoredMemory],
    budget: MemoryRetrievalBudget,
    diagnostics: dict[str, Any],
) -> list[RetrievedMemory]:
    selected: list[RetrievedMemory] = []
    counts = {WORKSPACE_SCOPE: 0, USER_SCOPE: 0}
    chars = {WORKSPACE_SCOPE: 0, USER_SCOPE: 0}
    max_items = {WORKSPACE_SCOPE: budget.max_workspace_items, USER_SCOPE: budget.max_user_items}
    max_chars = {WORKSPACE_SCOPE: budget.max_workspace_chars, USER_SCOPE: budget.max_user_chars}
    for item, record, score, body, _timestamp_value, hit_reasons in memories:
        scope = item.scope
        char_count = len(body)
        if counts[scope] >= max_items[scope] or chars[scope] + char_count > max_chars[scope]:
            diagnostics["skipped_over_budget"] += 1
            continue
        selected.append(
            RetrievedMemory(
                memory_id=record.memory_id,
                scope=record.scope,
                category=record.category,
                title=record.title,
                body=body,
                tags=list(record.tags),
                updated_at=record.updated_at,
                score=score,
                char_count=char_count,
                hit_reasons=hit_reasons,
            ),
        )
        counts[scope] += 1
        chars[scope] += char_count
    diagnostics["included_workspace"] = counts[WORKSPACE_SCOPE]
    diagnostics["included_user"] = counts[USER_SCOPE]
    diagnostics["included_workspace_chars"] = chars[WORKSPACE_SCOPE]
    diagnostics["included_user_chars"] = chars[USER_SCOPE]
    return selected


def _sort_scored_memories(memories: list[_ScoredMemory], budget: MemoryRetrievalBudget) -> list[_ScoredMemory]:
    return sorted(
        memories,
        key=lambda memory: (
            -memory[2],
            0 if memory[0].scope == WORKSPACE_SCOPE else 1,
            -memory[4],
            budget.priority_for(memory[0].category),
            memory[0].memory_id,
        ),
    )


def _tokenize_text(text: str) -> list[str]:
    """中文重叠 bigram + ASCII token，返回有序列表保留 TF 信息。

    依据: Chen 1997 TREC 研究证明 bigram 索引在中文 IR 中优于词典分词，
    且零依赖、确定性、可测试。
    """
    tokens: list[str] = []
    # ASCII: 连续字母数字下划线，长度 >= 2
    tokens.extend(t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) >= 2)
    # 中文: 提取连续汉字段，每段生成重叠 bigram；单字段保留原字
    for segment in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", text):
        if len(segment) == 1:
            tokens.append(segment)
        else:
            for i in range(len(segment) - 1):
                tokens.append(segment[i : i + 2])
    return tokens


def _score_bm25(
    query_tokens: list[str],
    index: _BM25Index,
    memory_id: str,
) -> tuple[float, list[dict[str, Any]]]:
    """BM25 评分（Lucene 非负 IDF 变体），返回 (score, hit_reasons)。"""
    score = 0.0
    hit_reasons: list[dict[str, Any]] = []
    doc_len = index.doc_lengths.get(memory_id, 0)
    if doc_len == 0 or index.corpus_size == 0:
        return 0.0, []
    avg_dl = max(index.avg_doc_length, 1.0)
    # 查询 token 去重，避免同一 token 重复计分
    for token in set(query_tokens):
        tf = index.tf_by_token.get(token, {}).get(memory_id, 0)
        if tf == 0:
            continue
        df = index.df.get(token, 0)
        # Lucene 非负 IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        idf = math.log(1.0 + (index.corpus_size - df + 0.5) / (df + 0.5))
        # TF 饱和 + 文档长度归一化
        tf_saturated = (tf * (_BM25_K1 + 1.0)) / (
            tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * doc_len / avg_dl)
        )
        field_boost = index.field_boosts.get(token, {}).get(memory_id, 1.0)
        contribution = idf * tf_saturated * field_boost
        score += contribution
        hit_reasons.append({"token": token, "tf": tf, "idf": round(idf, 3), "boost": field_boost})
    return score, hit_reasons


def _time_decay_factor(updated_at: str) -> float:
    """指数衰减，90 天半衰期，下限 0.1 防止老记忆完全消失。

    依据: Dakera 生产系统 (half-life decay engine), ACT-R 认知架构,
    Mem0 2026 基准报告 temporal reasoning +29.6 分。
    """
    ts = _timestamp(updated_at)
    if ts <= 0:
        return 1.0  # 无法解析时间时不惩罚
    age_days = max(0.0, (datetime.now(timezone.utc).timestamp() - ts) / 86400.0)
    return max(_DECAY_FLOOR, math.exp(-math.log(2) * age_days / _DECAY_HALF_LIFE_DAYS))


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


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
