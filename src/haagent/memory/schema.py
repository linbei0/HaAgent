"""
haagent/memory/schema.py - 长期记忆数据结构

定义 Memory System v1 第一阶段使用的候选、事实源、索引、审计和墓碑 schema。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MEMORY_SCHEMA_VERSION = "1.0"
WORKSPACE_SCOPE = "workspace"
USER_SCOPE = "user"
MEMORY_SCOPES = {WORKSPACE_SCOPE, USER_SCOPE}
WORKSPACE_CATEGORIES = {"facts", "sop", "glossary", "decisions"}
USER_CATEGORIES = {"user_preferences", "habits", "constraints"}
MEMORY_CATEGORIES_BY_SCOPE = {
    WORKSPACE_SCOPE: WORKSPACE_CATEGORIES,
    USER_SCOPE: USER_CATEGORIES,
}
CANDIDATE_SOURCES = {"user_explicit", "agent_proposed", "runtime", "extraction", "rule_engine"}
CANDIDATE_STATUSES = {"pending", "confirmed", "rejected"}
MEMORY_STATUSES = {"active", "deleted"}
AUDIT_EVENT_TYPES = {
    "candidate_created",
    "candidate_confirmed",
    "memory_committed",
    "memory_rejected",
    "memory_soft_deleted",
    "index_rebuilt",
}


@dataclass(frozen=True)
class CandidateEvidence:
    source_type: str
    evidence_summary: str
    session_id: str | None = None
    turn_index: int | None = None
    episode_path: str | None = None
    source_path: str | None = None
    source_summary: str | None = None
    basis: str | None = None
    category_rationale: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_type": self.source_type,
            "evidence_summary": self.evidence_summary,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "episode_path": self.episode_path,
            "source_path": self.source_path,
            "source_summary": self.source_summary,
            "basis": self.basis,
            "category_rationale": self.category_rationale,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "CandidateEvidence":
        if not isinstance(raw, dict):
            raise ValueError("evidence must be an object")
        return cls(
            source_type=_required_str(raw, "source_type"),
            evidence_summary=_required_str(raw, "evidence_summary"),
            session_id=_optional_str(raw, "session_id"),
            turn_index=_optional_int(raw, "turn_index"),
            episode_path=_optional_str(raw, "episode_path"),
            source_path=_optional_str(raw, "source_path"),
            source_summary=_optional_str(raw, "source_summary"),
            basis=_optional_str(raw, "basis"),
            category_rationale=_optional_str(raw, "category_rationale"),
        )


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    scope: str
    category: str
    title: str
    body: str
    evidence: CandidateEvidence
    source: str
    status: str
    created_at: str
    updated_at: str
    tags: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    committed_memory_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "scope": self.scope,
            "category": self.category,
            "title": self.title,
            "body": self.body,
            "evidence": self.evidence.to_dict(),
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "risk_flags": list(self.risk_flags),
            "committed_memory_id": self.committed_memory_id,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryCandidate":
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        return cls(
            candidate_id=_required_str(raw, "candidate_id"),
            scope=_required_str(raw, "scope"),
            category=_required_str(raw, "category"),
            title=_required_str(raw, "title"),
            body=_required_str(raw, "body"),
            evidence=CandidateEvidence.from_dict(raw.get("evidence")),
            source=_required_str(raw, "source"),
            status=_required_str(raw, "status"),
            created_at=_required_str(raw, "created_at"),
            updated_at=_required_str(raw, "updated_at"),
            tags=_str_list(raw, "tags"),
            risk_flags=_str_list(raw, "risk_flags"),
            committed_memory_id=_optional_str(raw, "committed_memory_id"),
        )


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    scope: str
    category: str
    title: str
    body: str
    evidence: CandidateEvidence
    source_candidate_id: str
    content_hash: str
    created_at: str
    updated_at: str
    tags: list[str] = field(default_factory=list)
    status: str = "active"

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope,
            "category": self.category,
            "title": self.title,
            "body": self.body,
            "evidence": self.evidence.to_dict(),
            "source_candidate_id": self.source_candidate_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": list(self.tags),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryRecord":
        if not isinstance(raw, dict):
            raise ValueError("memory record must be an object")
        return cls(
            memory_id=_required_str(raw, "memory_id"),
            scope=_required_str(raw, "scope"),
            category=_required_str(raw, "category"),
            title=_required_str(raw, "title"),
            body=_required_str(raw, "body"),
            evidence=CandidateEvidence.from_dict(raw.get("evidence")),
            source_candidate_id=_required_str(raw, "source_candidate_id"),
            content_hash=_required_str(raw, "content_hash"),
            created_at=_required_str(raw, "created_at"),
            updated_at=_required_str(raw, "updated_at"),
            tags=_str_list(raw, "tags"),
            status=_required_str(raw, "status"),
        )


@dataclass(frozen=True)
class MemoryIndexItem:
    id: str
    category: str
    title: str
    summary: str
    tags: list[str]
    updated_at: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "tags": list(self.tags),
            "updated_at": self.updated_at,
            "status": self.status,
        }


@dataclass(frozen=True)
class MemoryIndex:
    version: str
    updated_at: str
    source: str
    items: list[MemoryIndexItem]

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "source": self.source,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class MemoryAuditEvent:
    event_id: str
    event_type: str
    created_at: str
    actor: str
    scope: str
    category: str | None = None
    candidate_id: str | None = None
    memory_id: str | None = None
    status_from: str | None = None
    status_to: str | None = None
    reason: str | None = None
    summary: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "created_at": self.created_at,
            "actor": self.actor,
            "scope": self.scope,
            "category": self.category,
            "candidate_id": self.candidate_id,
            "memory_id": self.memory_id,
            "status_from": self.status_from,
            "status_to": self.status_to,
            "reason": self.reason,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class MemoryTombstone:
    memory_id: str
    scope: str
    category: str
    reason: str
    deleted_at: str
    actor: str

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "scope": self.scope,
            "category": self.category,
            "reason": self.reason,
            "deleted_at": self.deleted_at,
            "actor": self.actor,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "MemoryTombstone":
        if not isinstance(raw, dict):
            raise ValueError("tombstone must be an object")
        return cls(
            memory_id=_required_str(raw, "memory_id"),
            scope=_required_str(raw, "scope"),
            category=_required_str(raw, "category"),
            reason=_required_str(raw, "reason"),
            deleted_at=_required_str(raw, "deleted_at"),
            actor=_required_str(raw, "actor"),
        )


def _required_str(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_str(raw: dict[str, Any], name: str) -> str | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_int(raw: dict[str, Any], name: str) -> int | None:
    value = raw.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _str_list(raw: dict[str, Any], name: str) -> list[str]:
    value = raw.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return list(value)
