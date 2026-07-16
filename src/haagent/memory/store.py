"""
haagent/memory/store.py - 长期记忆确定性存储服务

协调 CandidateQueue、治理规则、事实源 JSONL、索引和审计日志。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from haagent.memory.audit import MemoryAuditLog
from haagent.memory.candidates import CandidateQueue
from haagent.memory.governance import (
    MemoryGovernanceError,
    check_duplicate_and_conflict,
    ensure_confirmable,
    memory_content_hash,
    memory_id_for,
    prepare_candidate_fields,
    redact_tombstone_reason,
    validate_candidate_source,
    validate_evidence,
    validate_scope_category,
)
from haagent.memory.schema import (
    MEMORY_SCHEMA_VERSION,
    MEMORY_CATEGORIES_BY_SCOPE,
    USER_SCOPE,
    WORKSPACE_SCOPE,
    CandidateEvidence,
    MemoryCandidate,
    MemoryIndex,
    MemoryIndexItem,
    MemoryRecord,
    MemoryTombstone,
)
from haagent.models.config.connections import user_config_dir


WORKSPACE_FILE_BY_CATEGORY = {
    "facts": "facts.jsonl",
    "sop": "sop.jsonl",
    "glossary": "glossary.jsonl",
    "decisions": "decisions.jsonl",
}
USER_FILE_BY_CATEGORY = {
    "user_preferences": "user_preferences.jsonl",
    "habits": "habits.jsonl",
    "constraints": "constraints.jsonl",
}
FILE_BY_SCOPE_CATEGORY = {
    WORKSPACE_SCOPE: WORKSPACE_FILE_BY_CATEGORY,
    USER_SCOPE: USER_FILE_BY_CATEGORY,
}


class MemoryStoreError(RuntimeError):
    """长期记忆存储文件损坏或操作目标不存在时抛出。"""


class MemoryStore:
    def __init__(self, *, workspace_root: Path, user_memory_root: Path | None = None) -> None:
        self.workspace_root = workspace_root.resolve()
        self.workspace_memory_root = self.workspace_root / ".haagent" / "memory"
        self.user_memory_root = (user_memory_root or user_config_dir() / "memory").resolve()

    def create_candidate(
        self,
        queue: CandidateQueue,
        *,
        scope: str,
        category: str,
        title: str,
        body: str,
        evidence: CandidateEvidence,
        source: str,
        tags: list[str] | None = None,
        actor: str = "agent",
    ) -> MemoryCandidate:
        validate_scope_category(scope, category)
        validate_candidate_source(source)
        validate_evidence(evidence)
        stored_title, stored_body, stored_evidence, stored_tags, risk_flags = prepare_candidate_fields(
            title,
            body,
            evidence,
            tags,
        )
        candidate = queue.create(
            scope=scope,
            category=category,
            title=stored_title,
            body=stored_body,
            evidence=stored_evidence,
            source=source,
            tags=stored_tags,
            risk_flags=risk_flags,
        )
        self._audit(scope).append(
            event_type="candidate_created",
            scope=scope,
            category=category,
            candidate_id=candidate.candidate_id,
            status_to="pending",
            actor=actor,
            summary=candidate.title,
        )
        return candidate

    def reject_candidate(
        self,
        queue: CandidateQueue,
        candidate_id: str,
        *,
        reason: str,
        actor: str = "user",
    ) -> MemoryCandidate:
        candidate = queue.get(candidate_id)
        rejected = queue.reject(candidate_id)
        self._audit(candidate.scope).append(
            event_type="memory_rejected",
            scope=candidate.scope,
            category=candidate.category,
            candidate_id=candidate.candidate_id,
            status_from="pending",
            status_to="rejected",
            reason=reason,
            actor=actor,
            summary=candidate.title,
        )
        return rejected

    def confirm_candidate(
        self,
        queue: CandidateQueue,
        candidate_id: str,
        *,
        edited_title: str | None = None,
        edited_body: str | None = None,
        edited_tags: list[str] | None = None,
        actor: str = "user",
    ) -> MemoryRecord:
        candidate = queue.get(candidate_id)
        title = edited_title if edited_title is not None else candidate.title
        body = edited_body if edited_body is not None else candidate.body
        tags = list(edited_tags if edited_tags is not None else candidate.tags)
        _, _, _, _, risk_flags = prepare_candidate_fields(title, body, candidate.evidence, tags)
        ensure_confirmable(candidate.evidence, risk_flags)
        validate_scope_category(candidate.scope, candidate.category)

        content_hash = memory_content_hash(title, body)
        now = _now_iso()
        record = MemoryRecord(
            memory_id=memory_id_for(candidate.scope, candidate.category, content_hash),
            scope=candidate.scope,
            category=candidate.category,
            title=title,
            body=body,
            evidence=candidate.evidence,
            source_candidate_id=candidate.candidate_id,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            tags=tags,
        )
        check_duplicate_and_conflict(record, self.list_records(scope=candidate.scope, category=candidate.category))
        self._append_record(record)
        queue.mark_confirmed(candidate_id, record.memory_id)
        audit = self._audit(candidate.scope)
        audit.append(
            event_type="candidate_confirmed",
            scope=candidate.scope,
            category=candidate.category,
            candidate_id=candidate.candidate_id,
            memory_id=record.memory_id,
            status_from="pending",
            status_to="confirmed",
            actor=actor,
            summary=record.title,
        )
        audit.append(
            event_type="memory_committed",
            scope=candidate.scope,
            category=candidate.category,
            candidate_id=candidate.candidate_id,
            memory_id=record.memory_id,
            status_to="active",
            actor=actor,
            summary=record.title,
        )
        self.rebuild_index(candidate.scope, actor=actor)
        return record

    def soft_delete(
        self,
        *,
        memory_id: str,
        scope: str,
        category: str,
        reason: str,
        actor: str = "user",
    ) -> MemoryTombstone:
        validate_scope_category(scope, category)
        records = self.list_records(scope=scope, category=category)
        if not any(record.memory_id == memory_id for record in records):
            raise MemoryStoreError(f"memory record not found: {memory_id}")
        safe_reason = redact_tombstone_reason(reason)
        tombstone = MemoryTombstone(
            memory_id=memory_id,
            scope=scope,
            category=category,
            reason=safe_reason,
            deleted_at=_now_iso(),
            actor=actor,
        )
        self._append_jsonl(self._root(scope) / "tombstones.jsonl", tombstone.to_dict())
        self._audit(scope).append(
            event_type="memory_soft_deleted",
            scope=scope,
            category=category,
            memory_id=memory_id,
            status_from="active",
            status_to="deleted",
            reason=safe_reason,
            actor=actor,
            summary=memory_id,
        )
        self.rebuild_index(scope, actor=actor)
        return tombstone

    def list_records(self, *, scope: str, category: str) -> list[MemoryRecord]:
        validate_scope_category(scope, category)
        path = self._category_path(scope, category)
        if not path.exists():
            return []
        records: list[MemoryRecord] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(MemoryRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise MemoryStoreError(f"invalid {path.name} line {line_number}") from error
        return records

    def rebuild_index(self, scope: str, *, actor: str = "system") -> MemoryIndex:
        if scope not in {WORKSPACE_SCOPE, USER_SCOPE}:
            raise MemoryGovernanceError(f"invalid memory scope: {scope}")
        deleted_ids = self._deleted_ids(scope)
        items: list[MemoryIndexItem] = []
        for category in sorted(MEMORY_CATEGORIES_BY_SCOPE[scope]):
            for record in self.list_records(scope=scope, category=category):
                status = "deleted" if record.memory_id in deleted_ids else "active"
                items.append(
                    MemoryIndexItem(
                        id=record.memory_id,
                        category=record.category,
                        title=record.title,
                        summary=_summary(record.body),
                        tags=list(record.tags),
                        updated_at=record.updated_at,
                        status=status,
                    ),
                )
        index = MemoryIndex(
            version=MEMORY_SCHEMA_VERSION,
            updated_at=_now_iso(),
            source=scope,
            items=items,
        )
        root = self._root(scope)
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.json").write_text(
            json.dumps(index.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._audit(scope).append(
            event_type="index_rebuilt",
            scope=scope,
            actor=actor,
            status_to="rebuilt",
            summary=f"{len(items)} items",
        )
        return index

    def _append_record(self, record: MemoryRecord) -> None:
        self._append_jsonl(self._category_path(record.scope, record.category), record.to_dict())

    def _append_jsonl(self, path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _deleted_ids(self, scope: str) -> set[str]:
        path = self._root(scope) / "tombstones.jsonl"
        if not path.exists():
            return set()
        deleted: set[str] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                deleted.add(MemoryTombstone.from_dict(json.loads(line)).memory_id)
            except (json.JSONDecodeError, ValueError) as error:
                raise MemoryStoreError(f"invalid tombstones.jsonl line {line_number}") from error
        return deleted

    def _category_path(self, scope: str, category: str) -> Path:
        validate_scope_category(scope, category)
        return self._root(scope) / FILE_BY_SCOPE_CATEGORY[scope][category]

    def _root(self, scope: str) -> Path:
        if scope == WORKSPACE_SCOPE:
            return self.workspace_memory_root
        if scope == USER_SCOPE:
            return self.user_memory_root
        raise MemoryGovernanceError(f"invalid memory scope: {scope}")

    def _audit(self, scope: str) -> MemoryAuditLog:
        return MemoryAuditLog(self._root(scope))


def _summary(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:160]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
