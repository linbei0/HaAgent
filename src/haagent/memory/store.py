"""
haagent/memory/store.py - 长期记忆确定性存储服务

协调 CandidateQueue、治理规则、事实源 JSONL、索引和审计日志。
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from haagent.memory.audit import MemoryAuditLog
from haagent.memory.candidates import CandidateQueue
from haagent.memory.governance import (
    MemoryGovernanceError,
    check_duplicate_and_conflict,
    ensure_confirmable,
    prepare_candidate_fields,
    redact_tombstone_reason,
    validate_candidate_source,
    validate_evidence,
    validate_scope_category,
)
from haagent.memory.identity import compute_identity
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

    def _persist_candidate(
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
        """仅供 MemoryCandidateIntake 在治理通过后落盘；禁止外部直调。"""
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

        # confirm 复用同一 MemoryIdentity 模块，与 intake 身份规则一致。
        identity = compute_identity(
            scope=candidate.scope,
            category=candidate.category,
            title=title,
            body=body,
            evidence=candidate.evidence,
        )
        now = _now_iso()
        record = MemoryRecord(
            memory_id=identity.memory_id,
            scope=candidate.scope,
            category=candidate.category,
            title=title,
            body=body,
            evidence=candidate.evidence,
            source_candidate_id=candidate.candidate_id,
            content_hash=identity.content_hash,
            created_at=now,
            updated_at=now,
            tags=tags,
        )
        existing_records = self.list_records(scope=candidate.scope, category=candidate.category)
        check_duplicate_and_conflict(record, existing_records)
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
        self._upsert_index_record(record, existing_records=existing_records, actor=actor)
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
        existing_deleted_ids = self._deleted_ids(scope)
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
        self._mark_index_deleted(scope, memory_id, existing_deleted_ids=existing_deleted_ids, actor=actor)
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
        return self._persist_index(scope, items, actor=actor)

    def _upsert_index_record(
        self,
        record: MemoryRecord,
        *,
        existing_records: list[MemoryRecord],
        actor: str,
    ) -> MemoryIndex:
        items = self._read_existing_index(record.scope)
        expected_ids = {existing.memory_id for existing in existing_records}
        indexed_ids = {item.id for item in items or [] if item.category == record.category}
        if items is None or indexed_ids != expected_ids or self._other_sources_newer_than_index(
            record.scope,
            ignored_category=record.category,
        ):
            # 缺失索引可能对应已有事实源，必须全量重建，不能静默丢失旧记忆。
            return self.rebuild_index(record.scope, actor=actor)
        replacement = MemoryIndexItem(
            id=record.memory_id,
            category=record.category,
            title=record.title,
            summary=_summary(record.body),
            tags=list(record.tags),
            updated_at=record.updated_at,
            status="active",
        )
        items = [item for item in items if item.id != record.memory_id]
        items.append(replacement)
        # stable sort 保持同分类 JSONL 写入顺序，与全量 rebuild 的结果一致。
        items.sort(key=lambda item: item.category)
        return self._persist_index(record.scope, items, actor=actor)

    def _mark_index_deleted(
        self,
        scope: str,
        memory_id: str,
        *,
        existing_deleted_ids: set[str],
        actor: str,
    ) -> MemoryIndex:
        items = self._read_existing_index(scope)
        indexed_deleted_ids = {item.id for item in items or [] if item.status == "deleted"}
        if (
            items is None
            or not any(item.id == memory_id for item in items)
            or indexed_deleted_ids != existing_deleted_ids
            or self._other_sources_newer_than_index(scope, ignore_tombstones=True)
        ):
            return self.rebuild_index(scope, actor=actor)
        updated = [replace(item, status="deleted") if item.id == memory_id else item for item in items]
        return self._persist_index(scope, updated, actor=actor)

    def _read_existing_index(self, scope: str) -> list[MemoryIndexItem] | None:
        path = self._root(scope) / "index.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(raw, dict)
                or raw.get("version") != MEMORY_SCHEMA_VERSION
                or raw.get("source") != scope
                or not isinstance(raw.get("items"), list)
            ):
                raise ValueError("invalid index root")
            return [_memory_index_item_from_dict(item, scope=scope) for item in raw["items"]]
        except (OSError, json.JSONDecodeError, ValueError) as error:
            # 索引损坏必须显式失败；自动重建会掩盖持久化错误和治理证据问题。
            raise MemoryStoreError(f"invalid memory index: {path}") from error

    def _other_sources_newer_than_index(
        self,
        scope: str,
        *,
        ignored_category: str | None = None,
        ignore_tombstones: bool = False,
    ) -> bool:
        index_path = self._root(scope) / "index.json"
        try:
            index_mtime = index_path.stat().st_mtime_ns
        except FileNotFoundError:
            return True
        for category, file_name in FILE_BY_SCOPE_CATEGORY[scope].items():
            if category == ignored_category:
                continue
            path = self._root(scope) / file_name
            try:
                if path.stat().st_mtime_ns > index_mtime:
                    return True
            except FileNotFoundError:
                continue
        if not ignore_tombstones:
            try:
                if (self._root(scope) / "tombstones.jsonl").stat().st_mtime_ns > index_mtime:
                    return True
            except FileNotFoundError:
                pass
        return False

    def _persist_index(self, scope: str, items: list[MemoryIndexItem], *, actor: str) -> MemoryIndex:
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


def _memory_index_item_from_dict(raw: object, *, scope: str) -> MemoryIndexItem:
    if not isinstance(raw, dict):
        raise ValueError("index item must be an object")
    required = ("id", "category", "title", "summary", "updated_at", "status")
    if any(not isinstance(raw.get(field), str) for field in required):
        raise ValueError("index item fields must be strings")
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("index item tags must be strings")
    if raw["category"] not in MEMORY_CATEGORIES_BY_SCOPE[scope]:
        raise ValueError("index item category does not match scope")
    if raw["status"] not in {"active", "deleted"}:
        raise ValueError("index item status is invalid")
    return MemoryIndexItem(
        id=raw["id"],
        category=raw["category"],
        title=raw["title"],
        summary=raw["summary"],
        tags=list(tags),
        updated_at=raw["updated_at"],
        status=raw["status"],
    )


def _summary(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:160]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
