"""
haagent/memory/governance.py - 长期记忆治理边界

提供 evidence、secret、猜测、去重和冲突检测，阻止不安全内容进入事实源。
"""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

from haagent.memory.schema import (
    CANDIDATE_SOURCES,
    CANDIDATE_STATUSES,
    MEMORY_CATEGORIES_BY_SCOPE,
    MEMORY_SCOPES,
    CandidateEvidence,
    MemoryRecord,
)


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|cookie|secret)\b\s*[:=]\s*['\"]?[^'\"\s]{8,}"),
]


class MemoryGovernanceError(RuntimeError):
    """候选或长期记忆违反治理规则时抛出。"""


class MemoryDuplicateError(MemoryGovernanceError):
    """同一 scope/category 下内容 hash 完全重复时抛出。"""


class MemoryConflictError(MemoryGovernanceError):
    """同一 scope/category 下标题或主题近似冲突时抛出。"""


def validate_scope_category(scope: str, category: str) -> None:
    if scope not in MEMORY_SCOPES:
        raise MemoryGovernanceError(f"invalid memory scope: {scope}")
    if category not in MEMORY_CATEGORIES_BY_SCOPE[scope]:
        raise MemoryGovernanceError(f"invalid memory category for {scope}: {category}")


def validate_candidate_source(source: str) -> None:
    if source not in CANDIDATE_SOURCES:
        raise MemoryGovernanceError(f"invalid candidate source: {source}")


def validate_candidate_status(status: str) -> None:
    if status not in CANDIDATE_STATUSES:
        raise MemoryGovernanceError(f"invalid candidate status: {status}")


def validate_evidence(evidence: CandidateEvidence) -> None:
    if not evidence.source_type.strip() or not evidence.evidence_summary.strip():
        raise MemoryGovernanceError("candidate evidence is required")


def prepare_candidate_fields(
    title: str,
    body: str,
    evidence: CandidateEvidence,
    tags: list[str] | None,
) -> tuple[str, str, CandidateEvidence, list[str], list[str]]:
    risk_flags = scan_persisted_candidate_fields(title, body, evidence, tags)
    stored_title = title
    stored_body = body
    stored_evidence = evidence
    stored_tags = list(tags or [])
    if "possible_secret" in risk_flags:
        stored_title = redact_sensitive(title)
        stored_body = redact_sensitive(body)
        stored_evidence = redact_candidate_evidence(evidence)
        stored_tags = redact_tags(stored_tags)
    return stored_title, stored_body, stored_evidence, stored_tags, risk_flags


def prepare_candidate_text(title: str, body: str) -> tuple[str, str, list[str]]:
    risk_flags = text_risk_flags(title, body)
    if "possible_secret" in risk_flags:
        title = redact_sensitive(title)
        body = redact_sensitive(body)
    return title, body, risk_flags


def scan_persisted_candidate_fields(
    title: str,
    body: str,
    evidence: CandidateEvidence,
    tags: list[str] | None,
) -> list[str]:
    flags = text_risk_flags(title, body)
    # 被脱敏后的候选仍保留 unresolved secret 风险，直到用户可编辑字段被安全替换。
    persisted_text = "\n".join(
        [
            evidence.source_type,
            evidence.evidence_summary,
            evidence.session_id or "",
            evidence.episode_path or "",
            evidence.source_path or "",
            evidence.evidence_quote or "",
            evidence.fingerprint or "",
            *(tags or []),
        ],
    )
    if scan_secrets(persisted_text) and "possible_secret" not in flags:
        flags.append("possible_secret")
    return flags


def text_risk_flags(title: str, body: str) -> list[str]:
    text = f"{title}\n{body}"
    flags: list[str] = []
    if scan_secrets(text):
        flags.append("possible_secret")
    return flags


def scan_secrets(text: str) -> list[str]:
    hits: list[str] = []
    if "[REDACTED_SECRET]" in text:
        hits.append("redacted_secret_marker")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def redact_sensitive(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def redact_candidate_evidence(evidence: CandidateEvidence) -> CandidateEvidence:
    return CandidateEvidence(
        source_type=redact_sensitive(evidence.source_type),
        evidence_summary=redact_sensitive(evidence.evidence_summary),
        session_id=redact_sensitive(evidence.session_id) if evidence.session_id is not None else None,
        turn_index=evidence.turn_index,
        episode_path=redact_sensitive(evidence.episode_path) if evidence.episode_path is not None else None,
        source_path=redact_sensitive(evidence.source_path) if evidence.source_path is not None else None,
        source_summary=redact_sensitive(evidence.source_summary) if evidence.source_summary is not None else None,
        basis=redact_sensitive(evidence.basis) if evidence.basis is not None else None,
        category_rationale=(
            redact_sensitive(evidence.category_rationale)
            if evidence.category_rationale is not None
            else None
        ),
        evidence_quote=redact_sensitive(evidence.evidence_quote) if evidence.evidence_quote is not None else None,
        fingerprint=redact_sensitive(evidence.fingerprint) if evidence.fingerprint is not None else None,
    )


def redact_tags(tags: list[str] | None) -> list[str]:
    return [redact_sensitive(tag) for tag in tags or []]


def redact_tombstone_reason(reason: str) -> str:
    return redact_sensitive(reason)


def ensure_confirmable(evidence: CandidateEvidence, risk_flags: list[str]) -> None:
    validate_evidence(evidence)
    if risk_flags:
        raise MemoryGovernanceError(f"candidate has unresolved risk flags: {', '.join(risk_flags)}")


def memory_content_hash(title: str, body: str) -> str:
    normalized = normalize_memory_text(title) + "\n" + normalize_memory_text(body)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def memory_id_for(scope: str, category: str, content_hash: str) -> str:
    digest = hashlib.sha256(f"{scope}\n{category}\n{content_hash}".encode("utf-8")).hexdigest()
    return "mem_" + digest[:16]


def check_duplicate_and_conflict(candidate: MemoryRecord, existing: list[MemoryRecord]) -> None:
    for record in existing:
        if record.content_hash == candidate.content_hash:
            raise MemoryDuplicateError(f"duplicate memory content: {record.memory_id}")
    candidate_title = normalize_title(candidate.title)
    for record in existing:
        existing_title = normalize_title(record.title)
        if existing_title == candidate_title:
            raise MemoryConflictError(f"memory title conflicts with {record.memory_id}")
        if SequenceMatcher(None, existing_title, candidate_title).ratio() >= 0.86:
            raise MemoryConflictError(f"memory title conflicts with {record.memory_id}")


def normalize_memory_text(value: str) -> str:
    return " ".join(value.split()).strip().lower()


def normalize_title(value: str) -> str:
    text = normalize_memory_text(value)
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
