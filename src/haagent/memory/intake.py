"""
src/haagent/memory/intake.py - 未信任记忆 draft 的唯一治理入口

模型/用户/runtime draft 只经 MemoryCandidateIntake.submit；
规范化、secret、identity、去重与 queue/audit 集中在此，extraction 不再直写 store。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haagent.memory.candidates import CandidateQueue
from haagent.memory.governance import (
    MemoryGovernanceError,
    prepare_candidate_fields,
    prepare_candidate_text,
    scan_secrets,
    text_risk_flags,
    validate_candidate_source,
    validate_evidence,
    validate_scope_category,
)
from haagent.memory.identity import MemoryIdentity, compute_identity, similar_title
from haagent.memory.schema import CandidateEvidence, MemoryCandidate
from haagent.memory.store import MemoryStore


# 固定拒绝原因；diagnostics / 调用方依赖这些字符串。
REASON_MISSING_REQUIRED = "missing_required_fields"
REASON_INVALID_SCOPE = "invalid_scope_category"
REASON_INVALID_SOURCE = "invalid_candidate_source"
REASON_INVALID_EVIDENCE = "invalid_evidence"
REASON_BLOCKED_RISK = "blocked_risk"
REASON_POSSIBLE_SECRET = "possible_secret"
REASON_DUPLICATE = "duplicate_fingerprint_or_content"
REASON_GOVERNANCE = "governance_error"


@dataclass(frozen=True)
class MemoryDraft:
    """未信任候选草稿；任何来源必须先变成 draft 再进入 intake。"""

    scope: str
    category: str
    title: str
    body: str
    evidence: CandidateEvidence
    source: str
    tags: list[str] = field(default_factory=list)
    actor: str = "agent"


@dataclass(frozen=True)
class IntakeResult:
    accepted: bool
    reason: str | None = None
    candidate: MemoryCandidate | None = None
    identity: MemoryIdentity | None = None


class MemoryCandidateIntake:
    """单一治理 Seam：validate → identity → dedupe → queue + audit。"""

    def __init__(self, store: MemoryStore, queue: CandidateQueue) -> None:
        self._store = store
        self._queue = queue

    def submit(self, draft: MemoryDraft, *, reject_secrets: bool = True) -> IntakeResult:
        try:
            validate_scope_category(draft.scope, draft.category)
            validate_candidate_source(draft.source)
            validate_evidence(draft.evidence)
        except MemoryGovernanceError as error:
            reason = REASON_INVALID_SCOPE
            message = str(error)
            if "source" in message:
                reason = REASON_INVALID_SOURCE
            elif "evidence" in message:
                reason = REASON_INVALID_EVIDENCE
            return IntakeResult(accepted=False, reason=reason)

        if not draft.title.strip() or not draft.body.strip():
            return IntakeResult(accepted=False, reason=REASON_MISSING_REQUIRED)

        if reject_secrets and _draft_has_secret_or_risk(draft):
            # 自动来源默认硬拒绝，不落盘候选，避免 secret 进入 queue。
            return IntakeResult(accepted=False, reason=REASON_POSSIBLE_SECRET)

        identity = compute_identity(
            scope=draft.scope,
            category=draft.category,
            title=draft.title,
            body=draft.body,
            evidence=draft.evidence,
        )
        # 指纹写入 evidence，后续 confirm/list 可复用。
        evidence = draft.evidence
        if evidence.fingerprint is None and identity.evidence_fingerprint is not None:
            evidence = CandidateEvidence(
                source_type=evidence.source_type,
                evidence_summary=evidence.evidence_summary,
                session_id=evidence.session_id,
                turn_index=evidence.turn_index,
                episode_path=evidence.episode_path,
                source_path=evidence.source_path,
                source_summary=evidence.source_summary,
                basis=evidence.basis,
                category_rationale=evidence.category_rationale,
                evidence_quote=evidence.evidence_quote,
                fingerprint=identity.evidence_fingerprint,
            )

        if _is_duplicate_identity(identity, draft.scope, draft.category, self._queue, self._store):
            return IntakeResult(accepted=False, reason=REASON_DUPLICATE, identity=identity)

        try:
            candidate = self._store._persist_candidate(
                self._queue,
                scope=draft.scope,
                category=draft.category,
                title=draft.title,
                body=draft.body,
                evidence=evidence,
                source=draft.source,
                tags=list(draft.tags),
                actor=draft.actor,
            )
        except MemoryGovernanceError:
            return IntakeResult(accepted=False, reason=REASON_GOVERNANCE, identity=identity)

        return IntakeResult(accepted=True, candidate=candidate, identity=identity)


def _draft_has_secret_or_risk(draft: MemoryDraft) -> bool:
    text = "\n".join(
        [
            draft.title,
            draft.body,
            draft.evidence.source_type,
            draft.evidence.evidence_summary,
            draft.evidence.source_summary or "",
            draft.evidence.basis or "",
            draft.evidence.category_rationale or "",
            draft.evidence.evidence_quote or "",
            *draft.tags,
        ],
    )
    if scan_secrets(text):
        return True
    _, _, flags = prepare_candidate_text(draft.title, draft.body)
    if flags:
        return True
    return bool(text_risk_flags(draft.evidence.source_summary or "", draft.evidence.basis or ""))


def _is_duplicate_identity(
    identity: MemoryIdentity,
    scope: str,
    category: str,
    queue: CandidateQueue,
    store: MemoryStore,
) -> bool:
    for record in store.list_records(scope=scope, category=category):
        if _matches_identity(identity, record_identity=compute_identity(
            scope=record.scope,
            category=record.category,
            title=record.title,
            body=record.body,
            evidence=record.evidence,
        )):
            return True
    for pending in queue.list():
        if pending.scope != scope or pending.category != category:
            continue
        other = compute_identity(
            scope=pending.scope,
            category=pending.category,
            title=pending.title,
            body=pending.body,
            evidence=pending.evidence,
        )
        if _matches_identity(identity, record_identity=other):
            return True
    return False


def _matches_identity(left: MemoryIdentity, *, record_identity: MemoryIdentity) -> bool:
    if (
        left.evidence_fingerprint
        and record_identity.evidence_fingerprint
        and left.evidence_fingerprint == record_identity.evidence_fingerprint
    ):
        return True
    if left.content_hash == record_identity.content_hash:
        return True
    if left.body_key and left.body_key == record_identity.body_key:
        return True
    if similar_title(left.title_key, record_identity.title_key):
        return True
    return False
