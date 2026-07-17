"""
src/haagent/memory/identity.py - 记忆身份与去重比较

统一 content_hash / memory_id / evidence fingerprint / 标题相似度，
供 intake 与 confirm 共用，避免提取与确认规则漂移。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from haagent.memory.governance import memory_content_hash, memory_id_for, normalize_title
from haagent.memory.schema import CandidateEvidence


@dataclass(frozen=True)
class MemoryIdentity:
    content_hash: str
    memory_id: str
    title_key: str
    body_key: str
    evidence_fingerprint: str | None = None


def body_key(body: str) -> str:
    return " ".join(body.split()).strip().lower()


def compute_identity(
    *,
    scope: str,
    category: str,
    title: str,
    body: str,
    evidence: CandidateEvidence | None = None,
) -> MemoryIdentity:
    content = memory_content_hash(title, body)
    fingerprint = None
    if evidence is not None:
        fingerprint = evidence.fingerprint or fingerprint_from_parts(
            category=category,
            body=body,
            evidence_source=evidence.source_type,
            evidence_quote=evidence.evidence_quote or "",
        )
    return MemoryIdentity(
        content_hash=content,
        memory_id=memory_id_for(scope, category, content),
        title_key=normalize_title(title),
        body_key=body_key(body),
        evidence_fingerprint=fingerprint,
    )


def fingerprint_from_parts(
    *,
    category: str,
    body: str,
    evidence_source: str,
    evidence_quote: str,
) -> str:
    parts = [category, body, evidence_source, evidence_quote]
    normalized = "\n".join(_normalize_evidence_text(part) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def similar_title(left: str, right: str) -> bool:
    return bool(left and right and (left == right or SequenceMatcher(None, left, right).ratio() >= 0.86))


def _normalize_evidence_text(value: str) -> str:
    lowered = value.lower()
    return "".join(re.findall(r"[a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", lowered))
