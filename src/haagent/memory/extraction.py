"""
haagent/memory/extraction.py - 长期记忆候选提取

从已完成 chat turn 中提取可审查候选，只写入 CandidateQueue，不直接写长期事实源。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from haagent.memory.candidates import CandidateQueue
from haagent.memory.governance import (
    MemoryGovernanceError,
    memory_content_hash,
    normalize_title,
    prepare_candidate_text,
    redact_sensitive,
    scan_secrets,
    text_risk_flags,
    validate_scope_category,
)
from haagent.memory.schema import (
    CandidateEvidence,
    MemoryCandidate,
)
from haagent.memory.store import MemoryStore
from haagent.models.gateway import ModelCallError, ModelGateway


@dataclass(frozen=True)
class MemoryExtractionPolicy:
    max_candidates: int = 3
    max_prompt_chars: int = 900
    max_response_chars: int = 900
    max_working_state_chars: int = 1200
    max_runtime_events: int = 8
    max_title_chars: int = 100
    max_body_chars: int = 700
    max_evidence_chars: int = 500


@dataclass(frozen=True)
class ExtractedMemoryCandidate:
    scope: str
    category: str
    title: str
    body: str
    source_summary: str
    basis: str
    category_rationale: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryExtractionRequest:
    session_id: str
    session_path: Path
    workspace_root: Path
    turn_index: int
    user_prompt: str
    final_response: str
    status: str
    verification_status: str
    episode_path: Path
    working_state: dict[str, object] | None = None
    runtime_events: list[dict[str, object]] = field(default_factory=list)
    model_gateway: ModelGateway | None = None
    user_memory_root: Path | None = None
    policy: MemoryExtractionPolicy = field(default_factory=MemoryExtractionPolicy)


@dataclass(frozen=True)
class MemoryExtractionResult:
    status: str
    reason: str
    created_candidates: list[MemoryCandidate] = field(default_factory=list)
    rejected_count: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def created_count(self) -> int:
        return len(self.created_candidates)


class MemoryExtractor:
    def extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        """运行一次有界提取，并把结果写入 session diagnostics。"""
        try:
            result = self._extract(request)
        except Exception as error:
            result = MemoryExtractionResult(
                status="error",
                reason=type(error).__name__,
                diagnostics={"error": redact_sensitive(str(error))},
            )
        self._write_diagnostics(request, result)
        return result

    def _extract(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        diagnostics = _base_diagnostics(request)
        if request.status != "completed":
            return MemoryExtractionResult("skipped", "turn not completed", diagnostics=diagnostics)
        if request.model_gateway is None:
            return MemoryExtractionResult("skipped", "model gateway unavailable", diagnostics=diagnostics)

        queue = CandidateQueue(request.session_path)
        store = MemoryStore(workspace_root=request.workspace_root, user_memory_root=request.user_memory_root)
        extracted = _model_candidates(request)
        if not extracted:
            return MemoryExtractionResult("skipped", "no durable memories proposed", diagnostics=diagnostics)

        created: list[MemoryCandidate] = []
        rejected = 0
        for candidate in extracted[: request.policy.max_candidates]:
            if not _candidate_has_required_evidence(candidate):
                rejected += 1
                continue
            if _candidate_has_blocked_risk(candidate):
                rejected += 1
                continue
            try:
                validate_scope_category(candidate.scope, candidate.category)
            except MemoryGovernanceError:
                rejected += 1
                continue
            if _is_duplicate(candidate, queue, store):
                rejected += 1
                continue
            evidence = CandidateEvidence(
                source_type="extraction",
                evidence_summary=_bounded(
                    f"{candidate.source_summary} {candidate.basis}",
                    request.policy.max_evidence_chars,
                ),
                session_id=request.session_id,
                turn_index=request.turn_index,
                episode_path=str(request.episode_path),
                source_summary=_bounded(candidate.source_summary, request.policy.max_evidence_chars),
                basis=_bounded(candidate.basis, request.policy.max_evidence_chars),
                category_rationale=_bounded(candidate.category_rationale, request.policy.max_evidence_chars),
            )
            try:
                created.append(
                    store.create_candidate(
                        queue,
                        scope=candidate.scope,
                        category=candidate.category,
                        title=_bounded(candidate.title, request.policy.max_title_chars),
                        body=_bounded(candidate.body, request.policy.max_body_chars),
                        evidence=evidence,
                        source="extraction",
                        tags=candidate.tags,
                        actor="agent",
                    ),
                )
            except MemoryGovernanceError:
                rejected += 1

        status = "created" if created else "skipped"
        reason = "" if created else "all candidates rejected"
        return MemoryExtractionResult(status, reason, created, rejected, diagnostics=diagnostics)

    def _write_diagnostics(self, request: MemoryExtractionRequest, result: MemoryExtractionResult) -> None:
        request.session_path.mkdir(parents=True, exist_ok=True)
        record = {
            "created_at": datetime.now(UTC).isoformat(),
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "status": result.status,
            "reason": result.reason,
            "created_count": result.created_count,
            "rejected_count": result.rejected_count,
            **result.diagnostics,
        }
        with (request.session_path / "memory_extraction.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _model_candidates(request: MemoryExtractionRequest) -> list[ExtractedMemoryCandidate]:
    prompt = _build_model_prompt(request)
    try:
        response = request.model_gateway.generate(
            messages=[{"role": "user", "content": prompt}],
            tool_schemas=[],
        )
    except ModelCallError:
        return []
    if response.tool_calls:
        return []
    payload = _parse_json_object(response.content)
    raw_candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(raw_candidates, list):
        raw_candidates = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[ExtractedMemoryCandidate] = []
    for raw in raw_candidates[: request.policy.max_candidates]:
        if not isinstance(raw, dict):
            continue
        tags_raw = raw.get("tags", [])
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()] if isinstance(tags_raw, list) else []
        candidates.append(
            ExtractedMemoryCandidate(
                scope=str(raw.get("scope") or "").strip(),
                category=str(raw.get("category") or "").strip(),
                title=str(raw.get("title") or "").strip(),
                body=str(raw.get("body") or "").strip(),
                source_summary=str(raw.get("source_summary") or "").strip(),
                basis=str(raw.get("basis") or "").strip(),
                category_rationale=str(raw.get("category_rationale") or "").strip(),
                tags=tags,
            ),
        )
    return candidates


def _build_model_prompt(request: MemoryExtractionRequest) -> str:
    policy = request.policy
    working_state = json.dumps(request.working_state or {}, ensure_ascii=False, sort_keys=True)
    runtime = [
        _runtime_event_summary(event)
        for event in request.runtime_events[-policy.max_runtime_events :]
    ]
    return "\n".join(
        [
            "Memory Extraction: propose only durable long-term memory candidates.",
            "Return JSON only: {\"candidates\":[{\"scope\":\"workspace|user\",\"category\":\"facts|sop|glossary|decisions|user_preferences|habits|constraints\",\"title\":\"...\",\"body\":\"...\",\"source_summary\":\"...\",\"basis\":\"...\",\"category_rationale\":\"...\",\"tags\":[\"...\"]}]}",
            "Never include secrets, guesses, raw transcript, raw tool output, temporary debug state, or one-off task chatter.",
            f"session_id={request.session_id} turn_index={request.turn_index} verification={request.verification_status}",
            f"user_prompt={_bounded(request.user_prompt, policy.max_prompt_chars)}",
            f"assistant_final_response={_bounded(request.final_response, policy.max_response_chars)}",
            f"working_state={_bounded(working_state, policy.max_working_state_chars)}",
            "runtime_events:",
            *runtime,
        ],
    )


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        return {}
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _candidate_has_required_evidence(candidate: ExtractedMemoryCandidate) -> bool:
    return all(
        value.strip()
        for value in [
            candidate.title,
            candidate.body,
            candidate.source_summary,
            candidate.basis,
            candidate.category_rationale,
        ]
    )


def _candidate_has_blocked_risk(candidate: ExtractedMemoryCandidate) -> bool:
    text = "\n".join(
        [
            candidate.title,
            candidate.body,
            candidate.source_summary,
            candidate.basis,
            candidate.category_rationale,
            *candidate.tags,
        ],
    )
    if scan_secrets(text):
        return True
    _, _, flags = prepare_candidate_text(candidate.title, candidate.body)
    return bool(flags or text_risk_flags(candidate.source_summary, candidate.basis))


def _is_duplicate(candidate: ExtractedMemoryCandidate, queue: CandidateQueue, store: MemoryStore) -> bool:
    content_hash = memory_content_hash(candidate.title, candidate.body)
    candidate_title = normalize_title(candidate.title)
    candidate_body = " ".join(candidate.body.split()).strip().lower()
    for record in store.list_records(scope=candidate.scope, category=candidate.category):
        if record.content_hash == content_hash:
            return True
        if " ".join(record.body.split()).strip().lower() == candidate_body:
            return True
        if _similar_title(candidate_title, normalize_title(record.title)):
            return True
    for pending in queue.list(status="pending"):
        if pending.scope != candidate.scope or pending.category != candidate.category:
            continue
        if memory_content_hash(pending.title, pending.body) == content_hash:
            return True
        if " ".join(pending.body.split()).strip().lower() == candidate_body:
            return True
        if _similar_title(candidate_title, normalize_title(pending.title)):
            return True
    return False


def _similar_title(left: str, right: str) -> bool:
    return bool(left and right and (left == right or SequenceMatcher(None, left, right).ratio() >= 0.86))


def _runtime_event_summary(event: dict[str, object]) -> str:
    event_type = str(event.get("event_type", "unknown"))
    tool_name = str(event.get("tool_name", ""))
    return _bounded(f"- event={event_type} tool={tool_name}", 160)


def _base_diagnostics(request: MemoryExtractionRequest) -> dict[str, object]:
    return {
        "source_chars": len(request.user_prompt) + len(request.final_response),
        "verification_status": request.verification_status,
        "episode_path": str(request.episode_path),
    }


def _bounded(value: str, limit: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"
