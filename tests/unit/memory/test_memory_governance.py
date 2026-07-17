"""
tests/unit/memory/test_memory_governance.py - 长期记忆治理规则测试

验证 secret、evidence、猜测、去重、冲突和软删除边界。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from haagent.memory import (
    CandidateEvidence,
    CandidateQueue,
    MemoryConflictError,
    MemoryGovernanceError,
    MemoryStore,
)
from haagent.memory.intake import MemoryCandidateIntake, MemoryDraft


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        source_type="episode",
        evidence_summary="用户确认的稳定事实。",
        session_id="session-test",
        turn_index=3,
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _all_text_under(path: Path) -> str:
    chunks: list[str] = []
    for child in sorted(path.rglob("*")):
        if child.is_file():
            chunks.append(child.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _submit(
    store: MemoryStore,
    queue: CandidateQueue,
    *,
    scope: str,
    category: str,
    title: str,
    body: str,
    source: str,
    tags: list[str] | None = None,
    evidence: CandidateEvidence | None = None,
    reject_secrets: bool = False,
):
    result = MemoryCandidateIntake(store, queue).submit(
        MemoryDraft(
            scope=scope,
            category=category,
            title=title,
            body=body,
            evidence=evidence or _evidence(),
            source=source,
            tags=list(tags or []),
            actor="user",
        ),
        reject_secrets=reject_secrets,
    )
    return result


def _must_submit(*args, **kwargs):
    result = _submit(*args, **kwargs)
    assert result.accepted is True
    assert result.candidate is not None
    return result.candidate


def test_secret_candidate_is_redacted_and_cannot_commit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    # 用户显式路径可选择入队后 redact（reject_secrets=False）；confirm 仍被 risk_flags 拦住。
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="API key",
        body=f"OpenAI key is {secret}",
        source="user_explicit",
        reject_secrets=False,
    )

    raw_queue = (queue.path).read_text(encoding="utf-8")
    raw_audit = (tmp_path / ".haagent" / "memory" / "audit.jsonl").read_text(encoding="utf-8")
    assert "possible_secret" in candidate.risk_flags
    assert secret not in raw_queue
    assert secret not in raw_audit
    with pytest.raises(MemoryGovernanceError, match="possible_secret"):
        store.confirm_candidate(queue, candidate.candidate_id)


def test_evidence_secret_is_redacted_and_blocks_confirm(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    secret = "token=abcdefghijklmnopqrstuvwxyz123456"

    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Evidence secret",
        body="The candidate body is safe.",
        evidence=CandidateEvidence(
            source_type="episode",
            evidence_summary=f"User pasted {secret}",
            session_id="session-test",
            turn_index=4,
            episode_path=".runs/episode-test",
        ),
        source="runtime",
        reject_secrets=False,
    )

    assert "possible_secret" in candidate.risk_flags
    assert secret not in queue.path.read_text(encoding="utf-8")
    with pytest.raises(MemoryGovernanceError, match="possible_secret"):
        store.confirm_candidate(
            queue,
            candidate.candidate_id,
            edited_title="Safe title",
            edited_body="Safe body.",
            edited_tags=["safe"],
        )


def test_tag_secret_is_redacted_blocks_confirm_and_can_be_edited(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    memory_root = tmp_path / ".haagent" / "memory"
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    secret = "cookie=abcdefghijklmnopqrstuvwxyz123456"
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Safe title",
        body="Safe body.",
        source="user_explicit",
        tags=["safe", secret],
        reject_secrets=False,
    )

    assert "possible_secret" in candidate.risk_flags
    assert secret not in queue.path.read_text(encoding="utf-8")
    with pytest.raises(MemoryGovernanceError, match="possible_secret"):
        store.confirm_candidate(queue, candidate.candidate_id)

    record = store.confirm_candidate(queue, candidate.candidate_id, edited_tags=["safe"])

    assert record.tags == ["safe"]
    assert secret not in _all_text_under(memory_root)


def test_candidate_requires_evidence(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")

    result = _submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="No evidence",
        body="This cannot be committed.",
        evidence=CandidateEvidence(source_type="episode", evidence_summary=""),
        source="runtime",
    )
    assert result.accepted is False
    assert result.reason == "invalid_evidence"
    assert queue.list() == []


def test_uncertain_words_are_not_phrase_flagged_or_blocked(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Maybe package manager",
        body="HaAgent 可能 uses uv.",
        source="agent_proposed",
    )

    assert "unverified_claim" not in candidate.risk_flags
    record = store.confirm_candidate(queue, candidate.candidate_id)
    assert record.body == "HaAgent 可能 uses uv."


def test_title_body_secret_can_commit_after_safe_user_edit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    secret = "password=abcdefghijklmnopqrstuvwxyz123456"
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Credential note",
        body=f"The password was {secret}.",
        source="user_explicit",
        tags=["credential-note"],
        reject_secrets=False,
    )

    record = store.confirm_candidate(
        queue,
        candidate.candidate_id,
        edited_title="Credential policy",
        edited_body="Do not store credentials in memory.",
        edited_tags=["security"],
    )

    assert record.title == "Credential policy"
    assert secret not in _all_text_under(tmp_path / ".haagent" / "memory")


def test_duplicate_content_hash_is_not_written_twice(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    first = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Quality gate",
        body="Run uv run pytest tests/test_memory_*.py -q.",
        source="user_explicit",
    )
    # Intake 在入队前去重：相同内容第二次不得再写 pending。
    second = _submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Quality gate",
        body="Run uv run pytest tests/test_memory_*.py -q.",
        source="runtime",
    )
    assert second.accepted is False
    assert second.reason == "duplicate_fingerprint_or_content"
    store.confirm_candidate(queue, first.candidate_id)
    assert len(_jsonl(tmp_path / ".haagent" / "memory" / "facts.jsonl")) == 1


def test_similar_title_conflict_is_explicit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    first = _must_submit(
        store,
        queue,
        scope="workspace",
        category="decisions",
        title="Long term memory write path",
        body="Candidates must be confirmed before durable commit.",
        source="user_explicit",
    )
    store.confirm_candidate(queue, first.candidate_id)

    # 相似标题：Intake 抑制再入队；confirm 路径仍保留 MemoryConflictError 语义（见下）。
    second = _submit(
        store,
        queue,
        scope="workspace",
        category="decisions",
        title="Long-term memory write path",
        body="Candidates require user review before durable commit.",
        source="runtime",
    )
    assert second.accepted is False
    assert second.reason == "duplicate_fingerprint_or_content"
    assert len(_jsonl(tmp_path / ".haagent" / "memory" / "decisions.jsonl")) == 1


def test_confirm_still_raises_conflict_for_queued_similar_titles(tmp_path: Path) -> None:
    """两条已 pending 候选若身份不同但 confirm 时标题冲突，仍应 MemoryConflictError。"""
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    first = _must_submit(
        store,
        queue,
        scope="workspace",
        category="decisions",
        title="Alpha decision path",
        body="First durable decision body.",
        source="user_explicit",
    )
    # 不同 body/title 足以绕过 intake 指纹/hash；confirm 时用编辑制造相似标题冲突。
    second = _must_submit(
        store,
        queue,
        scope="workspace",
        category="decisions",
        title="Beta decision path",
        body="Second durable decision body.",
        source="runtime",
    )
    store.confirm_candidate(queue, first.candidate_id)
    with pytest.raises(MemoryConflictError):
        store.confirm_candidate(
            queue,
            second.candidate_id,
            edited_title="Alpha decision path",
            edited_body="Second durable decision body.",
        )


def test_soft_delete_writes_tombstone_and_keeps_source_record(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Soft delete target",
        body="Soft delete keeps source JSONL records.",
        source="user_explicit",
    )
    record = store.confirm_candidate(queue, candidate.candidate_id)

    tombstone = store.soft_delete(
        memory_id=record.memory_id,
        scope="workspace",
        category="facts",
        reason="obsolete",
        actor="user",
    )

    memory_root = tmp_path / ".haagent" / "memory"
    index = json.loads((memory_root / "index.json").read_text(encoding="utf-8"))
    audit_types = [event["event_type"] for event in _jsonl(memory_root / "audit.jsonl")]
    assert tombstone.memory_id == record.memory_id
    assert _jsonl(memory_root / "facts.jsonl")[0]["memory_id"] == record.memory_id
    assert _jsonl(memory_root / "tombstones.jsonl")[0]["memory_id"] == record.memory_id
    assert index["items"][0]["status"] == "deleted"
    assert "memory_soft_deleted" in audit_types
    assert audit_types[-1] == "index_rebuilt"


def test_soft_delete_redacts_secret_reason_in_tombstone_and_audit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = _must_submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Delete reason target",
        body="Soft delete reason must be redacted.",
        source="user_explicit",
    )
    record = store.confirm_candidate(queue, candidate.candidate_id)
    secret = "api_key=abcdefghijklmnopqrstuvwxyz123456"

    tombstone = store.soft_delete(
        memory_id=record.memory_id,
        scope="workspace",
        category="facts",
        reason=f"Remove because {secret}",
        actor="user",
    )

    memory_root = tmp_path / ".haagent" / "memory"
    assert secret not in (memory_root / "tombstones.jsonl").read_text(encoding="utf-8")
    assert secret not in (memory_root / "audit.jsonl").read_text(encoding="utf-8")
    assert tombstone.reason == "Remove because [REDACTED_SECRET]"
