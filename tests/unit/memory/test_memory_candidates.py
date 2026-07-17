"""
tests/unit/memory/test_memory_candidates.py - 长期记忆候选队列测试

验证候选先进入 session 队列，并且未确认候选不会成为正式长期记忆。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.memory import CandidateEvidence, CandidateQueue, MemoryStore
from haagent.memory.intake import MemoryCandidateIntake, MemoryDraft


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        source_type="user_explicit",
        evidence_summary="用户明确确认这是可复用项目事实。",
        session_id="session-test",
        turn_index=1,
    )


def _audit_events(path: Path) -> list[dict[str, object]]:
    audit_path = path / "audit.jsonl"
    return [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]


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
    actor: str = "agent",
) -> object:
    result = MemoryCandidateIntake(store, queue).submit(
        MemoryDraft(
            scope=scope,
            category=category,
            title=title,
            body=body,
            evidence=evidence or _evidence(),
            source=source,
            tags=list(tags or []),
            actor=actor,
        ),
        reject_secrets=False,
    )
    assert result.accepted is True
    assert result.candidate is not None
    return result.candidate


def test_create_workspace_candidate_writes_pending_queue_and_audit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")

    candidate = _submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Default test command",
        body="Use uv run pytest tests/test_memory_*.py -q for memory changes.",
        source="user_explicit",
        tags=["testing"],
        actor="user",
    )

    assert candidate.status == "pending"
    assert candidate.source == "user_explicit"
    assert candidate.scope == "workspace"
    assert candidate.category == "facts"
    assert queue.get(candidate.candidate_id) == candidate
    assert queue.list(status="pending") == [candidate]
    assert _audit_events(tmp_path / ".haagent" / "memory")[0]["event_type"] == "candidate_created"


def test_candidate_queue_accepts_agent_and_runtime_sources(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")

    agent_candidate = _submit(
        store,
        queue,
        scope="workspace",
        category="sop",
        title="Memory review flow",
        body="Long-term memory candidates must be reviewed before commit.",
        source="agent_proposed",
    )
    runtime_candidate = _submit(
        store,
        queue,
        scope="workspace",
        category="decisions",
        title="Candidate queue is mandatory",
        body="Runtime-originated durable memory must also enter CandidateQueue first.",
        source="runtime",
    )

    assert [item.source for item in queue.list(status="pending")] == ["agent_proposed", "runtime"]
    assert agent_candidate.candidate_id != runtime_candidate.candidate_id


def test_reject_candidate_updates_queue_and_audit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = _submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Temporary observation",
        body="This was only useful for one turn.",
        source="agent_proposed",
    )

    rejected = store.reject_candidate(queue, candidate.candidate_id, reason="not durable", actor="user")

    assert rejected.status == "rejected"
    assert queue.get(candidate.candidate_id).status == "rejected"
    assert queue.list(status="pending") == []
    assert [event["event_type"] for event in _audit_events(tmp_path / ".haagent" / "memory")] == [
        "candidate_created",
        "memory_rejected",
    ]


def test_pending_candidate_is_not_listed_as_committed_memory(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    _submit(
        store,
        queue,
        scope="workspace",
        category="facts",
        title="Pending fact",
        body="Pending candidates are not durable memory.",
        source="runtime",
    )

    assert store.list_records(scope="workspace", category="facts") == []
    assert not (tmp_path / ".haagent" / "memory" / "facts.jsonl").exists()
