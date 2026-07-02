"""
tests/unit/memory/test_memory_commit.py - 长期记忆确定性落库测试

验证用户确认候选后，MemoryStore 写入事实源、索引和审计记录。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.memory import CandidateEvidence, CandidateQueue, MemoryStore


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        source_type="episode",
        evidence_summary="成功 turn 中的用户确认结论。",
        session_id="session-test",
        turn_index=2,
        episode_path=".runs/episode-test",
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_confirm_candidate_writes_workspace_fact_index_and_audit(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = store.create_candidate(
        queue,
        scope="workspace",
        category="facts",
        title="Package manager",
        body="HaAgent uses uv for dependency management.",
        evidence=_evidence(),
        source="user_explicit",
        tags=["setup"],
    )

    record = store.confirm_candidate(queue, candidate.candidate_id, actor="user")

    memory_root = tmp_path / ".haagent" / "memory"
    facts = _jsonl(memory_root / "facts.jsonl")
    index = json.loads((memory_root / "index.json").read_text(encoding="utf-8"))
    audit_types = [event["event_type"] for event in _jsonl(memory_root / "audit.jsonl")]
    assert facts == [record.to_dict()]
    assert index["source"] == "workspace"
    assert index["items"][0]["id"] == record.memory_id
    assert index["items"][0]["summary"] == "HaAgent uses uv for dependency management."
    assert audit_types == ["candidate_created", "candidate_confirmed", "memory_committed", "index_rebuilt"]
    assert queue.get(candidate.candidate_id).status == "confirmed"


def test_confirm_workspace_categories_write_separate_files(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")

    for category in ["sop", "glossary", "decisions"]:
        candidate = store.create_candidate(
            queue,
            scope="workspace",
            category=category,
            title=f"{category} title",
            body=f"{category} body",
            evidence=_evidence(),
            source="user_explicit",
        )
        store.confirm_candidate(queue, candidate.candidate_id)

    memory_root = tmp_path / ".haagent" / "memory"
    assert _jsonl(memory_root / "sop.jsonl")[0]["category"] == "sop"
    assert _jsonl(memory_root / "glossary.jsonl")[0]["category"] == "glossary"
    assert _jsonl(memory_root / "decisions.jsonl")[0]["category"] == "decisions"


def test_confirm_user_memory_writes_to_user_root_only(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    user_root = tmp_path / "home" / ".haagent" / "memory"
    store = MemoryStore(workspace_root=tmp_path / "workspace", user_memory_root=user_root)
    candidate = store.create_candidate(
        queue,
        scope="user",
        category="user_preferences",
        title="Response language",
        body="The user prefers Simplified Chinese responses.",
        evidence=_evidence(),
        source="user_explicit",
        tags=["preference"],
    )

    record = store.confirm_candidate(queue, candidate.candidate_id)

    assert _jsonl(user_root / "user_preferences.jsonl") == [record.to_dict()]
    assert not (tmp_path / "workspace" / ".haagent" / "memory" / "user_preferences.jsonl").exists()
    assert json.loads((user_root / "index.json").read_text(encoding="utf-8"))["source"] == "user"


def test_confirm_uses_user_edited_candidate_content(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")
    store = MemoryStore(workspace_root=tmp_path, user_memory_root=tmp_path / "user-memory")
    candidate = store.create_candidate(
        queue,
        scope="workspace",
        category="facts",
        title="Old title",
        body="Old body.",
        evidence=_evidence(),
        source="agent_proposed",
        tags=["old"],
    )

    record = store.confirm_candidate(
        queue,
        candidate.candidate_id,
        edited_title="Edited title",
        edited_body="Edited body is final.",
        edited_tags=["edited"],
    )

    assert record.title == "Edited title"
    assert record.body == "Edited body is final."
    assert record.tags == ["edited"]
    assert _jsonl(tmp_path / ".haagent" / "memory" / "facts.jsonl")[0]["body"] == "Edited body is final."
