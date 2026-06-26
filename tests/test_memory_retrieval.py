"""
tests/test_memory_retrieval.py - 长期记忆检索测试

验证 Memory Retrieval 只读取已确认事实源，并以有界形式接入 ContextBuilder。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.memory import CandidateEvidence, CandidateQueue, MemoryStore
from haagent.memory.retrieval import (
    MemoryRetrievalBudget,
    MemoryRetrievalRequest,
    MemoryRetriever,
)
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.task_contract import TaskSpec


def _evidence() -> CandidateEvidence:
    return CandidateEvidence(
        source_type="episode",
        evidence_summary="用户确认过的稳定结论。",
        session_id="session-test",
        turn_index=1,
        episode_path=".runs/episode-test",
    )


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(workspace_root=tmp_path / "workspace", user_memory_root=tmp_path / "user-memory")


def _queue(tmp_path: Path) -> CandidateQueue:
    return CandidateQueue(tmp_path / ".runs" / "sessions" / "session-test")


def _commit(
    tmp_path: Path,
    *,
    scope: str,
    category: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> str:
    store = _store(tmp_path)
    candidate = store.create_candidate(
        _queue(tmp_path),
        scope=scope,
        category=category,
        title=title,
        body=body,
        evidence=_evidence(),
        source="user_explicit",
        tags=tags,
    )
    return store.confirm_candidate(_queue(tmp_path), candidate.candidate_id).memory_id


def _retrieve(
    tmp_path: Path,
    query: str,
    *,
    budget: MemoryRetrievalBudget | None = None,
) -> object:
    return MemoryRetriever().retrieve(
        MemoryRetrievalRequest(
            query=query,
            workspace_root=tmp_path / "workspace",
            user_memory_root=tmp_path / "user-memory",
            budget=budget or MemoryRetrievalBudget(),
        ),
    )


def test_retrieval_reads_only_confirmed_active_memory_not_pending_candidates(tmp_path: Path) -> None:
    pending = _store(tmp_path).create_candidate(
        _queue(tmp_path),
        scope="workspace",
        category="facts",
        title="Pending pytest note",
        body="Pending candidates must never enter retrieval.",
        evidence=_evidence(),
        source="user_explicit",
    )
    committed_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Pytest command",
        body="Use uv run pytest for HaAgent tests.",
        tags=["pytest"],
    )

    result = _retrieve(tmp_path, "pytest command")

    assert [item.memory_id for item in result.memories] == [committed_id]
    assert pending.title not in result.to_model_block()


def test_retrieval_hydrates_source_record_instead_of_trusting_index_summary(tmp_path: Path) -> None:
    memory_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Hydrated title",
        body="The real source body must be used.",
        tags=["source"],
    )
    index_path = tmp_path / "workspace" / ".haagent" / "memory" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"][0]["summary"] = "Poisoned index summary."
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    result = _retrieve(tmp_path, "source body")

    assert result.memories[0].memory_id == memory_id
    assert result.memories[0].body == "The real source body must be used."
    assert "Poisoned index summary." not in result.to_model_block()


def test_tombstoned_and_missing_records_are_skipped_with_diagnostics(tmp_path: Path) -> None:
    deleted_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Deleted fact",
        body="This fact should be tombstoned.",
    )
    kept_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Kept fact",
        body="This fact should remain available.",
    )
    _store(tmp_path).soft_delete(
        memory_id=deleted_id,
        scope="workspace",
        category="facts",
        reason="obsolete",
    )
    index_path = tmp_path / "workspace" / ".haagent" / "memory" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["items"].append(
        {
            "id": "mem_missing",
            "category": "facts",
            "title": "Missing fact",
            "summary": "missing",
            "tags": ["missing"],
            "updated_at": "2026-06-25T00:00:00+00:00",
            "status": "active",
        },
    )
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    result = _retrieve(tmp_path, "fact missing obsolete available")

    assert [item.memory_id for item in result.memories] == [kept_id]
    assert result.diagnostics["skipped_deleted"] >= 1
    assert result.diagnostics["skipped_missing"] == 1


def test_workspace_memory_sorts_before_user_memory_for_equal_relevance(tmp_path: Path) -> None:
    workspace_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Response language",
        body="Workspace language rule mentions concise Chinese.",
        tags=["language"],
    )
    user_id = _commit(
        tmp_path,
        scope="user",
        category="user_preferences",
        title="Response language",
        body="User usually likes concise Chinese.",
        tags=["language"],
    )

    result = _retrieve(tmp_path, "response language concise Chinese")

    assert [item.memory_id for item in result.memories[:2]] == [workspace_id, user_id]


def test_user_memory_is_marked_lower_priority_than_current_task(tmp_path: Path) -> None:
    _commit(
        tmp_path,
        scope="user",
        category="user_preferences",
        title="Language preference",
        body="The user usually prefers English responses.",
        tags=["language"],
    )

    result = _retrieve(tmp_path, "当前任务明确要求使用中文回答 language")
    block = result.to_model_block()

    assert "Current turn, project instructions, session summary, and working_state override these memories." in block
    assert "The user usually prefers English responses." in block


def test_budget_limits_item_count_and_characters(tmp_path: Path) -> None:
    titles = ["Pytest budget alpha", "Runtime cache banana", "Context package zebra"]
    for index, title in enumerate(titles):
        _commit(
            tmp_path,
            scope="workspace",
            category="facts",
            title=title,
            body=f"pytest topic {index} " + ("long body " * 20),
            tags=["pytest"],
        )

    result = _retrieve(
        tmp_path,
        "pytest budget",
        budget=MemoryRetrievalBudget(max_workspace_items=1, max_workspace_chars=35, max_item_chars=35),
    )

    assert len(result.memories) == 1
    assert result.memories[0].char_count <= 35
    assert result.diagnostics["skipped_over_budget"] >= 1


def test_workspace_categories_are_retrievable(tmp_path: Path) -> None:
    ids = {
        category: _commit(
            tmp_path,
            scope="workspace",
            category=category,
            title=f"{category} retrieval",
            body=f"{category} retrieval marker",
            tags=[category],
        )
        for category in ["facts", "sop", "glossary", "decisions"]
    }

    result = _retrieve(tmp_path, "facts sop glossary decisions retrieval marker")

    assert set(ids.values()) <= {item.memory_id for item in result.memories}


def test_retrieval_does_not_recall_memory_from_task_kind_without_query_overlap(tmp_path: Path) -> None:
    _commit(
        tmp_path,
        scope="workspace",
        category="sop",
        title="Release checklist",
        body="Run the release smoke suite before publishing.",
        tags=["release"],
    )

    result = _retrieve(tmp_path, "implement unrelated calendar parser")

    assert result.memories == []


def test_empty_memory_files_return_explicit_empty_result(tmp_path: Path) -> None:
    result = _retrieve(tmp_path, "anything")

    assert result.memories == []
    assert result.diagnostics["workspace_index_missing"] == 1
    assert result.diagnostics["user_index_missing"] == 1
    assert result.to_model_block() == ""


def test_retrieval_does_not_inject_full_store_audit_tombstone_or_trace(tmp_path: Path) -> None:
    memory_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Trace boundary",
        body="Short useful memory.",
        tags=["trace"],
    )
    _store(tmp_path).soft_delete(
        memory_id=memory_id,
        scope="workspace",
        category="facts",
        reason="create tombstone and audit",
    )
    kept_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Compact memory boundary",
        body="Use compact memory only.",
        tags=["trace"],
    )

    result = _retrieve(tmp_path, "trace compact")
    block = result.to_model_block()

    assert kept_id in block
    assert "audit.jsonl" not in block
    assert "tombstones.jsonl" not in block
    assert "memory_candidates.jsonl" not in block
    assert "tool-calls.jsonl" not in block
    assert "transcript.jsonl" not in block


def test_context_builder_injects_compact_memory_and_manifest_audit(tmp_path: Path) -> None:
    memory_id = _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Context memory",
        body="ContextBuilder should include this compact memory.",
        tags=["context"],
    )
    writer = _make_writer(tmp_path)
    writer.write_plan({"planned_steps": ["Use context memory."]})

    context = ContextBuilder(
        task=_task("Use context memory"),
        workspace_root=tmp_path / "workspace",
        provider_name="fake",
        episode_writer=writer,
        working_state={"current_goal": "context", "key_findings": [], "completed_actions": [], "next_steps": [], "last_updated_turn": 1},
    ).build()

    assert "Relevant Memory:" in context.model_input
    assert memory_id in context.model_input
    manifest = json.loads((writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8"))
    assert manifest["memory"]["used_memories"][0]["id"] == memory_id
    assert manifest["memory"]["budget"]["max_workspace_items"] == 6
    assert "diagnostics" in manifest["memory"]


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
    return EpisodeWriter.create(tmp_path / ".runs", task_path)


def _task(goal: str) -> TaskSpec:
    return TaskSpec(
        goal=goal,
        workspace_root=".",
        allowed_tools=["file_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )
