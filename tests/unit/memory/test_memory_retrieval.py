"""
tests/unit/memory/test_memory_retrieval.py - 长期记忆检索测试

验证 Memory Retrieval 只读取已确认事实源，并以有界形式接入 ContextBuilder。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.memory import CandidateEvidence, CandidateQueue, MemoryStore
from haagent.memory.intake import MemoryCandidateIntake, MemoryDraft
from haagent.memory.retrieval import (
    MemoryRetrievalBudget,
    MemoryRetrievalRequest,
    MemoryRetriever,
)
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.contracts.task import TaskSpec


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


def _submit_candidate(
    tmp_path: Path,
    *,
    scope: str,
    category: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
):
    store = _store(tmp_path)
    queue = _queue(tmp_path)
    result = MemoryCandidateIntake(store, queue).submit(
        MemoryDraft(
            scope=scope,
            category=category,
            title=title,
            body=body,
            evidence=_evidence(),
            source="user_explicit",
            tags=list(tags or []),
            actor="user",
        ),
        reject_secrets=False,
    )
    assert result.accepted is True
    assert result.candidate is not None
    return store, queue, result.candidate


def _commit(
    tmp_path: Path,
    *,
    scope: str,
    category: str,
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> str:
    store, queue, candidate = _submit_candidate(
        tmp_path,
        scope=scope,
        category=category,
        title=title,
        body=body,
        tags=tags,
    )
    return store.confirm_candidate(queue, candidate.candidate_id).memory_id


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
    _store_ref, _queue_ref, pending = _submit_candidate(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Pending pytest note",
        body="Pending candidates must never enter retrieval.",
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


def test_retrieval_reuses_parsed_sources_until_files_change(tmp_path: Path, monkeypatch) -> None:
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Cached pytest fact",
        body="Use pytest for the cached retrieval check.",
        tags=["pytest"],
    )
    # 冻结时间，避免 time_decay_factor 在两次调用间产生微小分数差异
    from datetime import datetime, timezone
    import haagent.memory.retrieval as retrieval_mod

    frozen_now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen_now

    monkeypatch.setattr(retrieval_mod, "datetime", _FrozenDatetime)

    memory_root = tmp_path / "workspace" / ".haagent" / "memory"
    original_read_text = Path.read_text
    source_reads: list[Path] = []

    def counting_read_text(path: Path, *args, **kwargs):
        if path.parent == memory_root and path.name != "audit.jsonl":
            source_reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)
    retriever = MemoryRetriever()
    request = MemoryRetrievalRequest(
        query="pytest cached retrieval",
        workspace_root=tmp_path / "workspace",
        user_memory_root=tmp_path / "user-memory",
    )

    first = retriever.retrieve(request)
    reads_after_first = len(source_reads)
    second = retriever.retrieve(request)

    assert first.memories == second.memories
    assert reads_after_first > 0
    assert len(source_reads) == reads_after_first

    facts_path = memory_root / "facts.jsonl"
    facts_path.write_text(original_read_text(facts_path, encoding="utf-8") + "\n", encoding="utf-8")
    retriever.retrieve(request)

    assert len(source_reads) > reads_after_first


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
        working_state={"key_findings": [], "last_updated_turn": 1},
    ).build()

    assert "Relevant Memory:" in context.model_input
    assert memory_id in context.model_input
    manifest = json.loads((writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8"))
    assert manifest["memory"]["used_memories"][0]["id"] == memory_id
    assert manifest["memory"]["budget"]["max_workspace_items"] == 6
    assert "diagnostics" in manifest["memory"]
    memory_source = manifest["source_diagnostics"]["memory"]
    assert memory_source["used_count"] == 1
    assert memory_source["skipped_over_budget"] == manifest["memory"]["diagnostics"]["skipped_over_budget"]
    assert memory_source["budget"] == manifest["memory"]["budget"]
    assert memory_source["included_in_model_input"] is True


def test_context_builder_injects_memory_index_and_relevant_memory_together(tmp_path: Path) -> None:
    sop_id = _commit(
        tmp_path,
        scope="workspace",
        category="sop",
        title="Release SOP",
        body="Use the release checklist before publishing.",
        tags=["release"],
    )
    fact_id = _commit(
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
    ).build()

    model_input = context.model_input
    assert "Memory/SOP Navigation Index:" in model_input
    assert f"scope=workspace category=sop id={sop_id} title=Release SOP" in model_input
    assert "Relevant Memory:" in model_input
    assert fact_id in model_input
    manifest = json.loads((writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8"))
    selected_sources = {item["source_type"] for item in manifest["selection"]["selected"]}
    assert {"memory_index", "memory"} <= selected_sources


def test_context_builder_records_missing_memory_index_as_skipped(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    writer.write_plan({"planned_steps": ["Answer directly."]})

    context = ContextBuilder(
        task=_task("Answer without memory"),
        workspace_root=tmp_path / "workspace",
        provider_name="fake",
        episode_writer=writer,
    ).build()

    assert "Memory/SOP Navigation Index:" not in context.model_input
    manifest = json.loads((writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8"))
    memory_index = [item for item in manifest["selection"]["skipped"] if item["source_type"] == "memory_index"]
    assert memory_index
    assert memory_index[0]["skip_reason"] == "missing_index"


def test_context_builder_records_empty_memory_index_as_skipped(tmp_path: Path) -> None:
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Temporary fact",
        body="This memory will be deleted.",
    )
    store = _store(tmp_path)
    records = store.list_records(scope="workspace", category="facts")
    store.soft_delete(memory_id=records[0].memory_id, scope="workspace", category="facts", reason="obsolete")
    writer = _make_writer(tmp_path)
    writer.write_plan({"planned_steps": ["Answer directly."]})

    context = ContextBuilder(
        task=_task("Answer without active memory"),
        workspace_root=tmp_path / "workspace",
        provider_name="fake",
        episode_writer=writer,
    ).build()

    assert "Memory/SOP Navigation Index:" not in context.model_input
    manifest = json.loads((writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8"))
    memory_index = [item for item in manifest["selection"]["skipped"] if item["source_type"] == "memory_index"]
    assert memory_index
    assert memory_index[0]["skip_reason"] == "empty"


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


# --- BM25 升级新增测试 ---


def test_bigram_tokenization() -> None:
    """验证中文重叠 bigram 切分：'项目配置' -> ['项目', '目配', '配置']。"""
    from haagent.memory.retrieval import _tokenize_text

    tokens = _tokenize_text("项目配置")
    assert "项目" in tokens
    assert "目配" in tokens
    assert "配置" in tokens
    # 不应产生单字 token（段长度 > 1）
    assert "项" not in tokens
    assert "目" not in tokens


def test_bigram_tokenization_single_char() -> None:
    """单个汉字段保留原字。"""
    from haagent.memory.retrieval import _tokenize_text

    tokens = _tokenize_text("我")
    assert "我" in tokens


def test_bigram_tokenization_ascii() -> None:
    """ASCII token 保持完整，长度 >= 2。"""
    from haagent.memory.retrieval import _tokenize_text

    tokens = _tokenize_text("HaAgent memory_system")
    assert "haagent" in tokens
    assert "memory_system" in tokens
    # 单字符 ASCII 不入选
    tokens2 = _tokenize_text("a b cd")
    assert "a" not in tokens2
    assert "b" not in tokens2
    assert "cd" in tokens2


def test_bm25_idf_weighting(tmp_path: Path) -> None:
    """稀有 token 命中得分应高于高频 token。"""
    # 创建 5 条记忆，4 条包含 "pytest"，只有 1 条包含 "kubernetes"
    entries = [
        ("Coverage flags", "Run pytest with coverage flags enabled.", ["pytest"]),
        ("Slow markers", "Configure pytest markers for slow tests.", ["pytest"]),
        ("Fixture debugging", "Debug pytest fixtures with breakpoints.", ["pytest"]),
        ("Parallel execution", "Parallel pytest execution via xdist plugin.", ["pytest"]),
        ("Cluster orchestration", "Deploy with kubernetes cluster orchestration.", ["kubernetes"]),
    ]
    for title, body, tags in entries:
        _commit(
            tmp_path,
            scope="workspace",
            category="facts",
            title=title,
            body=body,
            tags=tags,
        )

    # 查询同时包含两个 token，kubernetes 的 IDF 更高
    result = _retrieve(tmp_path, "pytest kubernetes")
    assert len(result.memories) >= 1
    # kubernetes 记忆应排第一（IDF 更高 + title 命中）
    assert result.memories[0].title == "Cluster orchestration"


def test_time_decay_recent_vs_old(tmp_path: Path, monkeypatch) -> None:
    """相同匹配分下，近期记忆排序优先于老旧记忆。"""
    from datetime import datetime, timezone
    import haagent.memory.retrieval as retrieval_mod

    # 冻结当前时间为 2026-07-28
    frozen_now = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen_now

    monkeypatch.setattr(retrieval_mod, "datetime", _FrozenDatetime)

    # 直接写入两条记忆，手动控制 updated_at
    store = _store(tmp_path)
    queue = _queue(tmp_path)
    intake = MemoryCandidateIntake(store, queue)

    # 近期记忆（1 天前）
    draft_recent = MemoryDraft(
        scope="workspace",
        category="facts",
        title="Recent deploy config",
        body="Deploy uses docker compose.",
        evidence=_evidence(),
        source="user_explicit",
        actor="user",
        tags=["deploy"],
    )
    result_recent = intake.submit(draft_recent, reject_secrets=False)
    assert result_recent.accepted and result_recent.candidate
    store.confirm_candidate(queue, result_recent.candidate.candidate_id)

    # 老旧记忆（200 天前）—— 手动修改 updated_at
    draft_old = MemoryDraft(
        scope="workspace",
        category="facts",
        title="Old deploy config",
        body="Deploy uses docker compose legacy.",
        evidence=_evidence(),
        source="user_explicit",
        actor="user",
        tags=["deploy"],
    )
    result_old = intake.submit(draft_old, reject_secrets=False)
    assert result_old.accepted and result_old.candidate
    store.confirm_candidate(queue, result_old.candidate.candidate_id)

    # 手动篡改旧记忆的 updated_at 为 200 天前
    import json as _json

    facts_path = tmp_path / "workspace" / ".haagent" / "memory" / "facts.jsonl"
    lines = facts_path.read_text(encoding="utf-8").splitlines()
    modified = []
    for line in lines:
        if not line.strip():
            continue
        record = _json.loads(line)
        if record.get("title") == "Old deploy config":
            record["updated_at"] = "2025-12-10T12:00:00+00:00"  # ~230 天前
        modified.append(_json.dumps(record, ensure_ascii=False))
    facts_path.write_text("\n".join(modified) + "\n", encoding="utf-8")

    # 同时修改 index.json 中的 updated_at
    index_path = tmp_path / "workspace" / ".haagent" / "memory" / "index.json"
    index_data = _json.loads(index_path.read_text(encoding="utf-8"))
    for item in index_data["items"]:
        if item.get("title") == "Old deploy config":
            item["updated_at"] = "2025-12-10T12:00:00+00:00"
    index_path.write_text(_json.dumps(index_data, ensure_ascii=False), encoding="utf-8")

    result = _retrieve(tmp_path, "deploy docker compose")
    assert len(result.memories) == 2
    # 近期记忆应排第一
    assert result.memories[0].title == "Recent deploy config"
    assert result.memories[1].title == "Old deploy config"
    # 近期分数 > 老旧分数
    assert result.memories[0].score > result.memories[1].score


def test_time_decay_floor() -> None:
    """极老记忆衰减不低于 0.1 下限。"""
    from haagent.memory.retrieval import _time_decay_factor

    # 1000 天前的记忆
    factor = _time_decay_factor("2023-10-01T00:00:00+00:00")
    assert factor == 0.1  # 触底


def test_time_decay_unparseable() -> None:
    """无法解析的时间戳不惩罚。"""
    from haagent.memory.retrieval import _time_decay_factor

    assert _time_decay_factor("invalid") == 1.0
    assert _time_decay_factor("") == 1.0


def test_hit_reasons_structure(tmp_path: Path) -> None:
    """检索结果包含结构化命中原因。"""
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Python project setup",
        body="Use uv for Python dependency management.",
        tags=["python", "uv"],
    )
    result = _retrieve(tmp_path, "python uv setup")
    assert len(result.memories) >= 1
    memory = result.memories[0]
    assert len(memory.hit_reasons) > 0
    # 每条 hit_reason 包含 token, tf, idf, boost 字段
    for reason in memory.hit_reasons:
        assert "token" in reason
        assert "tf" in reason
        assert "idf" in reason
        assert "boost" in reason
        assert isinstance(reason["tf"], int)
        assert isinstance(reason["idf"], float)
        assert isinstance(reason["boost"], float)
    # manifest 输出也包含 hit_reasons
    manifest = result.memories[0].to_manifest_dict()
    assert "hit_reasons" in manifest


def test_min_relevance_threshold(tmp_path: Path) -> None:
    """与查询完全无关的记忆不入选。"""
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="Kubernetes networking",
        body="Calico CNI plugin configuration for pod networking.",
        tags=["kubernetes"],
    )
    # 查询与记忆完全无关
    result = _retrieve(tmp_path, "chocolate cake recipe baking")
    assert len(result.memories) == 0


def test_phrase_level_match(tmp_path: Path) -> None:
    """'记忆系统' 作为 bigram 组合命中，而非单字误匹配。"""
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="HaAgent 记忆系统架构",
        body="记忆系统使用 BM25 检索算法。",
        tags=["记忆", "检索"],
    )
    # 另一条包含 "记" 和 "忆" 但不包含 "记忆" bigram 的干扰记忆
    _commit(
        tmp_path,
        scope="workspace",
        category="facts",
        title="日记和回忆",
        body="记录生活点滴，忆往昔峥嵘岁月。",
        tags=["日记"],
    )

    result = _retrieve(tmp_path, "记忆系统")
    assert len(result.memories) >= 1
    # "记忆系统" 的 bigram: 记忆, 忆系, 系统 —— 第一条记忆应高分命中
    assert result.memories[0].title == "HaAgent 记忆系统架构"

