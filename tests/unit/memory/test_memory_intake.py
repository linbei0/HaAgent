"""
tests/unit/memory/test_memory_intake.py - MemoryCandidateIntake 统一治理入口
"""

from __future__ import annotations

from pathlib import Path

from haagent.memory.candidates import CandidateQueue
from haagent.memory.identity import compute_identity
from haagent.memory.intake import MemoryCandidateIntake, MemoryDraft
from haagent.memory.schema import CandidateEvidence
from haagent.memory.store import MemoryStore


def _draft(
    *,
    title: str = "Prefer dark theme",
    body: str = "User prefers dark theme in the editor.",
    source: str = "user_explicit",
    fingerprint: str | None = "fp-1",
) -> MemoryDraft:
    return MemoryDraft(
        scope="user",
        category="user_preferences",
        title=title,
        body=body,
        evidence=CandidateEvidence(
            source_type="user_prompt",
            evidence_summary="user said prefer dark theme",
            evidence_quote="prefer dark theme",
            fingerprint=fingerprint,
        ),
        source=source,
        actor="user",
    )


def test_model_user_runtime_drafts_share_identity_and_governance(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / "session")
    store = MemoryStore(workspace_root=tmp_path)
    intake = MemoryCandidateIntake(store, queue)

    for source in ("extraction", "user_explicit", "runtime"):
        draft = _draft(
            source=source,
            title=f"Title {source}",
            body=f"Body for {source}",
            fingerprint=f"fp-{source}",
        )
        first = intake.submit(draft, reject_secrets=False)
        assert first.accepted is True
        assert first.identity is not None
        second = intake.submit(draft, reject_secrets=False)
        assert second.accepted is False
        assert second.reason == "duplicate_fingerprint_or_content"
        # identity 与 confirm 使用同一 compute_identity
        recomputed = compute_identity(
            scope=draft.scope,
            category=draft.category,
            title=draft.title,
            body=draft.body,
            evidence=draft.evidence,
        )
        assert first.identity.content_hash == recomputed.content_hash
        assert first.identity.memory_id == recomputed.memory_id


def test_secret_draft_rejected_with_fixed_reason(tmp_path: Path) -> None:
    queue = CandidateQueue(tmp_path / "session")
    store = MemoryStore(workspace_root=tmp_path)
    intake = MemoryCandidateIntake(store, queue)
    draft = _draft(title="API key", body="token=sk-abcdefghijklmnopqrstuvwxyz012345")
    result = intake.submit(draft, reject_secrets=True)
    assert result.accepted is False
    assert result.reason == "possible_secret"
    assert queue.list() == []


def test_extractor_path_does_not_import_store_create_in_loop() -> None:
    import inspect

    from haagent.memory import extraction

    source = inspect.getsource(extraction.MemoryExtractor._extract)
    assert "store.create_candidate" not in source
    assert "create_candidate" not in source
    assert "MemoryCandidateIntake" in source


def test_store_has_no_public_create_candidate_bypass() -> None:
    # 候选入队只能经 Intake；Store 只保留内部持久化，防止未来旁路。
    assert not hasattr(MemoryStore, "create_candidate")
    assert hasattr(MemoryStore, "_persist_candidate")
    assert callable(MemoryStore._persist_candidate)


def test_intake_is_only_production_writer_of_persist(tmp_path: Path) -> None:
    import inspect

    from haagent.memory import intake as intake_mod

    source = inspect.getsource(intake_mod.MemoryCandidateIntake.submit)
    assert "_persist_candidate" in source
