"""
tests/unit/context/test_versioned_state.py - 版本化上下文状态契约
"""

from __future__ import annotations

import pytest

from haagent.context.versioned_state import (
    ContextStateError,
    ContextStateSnapshot,
    apply_delta,
    diff_snapshots,
    next_snapshot,
)


def test_snapshot_hash_is_stable_across_field_order() -> None:
    first = ContextStateSnapshot.create(
        sections={"working_state": {"b": 2, "a": 1}, "task_ledger": ["x"]},
        source_digests={"working_state": "sha256:w"},
    )
    second = ContextStateSnapshot.create(
        sections={"task_ledger": ["x"], "working_state": {"a": 1, "b": 2}},
        source_digests={"working_state": "sha256:w"},
    )
    assert first.snapshot_id == second.snapshot_id


def test_next_snapshot_creates_top_level_replace_delta_and_applies() -> None:
    base = ContextStateSnapshot.create(revision=1, sections={"todo": {"status": "pending"}, "old": "x"})
    target, delta = next_snapshot(base, {"todo": {"status": "done"}})
    assert delta is not None
    assert delta.changed == {"todo": {"status": "done"}}
    assert delta.removed == ["old"]
    assert apply_delta(base, delta) == target


def test_unchanged_state_does_not_emit_delta() -> None:
    base = ContextStateSnapshot.create(revision=1, sections={"todo": "same"})
    target, delta = next_snapshot(base, {"todo": "same"})
    assert target == base
    assert delta is None


def test_stale_base_is_rejected() -> None:
    base = ContextStateSnapshot.create(revision=1, sections={"todo": "a"})
    target, delta = next_snapshot(base, {"todo": "b"})
    assert delta is not None
    _, later_delta = next_snapshot(target, {"todo": "c"})
    assert later_delta is not None
    with pytest.raises(ContextStateError, match="stale"):
        apply_delta(base, later_delta)


def test_corrupted_snapshot_hash_is_rejected() -> None:
    snapshot = ContextStateSnapshot.create(sections={"todo": "a"})
    raw = snapshot.to_dict()
    raw["sections"] = {"todo": "b"}
    with pytest.raises(ContextStateError, match="snapshot_id"):
        ContextStateSnapshot.from_dict(raw)


def test_cross_epoch_diff_is_rejected() -> None:
    first = ContextStateSnapshot.create(epoch=0, revision=1)
    second = ContextStateSnapshot.create(epoch=1, revision=1)
    with pytest.raises(ContextStateError, match="different epochs"):
        diff_snapshots(first, second)
