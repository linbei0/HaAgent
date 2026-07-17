"""
tests/unit/scheduling/test_schedule_draft.py - ScheduleDraft/Patch 规范化与三态语义
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from haagent.scheduling.draft import (
    FieldPatch,
    ScheduleDraft,
    SchedulePatch,
    apply_patch,
    definition_from_draft,
)
from haagent.scheduling.models import RetryPolicy


def _draft(ws: Path, **overrides) -> ScheduleDraft:
    base = dict(
        name="daily",
        prompt="do work",
        workspace_root=ws,
        destination_kind="new_session",
        destination_session_path=None,
        connection_id="local",
        model="m1",
        web_enabled=True,
        allowed_tools=("file_read",),
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule="FREQ=DAILY",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=RetryPolicy(),
    )
    base.update(overrides)
    return ScheduleDraft(**base)  # type: ignore[arg-type]


def test_create_and_patch_share_web_tool_normalize(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    created = definition_from_draft(_draft(ws), schedule_id="sch_1")
    assert "web_search" in created.allowed_tools
    assert "web_fetch" in created.allowed_tools

    patched = apply_patch(
        created,
        SchedulePatch(
            expected_revision=1,
            web_enabled=FieldPatch.set(False),
            allowed_tools=FieldPatch.set(("file_read",)),
        ),
    )
    assert "web_search" not in patched.allowed_tools
    assert patched.revision == 2


def test_field_patch_rejects_illegal_states() -> None:
    import pytest

    with pytest.raises(ValueError, match="kind"):
        FieldPatch(kind="unknown", value="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="clear"):
        FieldPatch(kind="clear", value="unused")
    with pytest.raises(ValueError, match="set"):
        FieldPatch(kind="set", value=None)
    with pytest.raises(ValueError, match="unchanged"):
        FieldPatch(kind="unchanged", value="x")


def test_patch_clear_and_unchanged_semantics(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    current = definition_from_draft(
        _draft(ws, rrule="FREQ=DAILY", destination_kind="new_session"),
        schedule_id="sch_1",
    )
    cleared = apply_patch(
        current,
        SchedulePatch(expected_revision=1, rrule=FieldPatch.clear()),
    )
    assert cleared.rrule is None
    assert cleared.name == current.name

    renamed = apply_patch(
        current,
        SchedulePatch(expected_revision=1, name=FieldPatch.set("new-name")),
    )
    assert renamed.name == "new-name"
    assert renamed.rrule == current.rrule
