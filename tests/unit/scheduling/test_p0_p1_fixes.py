"""
tests/unit/scheduling/test_p0_p1_fixes.py - 计划任务 P0/P1 修复覆盖

租约续期、retry 接入、misfire skip 单槽、cancel CAS、编辑器 round-trip、脱敏。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.scheduling.coordinator import ScheduleCoordinator
from haagent.scheduling.executor import ScheduledRunExecutor, _bounded_summary, _map_exception
from haagent.scheduling.models import RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore
from haagent.scheduling.worker import ScheduleWorker
from haagent.tui.overlays.schedule_editor import (
    ScheduleEditorState,
    parse_rrule_fields,
)


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _def(
    tmp_path: Path,
    *,
    schedule_id: str = "sch_1",
    rrule: str | None = "FREQ=HOURLY",
    misfire: str = "latest",
    retry: RetryPolicy | None = None,
    allowed: tuple[str, ...] = ("file_read", "file_write"),
    approval: tuple[str, ...] = ("shell",),
    approved: tuple[str, ...] = (),
) -> ScheduleDefinition:
    return ScheduleDefinition(
        id=schedule_id,
        name="c",
        prompt="p",
        workspace_root=tmp_path / "ws",
        destination_kind="new_session",
        destination_session_path=None,
        connection_id="c",
        model="m",
        web_enabled=False,
        allowed_tools=allowed,
        approval_allowed_tools=approval,
        approved_tools=approved,
        permission_mode="request_approval",
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule=rrule,
        status="active",
        misfire_policy=misfire,  # type: ignore[arg-type]
        overlap_policy="skip",
        retry_policy=retry
        or RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=30,
            multiplier=2.0,
            max_delay_seconds=900,
        ),
        revision=1,
    )


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self.cancelled: list[str] = []

    def execute(self, claim) -> object:
        self.executed.append(claim.run_id)
        return {"run_id": claim.run_id}

    def request_cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)


def test_misfire_skip_single_late_still_fires(tmp_path: Path) -> None:
    """单槽略晚仍应创建 run，避免 skip 永不触发。"""
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 1)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path, schedule_id="sch_skip1", rrule="FREQ=HOURLY", misfire="skip"),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 10, 0, 0),
        )
        store._conn.execute("DELETE FROM coordinator_lease")
        coord = ScheduleCoordinator(store, owner_id="w")
        coord.tick(now=now)
        runs = [r for r in store.list_runs(schedule_id="sch_skip1") if r.trigger_kind == "scheduled"]
        assert len(runs) == 1


def test_misfire_skip_backlog_still_zero(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    # 11:30：最近 occurrence 11:00 已超出 grace，整批跳过
    now = _utc(2026, 7, 13, 11, 30, 0)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path, schedule_id="sch_skip_b", rrule="FREQ=HOURLY", misfire="skip"),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 9, 0, 0),
        )
        store._conn.execute("DELETE FROM coordinator_lease")
        coord = ScheduleCoordinator(store, owner_id="w")
        coord.tick(now=now)
        runs = store.list_runs(schedule_id="sch_skip_b")
        assert [r for r in runs if r.trigger_kind == "scheduled"] == []


def test_worker_run_once_dispatches_without_execute_on_tick(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
    (tmp_path / "ws").mkdir()
    executor = RecordingExecutor()
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 10, 0, 0),
        )
        worker = ScheduleWorker(
            store,
            owner_id="worker-a",
            executor=executor,
            clock=lambda: now,
            sleep=lambda _s: None,
        )
        assert worker.run_once() == 0
    assert len(executor.executed) == 1


def test_retry_wait_cancel_is_terminal(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path), now=now)
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="t1",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        store.claim_run(
            run.id,
            worker_id="w",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        store.finish_run(
            run.id,
            status="retry_wait",
            now=now,
            summary="rate limit",
            failure_category="model_transient",
            retry_at_utc=now + timedelta(seconds=30),
            expected_worker_id="w",
            expected_attempt=1,
        )
        cancelled = store.request_cancel(run.id)
        assert cancelled.status == "cancelled"
        claimable = store.list_claimable_runs(now=now + timedelta(hours=1), limit=10)
        assert all(r.id != run.id for r in claimable)


def test_finish_run_honors_cancellation_cas(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path), now=now)
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="t2",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        store.claim_run(
            run.id,
            worker_id="w",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        store.request_cancel(run.id)
        finished = store.finish_run(
            run.id,
            status="succeeded",
            now=now,
            summary="should not win",
            expected_worker_id="w",
            expected_attempt=1,
        )
        assert finished.status == "cancelled"


def test_parse_rrule_weekly_friday() -> None:
    fields = parse_rrule_fields("FREQ=WEEKLY;BYDAY=FR")
    assert fields["frequency"] == "weekly"
    assert fields["byday"] == "FR"


def test_editor_round_trip_preserves_definition(tmp_path: Path) -> None:
    write_tools = (
        "file_list",
        "grep",
        "file_read",
        "file_write",
        "apply_patch",
        "skill_list",
        "skill_read",
    )
    item = _def(
        tmp_path,
        rrule="FREQ=WEEKLY;BYDAY=FR",
        allowed=write_tools,
        approval=("shell",),
        approved=(),
        retry=RetryPolicy(
            max_attempts=5,
            initial_delay_seconds=10,
            multiplier=3.0,
            max_delay_seconds=600,
        ),
    )
    # 伪装 AssistantSchedule 字段
    state = ScheduleEditorState.from_schedule(item)
    assert state.frequency == "weekly"
    assert state.byday == "FR"
    assert state.retry_max_attempts == 5
    assert state.retry_initial_delay_seconds == 10
    assert state.retry_multiplier == 3.0
    assert state.tool_preset == "workspace_write"
    req = state.to_create_request()
    assert req.rrule == "FREQ=WEEKLY;BYDAY=FR"
    assert req.retry_policy.max_attempts == 5
    assert req.retry_policy.initial_delay_seconds == 10
    assert "file_write" in req.allowed_tools
    assert req.approval_allowed_tools == ("shell",)
    assert req.permission_mode == "request_approval"


def test_editor_rejects_empty_name_prompt() -> None:
    state = ScheduleEditorState(name="", prompt="")
    assert state.validate_for_save() is not None
    with pytest.raises(ValueError):
        state.to_create_request()


def test_bounded_summary_redacts_secrets() -> None:
    text = _bounded_summary("token sk-abcdefghijklmnopqrstuvwxyz value")
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in text
    assert "REDACTED" in text


def test_map_exception_prefers_structured_types() -> None:
    status, category, _ = _map_exception(FileNotFoundError("missing"))
    assert status == "needs_attention"
    assert category == "workspace_unavailable"
