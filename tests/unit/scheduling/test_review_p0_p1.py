"""
tests/unit/scheduling/test_review_p0_p1.py - 评审阻断项回归

fencing finish、parallel 副作用、queue 串行、misfire grace、有限 RRULE 完成、
编辑器 INTERVAL/retry0/空 custom tools、get_last_run_at。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.scheduling.coordinator import MISFIRE_GRACE_SECONDS, ScheduleCoordinator
from haagent.scheduling.models import (
    RetryPolicy,
    ScheduleDefinition,
    ScheduleValidationError,
    validate_schedule,
)
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError
from haagent.scheduling.worker import ScheduleWorker
from haagent.tui.overlays.schedule_editor import ScheduleEditorState, parse_rrule_fields


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _def(
    tmp_path: Path,
    **kwargs: object,
) -> ScheduleDefinition:
    base = dict(
        id="sch_1",
        name="n",
        prompt="p",
        workspace_root=tmp_path / "ws",
        destination_kind="new_session",
        destination_session_path=None,
        connection_id="c",
        model="m",
        web_enabled=False,
        allowed_tools=("file_read",),
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule="FREQ=HOURLY",
        status="active",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=30,
            multiplier=2.0,
            max_delay_seconds=900,
        ),
        revision=1,
    )
    base.update(kwargs)
    return ScheduleDefinition(**base)  # type: ignore[arg-type]


def test_finish_run_rejects_stale_worker_after_reclaim(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path)
    (tmp_path / "ws").mkdir()
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    run = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="t1",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 10),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    t0 = _utc(2026, 7, 13, 10, 0, 1)
    claimed1 = store.claim_run(
        run.id,
        worker_id="w1",
        lease_expires_at=t0 + timedelta(seconds=5),
        now=t0,
    )
    assert claimed1 is not None
    assert claimed1.worker_id == "w1"
    assert claimed1.attempt_count == 1

    # lease 过期后 recover 再被 w2 claim
    t_exp = t0 + timedelta(seconds=10)
    store.finish_run(
        run.id,
        status="interrupted",
        now=t_exp,
        summary="expired",
        failure_category="worker_interrupted",
        expected_worker_id="w1",
        expected_attempt=1,
    )
    store.finish_run(
        run.id,
        status="retry_wait",
        now=t_exp,
        summary="retry",
        failure_category="worker_interrupted",
        retry_at_utc=t_exp,
        expected_worker_id="w1",
        expected_attempt=1,
    )
    claimed2 = store.claim_run(
        run.id,
        worker_id="w2",
        lease_expires_at=t_exp + timedelta(seconds=30),
        now=t_exp,
    )
    assert claimed2 is not None
    assert claimed2.worker_id == "w2"
    assert claimed2.attempt_count == 2

    # 过期 w1 完成必须被 fencing 拒绝
    with pytest.raises(ScheduleStoreError) as exc:
        store.finish_run(
            run.id,
            status="succeeded",
            now=t_exp + timedelta(seconds=1),
            summary="stale w1",
            expected_worker_id="w1",
            expected_attempt=1,
        )
    assert exc.value.code == "stale_finish"

    final = store.get_run(run.id)
    assert final is not None
    assert final.status == "running"
    assert final.worker_id == "w2"
    assert final.attempt_count == 2

    ok = store.finish_run(
        run.id,
        status="succeeded",
        now=t_exp + timedelta(seconds=2),
        summary="w2 ok",
        expected_worker_id="w2",
        expected_attempt=2,
    )
    assert ok.status == "succeeded"
    assert ok.worker_id == "w2"


def test_parallel_forbids_side_effect_tools(tmp_path: Path) -> None:
    d = _def(
        tmp_path,
        allowed_tools=("file_read", "file_write"),
        overlap_policy="parallel",
    )
    with pytest.raises(ScheduleValidationError) as exc:
        validate_schedule(d)
    assert exc.value.code == "parallel_forbids_side_effects"

    d2 = _def(
        tmp_path,
        allowed_tools=("file_read", "shell"),
        overlap_policy="parallel",
        approval_allowed_tools=("shell",),
    )
    with pytest.raises(ScheduleValidationError):
        validate_schedule(d2)

    ok = _def(
        tmp_path,
        allowed_tools=("file_read", "grep", "file_list"),
        overlap_policy="parallel",
    )
    validate_schedule(ok)


def test_queue_overlap_does_not_claim_second_while_running(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "q.db")
    (tmp_path / "ws").mkdir()
    d = _def(tmp_path, overlap_policy="queue", rrule="FREQ=MINUTELY;INTERVAL=1")
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    r1 = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="a",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 10, 0),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    r2 = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="b",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 10, 1),
        status="queued",
        now=_utc(2026, 7, 13, 10, 1),
    )
    now = _utc(2026, 7, 13, 10, 2)
    claimable = store.list_claimable_runs(now=now, limit=10)
    assert [c.id for c in claimable] == [r1.id]

    store.claim_run(
        r1.id,
        worker_id="w",
        lease_expires_at=now + timedelta(seconds=60),
        now=now,
    )
    claimable2 = store.list_claimable_runs(now=now, limit=10)
    assert r2.id not in {c.id for c in claimable2}

    store.finish_run(
        r1.id,
        status="succeeded",
        now=now + timedelta(seconds=1),
        expected_worker_id="w",
        expected_attempt=1,
    )
    claimable3 = store.list_claimable_runs(now=now + timedelta(seconds=2), limit=10)
    assert [c.id for c in claimable3] == [r2.id]


def test_misfire_skip_skips_even_single_past_outside_grace(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "m.db")
    (tmp_path / "ws").mkdir()
    d = _def(
        tmp_path,
        misfire_policy="skip",
        rrule="FREQ=HOURLY",
        dtstart_local=datetime(2026, 7, 13, 8, 0, 0),
    )
    # next 在 1 小时前，远超 grace
    past = _utc(2026, 7, 13, 9, 0, 0)
    store.create(d, next_run_at_utc=past, now=past - timedelta(hours=1))
    coord = ScheduleCoordinator(store, owner_id="c1")
    now = past + timedelta(seconds=MISFIRE_GRACE_SECONDS + 30)
    result = coord.tick(now=now)
    assert result.runs_created == 0
    runs = store.list_runs(schedule_id=d.id, limit=20)
    assert runs == []
    # next 已推进
    nxt = store.get_next_run_at_utc(d.id)
    assert nxt is not None
    assert nxt > past


def test_misfire_skip_fires_within_grace(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "mg.db")
    (tmp_path / "ws").mkdir()
    d = _def(
        tmp_path,
        misfire_policy="skip",
        rrule="FREQ=HOURLY",
        dtstart_local=datetime(2026, 7, 13, 8, 0, 0),
    )
    due = _utc(2026, 7, 13, 10, 0, 0)
    store.create(d, next_run_at_utc=due, now=due - timedelta(hours=1))
    coord = ScheduleCoordinator(store, owner_id="c1")
    now = due + timedelta(seconds=max(1, MISFIRE_GRACE_SECONDS // 2))
    result = coord.tick(now=now)
    assert result.runs_created == 1


def test_finite_rrule_count_completes_schedule(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "c.db")
    (tmp_path / "ws").mkdir()
    d = _def(
        tmp_path,
        rrule="FREQ=HOURLY;COUNT=1",
        dtstart_local=datetime(2026, 7, 13, 9, 0, 0),
        misfire_policy="all",
    )
    due = _utc(2026, 7, 13, 9, 0, 0)
    store.create(d, next_run_at_utc=due, now=due - timedelta(minutes=1))
    coord = ScheduleCoordinator(store, owner_id="c1")
    coord.tick(now=due + timedelta(seconds=5))
    sch = store.get(d.id)
    assert sch is not None
    assert sch.status == "completed"
    assert store.get_next_run_at_utc(d.id) is None


def test_editor_weekly_interval_round_trip() -> None:
    fields = parse_rrule_fields("FREQ=WEEKLY;INTERVAL=2;BYDAY=MO")
    state = ScheduleEditorState(
        name="n",
        prompt="p",
        frequency=fields["frequency"],
        interval_value=int(fields["interval_value"]),
        interval_unit=str(fields["interval_unit"]),
        byday=str(fields["byday"]),
        bymonthday=int(fields["bymonthday"]),
        custom_rrule=str(fields["custom_rrule"]),
    )
    # weekly with INTERVAL>1 should stay weekly and emit INTERVAL
    assert state.frequency == "weekly"
    rrule = state.build_rrule()
    assert rrule is not None
    assert "INTERVAL=2" in rrule
    assert "BYDAY=MO" in rrule


def test_editor_monthly_interval_preserved() -> None:
    fields = parse_rrule_fields("FREQ=MONTHLY;INTERVAL=3;BYMONTHDAY=15")
    state = ScheduleEditorState(
        name="n",
        prompt="p",
        frequency=fields["frequency"],
        interval_value=int(fields["interval_value"]),
        bymonthday=int(fields["bymonthday"]),
        byday=str(fields["byday"]),
        custom_rrule=str(fields["custom_rrule"]),
    )
    rrule = state.build_rrule()
    assert rrule is not None
    assert "INTERVAL=3" in rrule
    assert "BYMONTHDAY=15" in rrule


def test_editor_custom_empty_tools_not_replaced_by_readonly() -> None:
    state = ScheduleEditorState(
        name="n",
        prompt="p",
        tool_preset="custom",
        custom_allowed_tools=(),
        connection_id="c",
        model="m",
        workspace_root=str(Path("E:/ws").resolve()) if False else "E:/ws",
    )
    # use absolute path that may not exist on disk — to_create uses Path only
    state = state.with_field("workspace_root", "E:/absolute/ws")
    req = state.to_create_request()
    assert req.allowed_tools == ()


def test_editor_retry_initial_delay_zero_preserved() -> None:
    state = ScheduleEditorState(
        name="n",
        prompt="p",
        workspace_root="E:/ws",
        connection_id="c",
        model="m",
        retry_max_attempts=2,
        retry_initial_delay_seconds=0,
        retry_multiplier=2.0,
        retry_max_delay_seconds=60,
    )
    req = state.to_create_request()
    assert req.retry_policy.initial_delay_seconds == 0


def test_get_last_run_at_public_api(tmp_path: Path) -> None:
    store = ScheduleStore(tmp_path / "l.db")
    (tmp_path / "ws").mkdir()
    d = _def(tmp_path)
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    assert store.get_last_run_at_utc(d.id) is None
    ts = _utc(2026, 7, 13, 10, 5)
    store.set_last_run(d.id, last_run_at_utc=ts, now=ts)
    assert store.get_last_run_at_utc(d.id) == ts


def test_worker_passes_fencing_to_executor_path(tmp_path: Path) -> None:
    """claim 后 finish 必须带 worker_id；无 token 时 running 状态不允许裸 finish。"""
    store = ScheduleStore(tmp_path / "f.db")
    (tmp_path / "ws").mkdir()
    d = _def(tmp_path)
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    run = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="x",
        trigger_kind="manual",
        scheduled_for_utc=_utc(2026, 7, 13, 10),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    now = _utc(2026, 7, 13, 10, 1)
    claimed = store.claim_run(
        run.id,
        worker_id="w-fence",
        lease_expires_at=now + timedelta(seconds=60),
        now=now,
    )
    assert claimed is not None
    # 不带 expected_worker_id 的 finish 在 running 上应拒绝（防遗漏）
    with pytest.raises(ScheduleStoreError) as exc:
        store.finish_run(
            run.id,
            status="succeeded",
            now=now + timedelta(seconds=1),
            summary="no fence",
        )
    assert exc.value.code in {"stale_finish", "missing_fence"}
