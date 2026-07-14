"""
tests/unit/scheduling/test_coordinator.py - 租约、幂等 claim 与调度状态机

使用 fake clock；禁止 sleep。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from haagent.scheduling.coordinator import MISFIRE_GRACE_SECONDS, ScheduleCoordinator
from haagent.scheduling.models import RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _def(
    tmp_path: Path,
    *,
    schedule_id: str = "sch_1",
    rrule: str | None = "FREQ=HOURLY",
    dtstart: datetime | None = None,
    misfire: str = "latest",
    overlap: str = "skip",
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
        allowed_tools=("file_read",),
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=dtstart or datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule=rrule,
        status="active",
        misfire_policy=misfire,  # type: ignore[arg-type]
        overlap_policy=overlap,  # type: ignore[arg-type]
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=30,
            multiplier=2.0,
            max_delay_seconds=900,
        ),
        revision=1,
    )


def test_only_one_coordinator_gets_lease(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        a = ScheduleCoordinator(store, owner_id="a")
        b = ScheduleCoordinator(store, owner_id="b")
        assert a.heartbeat(now=now) is True
        assert b.heartbeat(now=now) is False
        # 过期后 b 可抢
        later = now + timedelta(seconds=31)
        assert b.heartbeat(now=later) is True


def test_duplicate_tick_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path, rrule="FREQ=HOURLY", dtstart=datetime(2026, 7, 13, 9, 0, 0)),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 9, 0, 0),
        )
        coord = ScheduleCoordinator(store, owner_id="w1")
        assert coord.tick(now=now) is True
        assert coord.tick(now=now) is True
        runs = store.list_runs(schedule_id="sch_1")
        scheduled = [r for r in runs if r.trigger_kind == "scheduled"]
        assert len(scheduled) == 1


def test_misfire_skip_latest_all(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    # 每小时，从 09:00 起；now=12:00 过期 09,10,11,12
    base = datetime(2026, 7, 13, 9, 0, 0)
    now = _utc(2026, 7, 13, 12, 0, 0)

    for policy, expected_count, expected_for in [
        # skip：grace 内的 12:00 仍执行；09/10/11 真正过期跳过
        ("skip", 1, _utc(2026, 7, 13, 12, 0, 0)),
        ("latest", 1, _utc(2026, 7, 13, 12, 0, 0)),
        ("all", 4, None),
    ]:
        sid = f"sch_{policy}"
        db = tmp_path / f"{sid}.db"
        with ScheduleStore(db) as store:
            store.create(
                _def(
                    tmp_path,
                    schedule_id=sid,
                    rrule="FREQ=HOURLY",
                    dtstart=base,
                    misfire=policy,
                ),
                now=now,
                next_run_at_utc=_utc(2026, 7, 13, 9, 0, 0),
            )
            coord = ScheduleCoordinator(store, owner_id=f"w_{policy}")
            assert coord.tick(now=now) is True
            runs = [
                r
                for r in store.list_runs(schedule_id=sid)
                if r.trigger_kind == "scheduled"
            ]
            if policy == "skip":
                assert len(runs) == expected_count
                assert runs[0].scheduled_for_utc == expected_for
                nxt = store.get_next_run_at_utc(sid)
                assert nxt is not None and nxt > now
            elif policy == "latest":
                assert len(runs) == 1
                assert runs[0].scheduled_for_utc == expected_for
            else:
                assert len(runs) == 4


def test_misfire_skip_honors_grace_boundary(tmp_path: Path) -> None:
    due = _utc(2026, 7, 13, 10, 0, 0)
    for late_seconds, expected_count in [
        (max(1, MISFIRE_GRACE_SECONDS // 2), 1),
        (MISFIRE_GRACE_SECONDS + 1, 0),
    ]:
        schedule_id = f"grace-{late_seconds}"
        with ScheduleStore(tmp_path / f"{schedule_id}.db") as store:
            store.create(
                _def(
                    tmp_path,
                    schedule_id=schedule_id,
                    misfire="skip",
                    dtstart=datetime(2026, 7, 13, 10, 0, 0),
                ),
                now=due - timedelta(hours=1),
                next_run_at_utc=due,
            )
            assert ScheduleCoordinator(store, owner_id="worker").tick(
                now=due + timedelta(seconds=late_seconds)
            )
            assert len(store.list_runs(schedule_id=schedule_id)) == expected_count


def test_finite_recurrence_marks_schedule_completed(tmp_path: Path) -> None:
    due = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(tmp_path / "finite.db") as store:
        definition = _def(
            tmp_path,
            rrule="FREQ=HOURLY;COUNT=1",
            dtstart=datetime(2026, 7, 13, 10, 0, 0),
            misfire="all",
        )
        store.create(
            definition,
            now=due - timedelta(minutes=1),
            next_run_at_utc=due,
        )
        assert ScheduleCoordinator(store, owner_id="worker").tick(
            now=due + timedelta(seconds=5)
        )
        completed = store.get(definition.id)
        assert completed is not None and completed.status == "completed"
        assert store.get_next_run_at_utc(definition.id) is None


def test_overlap_skip_queue_parallel(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path, schedule_id="ov_skip", overlap="skip", rrule="FREQ=HOURLY"),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 10, 0, 0),
        )
        store.create_run(
            schedule_id="ov_skip",
            schedule_revision=1,
            trigger_key="manual:hold",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="running",
            now=now,
        )
        coord = ScheduleCoordinator(store, owner_id="ow")
        coord.tick(now=now)
        scheduled = [
            r
            for r in store.list_runs(schedule_id="ov_skip")
            if r.trigger_kind == "scheduled"
        ]
        assert len(scheduled) == 1
        assert scheduled[0].status == "skipped"

        store.create(
            _def(tmp_path, schedule_id="ov_queue", overlap="queue", rrule="FREQ=HOURLY"),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 10, 0, 0),
        )
        store.create_run(
            schedule_id="ov_queue",
            schedule_revision=1,
            trigger_key="manual:hold2",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="running",
            now=now,
        )
        coord.release()
        coord2 = ScheduleCoordinator(store, owner_id="ow2")
        coord2.tick(now=now)
        q = [r for r in store.list_runs(schedule_id="ov_queue") if r.trigger_kind == "scheduled"]
        assert len(q) == 1
        assert q[0].status == "queued"
