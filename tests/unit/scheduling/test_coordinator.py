"""
tests/unit/scheduling/test_coordinator.py - 租约、幂等 claim 与调度状态机

使用 fake clock；禁止 sleep。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.scheduling.coordinator import CoordinatorResult, ScheduleCoordinator
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
    destination: str = "new_session",
    session_path: Path | None = None,
    retry: RetryPolicy | None = None,
    allowed: tuple[str, ...] = ("file_read",),
) -> ScheduleDefinition:
    return ScheduleDefinition(
        id=schedule_id,
        name="c",
        prompt="p",
        workspace_root=tmp_path / "ws",
        destination_kind=destination,  # type: ignore[arg-type]
        destination_session_path=session_path,
        connection_id="c",
        model="m",
        web_enabled=False,
        allowed_tools=allowed,
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=dtstart or datetime(2026, 7, 13, 9, 0, 0),
        timezone="UTC",
        rrule=rrule,
        status="active",
        misfire_policy=misfire,  # type: ignore[arg-type]
        overlap_policy=overlap,  # type: ignore[arg-type]
        retry_policy=retry or RetryPolicy(max_attempts=3, initial_delay_seconds=30, multiplier=2.0, max_delay_seconds=900),
        revision=1,
    )


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, claim) -> object:
        self.executed.append(claim.run_id)
        return {"run_id": claim.run_id, "status": "succeeded"}


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
        exec_ = RecordingExecutor()
        coord = ScheduleCoordinator(store, owner_id="w1")
        r1 = coord.tick(now=now)
        r2 = coord.tick(now=now)
        runs = store.list_runs(schedule_id="sch_1")
        scheduled = [r for r in runs if r.trigger_kind == "scheduled"]
        assert len(scheduled) == 1
        assert r1.runs_created >= 1
        assert r2.runs_created == 0


def test_misfire_skip_latest_all(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
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
        with ScheduleStore(db) as store:
            # 清理：每策略用不同 id 在同一 db
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
            # 独占 lease：先 release 再 acquire
            store.release_lease(owner_id=f"w_{policy}")
            # 强制拿 lease：清空
            store._conn.execute("DELETE FROM coordinator_lease")
            result = coord.tick(now=now)
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
        store._conn.execute("DELETE FROM coordinator_lease")
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
        store._conn.execute("DELETE FROM coordinator_lease")
        coord2 = ScheduleCoordinator(store, owner_id="ow2")
        coord2.tick(now=now)
        q = [r for r in store.list_runs(schedule_id="ov_queue") if r.trigger_kind == "scheduled"]
        assert len(q) == 1
        assert q[0].status == "queued"


def test_resume_forbids_parallel_at_tick(tmp_path: Path) -> None:
    # 创建时 validate 已禁止；此处确保 store 中若出现则 tick 跳过
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    sess = tmp_path / "sess"
    sess.mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        # 直接插入绕过 validate 不现实；验证 create 失败
        with pytest.raises(Exception):
            store.create(
                _def(
                    tmp_path,
                    destination="resume_session",
                    session_path=sess,
                    overlap="parallel",
                ),
                now=now,
            )


def test_retry_backoff_and_exhaustion(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, retry=RetryPolicy(max_attempts=2, initial_delay_seconds=30, multiplier=2.0, max_delay_seconds=900)), now=now)
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="t1",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        claimed = store.claim_run(
            run.id,
            worker_id="w",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        assert claimed is not None
        coord = ScheduleCoordinator(store, owner_id="w")
        retry_at = coord.schedule_retry(
            run.id,
            now=now,
            category="model_transient",
            reason="rate limit",
        )
        assert retry_at is not None
        got = store.get_run(run.id)
        assert got is not None
        assert got.status == "retry_wait"
        assert got.retry_at_utc == now + timedelta(seconds=30)

        # 第二次 attempt 后耗尽
        later = now + timedelta(seconds=30)
        claimed2 = store.claim_run(
            run.id,
            worker_id="w",
            lease_expires_at=later + timedelta(seconds=45),
            now=later,
        )
        assert claimed2 is not None
        assert claimed2.attempt_count == 2
        exhausted = coord.schedule_retry(
            run.id,
            now=later,
            category="model_transient",
            reason="again",
        )
        assert exhausted is None
        final = store.get_run(run.id)
        assert final is not None
        assert final.status == "failed"


def test_non_retryable_needs_attention(tmp_path: Path) -> None:
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
        coord = ScheduleCoordinator(store, owner_id="w")
        coord.mark_needs_attention(
            run.id,
            now=now,
            category="interaction_required",
            reason="user input",
        )
        got = store.get_run(run.id)
        assert got is not None
        assert got.status == "needs_attention"


def test_recover_expired_running(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    (tmp_path / "ws").mkdir()
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path), now=now)
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="t3",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        store.claim_run(
            run.id,
            worker_id="dead",
            lease_expires_at=now + timedelta(seconds=1),
            now=now,
        )
        later = now + timedelta(seconds=60)
        coord = ScheduleCoordinator(store, owner_id="new")
        n = coord.recover_expired_runs(now=later)
        assert n == 1
        got = store.get_run(run.id)
        assert got is not None
        assert got.status in {"retry_wait", "interrupted", "queued"}
        # 按 retry policy 应进入 retry_wait 或 requeue
        assert got.failure_category == "worker_interrupted" or got.status == "retry_wait"
