"""
tests/unit/scheduling/test_review_blockers_wave2.py - 评审阻断 P0/P1 复现

executor claim-token fencing、无 lease 不派发、parallel 可并行 claim、
queue 不越过未来 retry、续租失败触发取消、adapter 卸载 fail-fast。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from haagent.scheduling.executor import ScheduledRunExecutor
from haagent.scheduling.models import RunClaim, RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError
from haagent.scheduling.worker import ScheduleWorker


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _def(tmp_path: Path, **kwargs: object) -> ScheduleDefinition:
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
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=30),
        revision=1,
    )
    base.update(kwargs)
    return ScheduleDefinition(**base)  # type: ignore[arg-type]


def test_executor_finish_uses_claim_token_not_live_db_worker(tmp_path: Path) -> None:
    """stale w1 的 _finish 必须用 claim 时 token，不能读到 w2 后成功写入。"""
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path)
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
    claimed1 = store.claim_run(
        run.id,
        worker_id="w1",
        lease_expires_at=_utc(2026, 7, 13, 10, 0, 30),
        now=_utc(2026, 7, 13, 10),
    )
    assert claimed1 is not None
    assert claimed1.attempt_count == 1

    # w2 reclaims after lease expiry
    store._conn.execute(
        "UPDATE schedule_runs SET lease_expires_at_utc = ? WHERE id = ?",
        (_utc(2026, 7, 13, 10, 0, 1).isoformat().replace("+00:00", "Z"), run.id),
    )
    store._conn.commit()
    # recover path: mark interrupted then requeue-like reclaim via claim after status fix
    # Simulate reclaim: set queued then claim as w2 with attempt 2
    store.finish_run(
        run.id,
        status="queued",
        now=_utc(2026, 7, 13, 10, 1),
        expected_worker_id="w1",
        expected_attempt=1,
    )
    # finish to queued may not be valid - use direct SQL to re-queue for reclaim scenario
    # Better: use coordinator recover. Simpler: set status queued attempt stays
    store._conn.execute(
        """
        UPDATE schedule_runs SET status='queued', worker_id=NULL,
        lease_expires_at_utc=NULL WHERE id=?
        """,
        (run.id,),
    )
    store._conn.commit()
    claimed2 = store.claim_run(
        run.id,
        worker_id="w2",
        lease_expires_at=_utc(2026, 7, 13, 10, 5),
        now=_utc(2026, 7, 13, 10, 2),
    )
    assert claimed2 is not None
    assert claimed2.worker_id == "w2"
    assert claimed2.attempt_count == 2

    # stale executor path: _finish with claim token w1/1 must fail (stale_finish)
    executor = ScheduledRunExecutor(store)
    with pytest.raises(ScheduleStoreError) as ei:
        executor._finish(
            run.id,
            status="succeeded",
            now=_utc(2026, 7, 13, 10, 3),
            summary="stale w1",
            claim=RunClaim(run_id=run.id, worker_id="w1", attempt=1),
        )
    assert ei.value.code == "stale_finish"
    current = store.get_run(run.id)
    assert current is not None
    assert current.status == "running"
    assert current.worker_id == "w2"
    assert current.attempt_count == 2


def test_worker_does_not_dispatch_without_coordinator_lease(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path, id="sch_lease")
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 12), now=_utc(2026, 7, 13, 11))
    run = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="manual:pre",
        trigger_kind="manual",
        scheduled_for_utc=_utc(2026, 7, 13, 11, 30),
        status="queued",
        now=_utc(2026, 7, 13, 11),
    )
    # other owner holds coordinator lease
    assert store.acquire_lease(
        owner_id="other-owner",
        now=_utc(2026, 7, 13, 11),
        ttl_seconds=3600,
    )

    executed: list[str] = []

    class Rec:
        def execute(self, claim) -> object:
            executed.append(claim.run_id)
            return SimpleNamespace(run_id=claim.run_id, status="succeeded")

        def request_cancel(self, run_id: str) -> None:
            del run_id

    worker = ScheduleWorker(
        store,
        owner_id="me",
        executor=Rec(),
        clock=lambda: _utc(2026, 7, 13, 11, 30),
        sleep=lambda _: None,
    )
    worker.run_once()
    assert executed == [], f"must not dispatch without lease, got {executed}"
    still = store.get_run(run.id)
    assert still is not None and still.status == "queued"


def test_parallel_claimable_while_running(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path, id="sch_par", overlap_policy="parallel")
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    r1 = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="a",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 10),
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
        now=_utc(2026, 7, 13, 10),
    )
    store.claim_run(
        r1.id,
        worker_id="w1",
        lease_expires_at=_utc(2026, 7, 13, 10, 5),
        now=_utc(2026, 7, 13, 10),
    )
    claimable = store.list_claimable_runs(now=_utc(2026, 7, 13, 10, 2), limit=10)
    ids = [r.id for r in claimable]
    assert r2.id in ids, f"parallel_claimable_after_first {ids}"


def test_queue_blocks_later_when_earlier_retry_in_future(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path, id="sch_q", overlap_policy="queue")
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    first = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="first",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 10),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    second = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="second",
        trigger_kind="scheduled",
        scheduled_for_utc=_utc(2026, 7, 13, 11),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    # first enters future retry_wait
    store.claim_run(
        first.id,
        worker_id="w1",
        lease_expires_at=_utc(2026, 7, 13, 10, 5),
        now=_utc(2026, 7, 13, 10),
    )
    store.finish_run(
        first.id,
        status="retry_wait",
        now=_utc(2026, 7, 13, 10, 1),
        retry_at_utc=_utc(2026, 7, 13, 12),
        expected_worker_id="w1",
        expected_attempt=1,
    )
    claimable = store.list_claimable_runs(now=_utc(2026, 7, 13, 11, 30), limit=10)
    keys = [(r.trigger_key, r.status) for r in claimable if r.schedule_id == d.id]
    assert keys == [], f"queue_with_future_retry_claimable {keys}"
    # after retry due, first is claimable not second
    claimable2 = store.list_claimable_runs(now=_utc(2026, 7, 13, 12, 1), limit=10)
    assert [r.id for r in claimable2] == [first.id]


def test_run_lease_renew_failure_requests_cancel(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path)
    store.create(d, next_run_at_utc=_utc(2026, 7, 13, 10), now=_utc(2026, 7, 13, 9))
    run = store.create_run(
        schedule_id=d.id,
        schedule_revision=1,
        trigger_key="t",
        trigger_kind="manual",
        scheduled_for_utc=_utc(2026, 7, 13, 10),
        status="queued",
        now=_utc(2026, 7, 13, 10),
    )
    store.claim_run(
        run.id,
        worker_id="w1",
        lease_expires_at=_utc(2026, 7, 13, 10, 5),
        now=_utc(2026, 7, 13, 10),
    )
    cancelled: list[str] = []

    class Rec:
        def execute(self, claim) -> object:
            # simulate long work while heartbeat loses lease
            import time

            time.sleep(0.05)
            return SimpleNamespace(run_id=claim.run_id, status="succeeded")

        def request_cancel(self, run_id: str) -> None:
            cancelled.append(run_id)

    # force renew_run_lease to fail
    original = store.renew_run_lease

    def fail_renew(*a, **k):
        return False

    store.renew_run_lease = fail_renew  # type: ignore[method-assign]
    worker = ScheduleWorker(
        store,
        owner_id="w1",
        executor=Rec(),
        clock=lambda: _utc(2026, 7, 13, 10, 1),
        sleep=lambda _: None,
    )
    # inject short heartbeat for test
    import haagent.scheduling.worker as worker_mod

    old = worker_mod.HEARTBEAT_INTERVAL_SECONDS
    worker_mod.HEARTBEAT_INTERVAL_SECONDS = 0.01
    try:
        worker._execute_with_leases(RunClaim(run_id=run.id, worker_id=worker.owner_id, attempt=1))
    finally:
        worker_mod.HEARTBEAT_INTERVAL_SECONDS = old
        store.renew_run_lease = original  # type: ignore[method-assign]
    assert run.id in cancelled


def test_launchd_uninstall_fails_when_bootout_fails_even_if_file_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from haagent.scheduling.background.base import BackgroundServiceError
    from haagent.scheduling.background.launchd import LaunchdBackgroundAdapter

    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    adapter = LaunchdBackgroundAdapter(agents_dir=agents)
    adapter._plist_path.write_text("x", encoding="utf-8")

    def fake_run(args):
        if "print" in args:
            return SimpleNamespace(
                returncode=0, stdout="state = running\npid = 1\n", stderr=""
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="bootout failed")

    monkeypatch.setattr(adapter, "_run", fake_run)
    with pytest.raises(BackgroundServiceError):
        adapter.uninstall()


def test_systemd_uninstall_fails_when_stop_fails_even_if_unit_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from haagent.scheduling.background.base import BackgroundServiceError
    from haagent.scheduling.background.systemd import SystemdBackgroundAdapter

    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    adapter = SystemdBackgroundAdapter(unit_dir=unit_dir)
    adapter._unit_path.write_text("x", encoding="utf-8")

    def fake_systemctl(args):
        if "is-active" in args:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="access denied")

    monkeypatch.setattr(adapter, "_systemctl", fake_systemctl)
    with pytest.raises(BackgroundServiceError):
        adapter.uninstall()
