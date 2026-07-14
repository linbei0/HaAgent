"""
tests/unit/scheduling/test_store.py - ScheduleStore schema、CRUD 与并发

验证 WAL、revision 快照、JSON 严格加载与 Windows 句柄释放。
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.scheduling.models import (
    RetryPolicy,
    ScheduleDefinition,
    ScheduleValidationError,
)
from haagent.scheduling.store import (
    ScheduleStore,
    ScheduleStoreError,
)


def _def(
    tmp_path: Path,
    *,
    schedule_id: str = "sch_1",
    name: str = "plan",
    status: str = "active",
    revision: int = 1,
    **overrides: object,
) -> ScheduleDefinition:
    base: dict[str, object] = {
        "id": schedule_id,
        "name": name,
        "prompt": "do it",
        "workspace_root": tmp_path / "ws",
        "destination_kind": "new_session",
        "destination_session_path": None,
        "connection_id": "conn",
        "model": "model-a",
        "web_enabled": False,
        "allowed_tools": ("file_read",),
        "approval_allowed_tools": ("file_write",),
        "approved_tools": (),
        "permission_mode": "request_approval",
        "dtstart_local": datetime(2026, 7, 13, 9, 0, 0),
        "timezone": "UTC",
        "rrule": "FREQ=DAILY",
        "status": status,
        "misfire_policy": "latest",
        "overlap_policy": "skip",
        "retry_policy": RetryPolicy(),
        "revision": revision,
    }
    base.update(overrides)
    return ScheduleDefinition(**base)  # type: ignore[arg-type]


def test_store_creates_schema_with_wal_and_version(tmp_path: Path) -> None:
    db = tmp_path / "schedules.sqlite3"
    with ScheduleStore(db):
        pass
    with sqlite3.connect(db) as conn:
        version = conn.execute("SELECT version FROM schema_version").fetchone()
        assert version == (1,)
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"


def test_create_get_list_update_revision(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    definition = _def(tmp_path, workspace_root=ws)
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        created = store.create(definition, now=now, next_run_at_utc=now)
        assert created.revision == 1
        got = store.get(created.id)
        assert got is not None
        assert got.name == "plan"
        assert got.allowed_tools == ("file_read",)
        assert store.list_schedules()[0].id == created.id

        updated = store.update(
            created.id,
            name="plan-v2",
            expected_revision=1,
            now=now,
            next_run_at_utc=now,
        )
        assert updated.revision == 2
        assert updated.name == "plan-v2"
        revision_1 = store.get_revision(created.id, 1)
        revision_2 = store.get_revision(created.id, 2)
        assert revision_1 is not None and revision_1.name == "plan"
        assert revision_2 is not None and revision_2.name == "plan-v2"


def test_optimistic_revision_conflict(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        created = store.create(_def(tmp_path, workspace_root=ws), now=now)
        with pytest.raises(ScheduleStoreError) as exc:
            store.update(
                created.id,
                name="x",
                expected_revision=99,
                now=now,
            )
        assert exc.value.code == "revision_conflict"


def test_invalid_json_tools_rejected_on_load(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)
        store._conn.execute(
            "UPDATE schedules SET allowed_tools_json = ? WHERE id = ?",
            ("{}", "sch_1"),
        )
        store._conn.commit()
        with pytest.raises(ScheduleStoreError) as exc:
            store.get("sch_1")
        assert exc.value.code == "invalid_json"


def test_archive_hidden_from_default_list(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)
        store.archive("sch_1", now=now)
        assert store.list_schedules() == []
        assert store.list_schedules(include_archived=True)[0].status == "archived"


def test_completed_filter(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws, schedule_id="a"), now=now)
        store.create(
            _def(tmp_path, workspace_root=ws, schedule_id="b", status="completed"),
            now=now,
        )
        completed = store.list_schedules(status="completed")
        assert [s.id for s in completed] == ["b"]


def test_delete_cascades_runs(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)
        store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="k1",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        store.delete("sch_1")
        assert store.get("sch_1") is None
        assert store.list_runs() == []


def test_future_schema_version_rejected(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version(version) VALUES (99)")
    conn.commit()
    conn.close()
    with pytest.raises(ScheduleStoreError) as exc:
        ScheduleStore(db)
    assert exc.value.code == "unsupported_schema"


def test_pause_resume(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    later = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now, next_run_at_utc=now)
        paused = store.pause("sch_1", now=now)
        assert paused.status == "paused"
        assert store.get_next_run_at_utc("sch_1") is None
        resumed = store.resume("sch_1", now=later, next_run_at_utc=later)
        assert resumed.status == "active"
        assert store.get_next_run_at_utc("sch_1") == later


def test_run_inbox_unread(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="manual:1",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="succeeded",
            now=now,
            summary="ok",
            unread=True,
        )
        assert run.unread is True
        store.mark_run_read(run.id)
        assert store.get_run(run.id).unread is False
        assert store.list_runs(unread_only=True) == []


def test_unique_trigger_key(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)
        store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="occ:2026-07-13T09:00:00+00:00",
            trigger_kind="scheduled",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        with pytest.raises(ScheduleStoreError) as exc:
            store.create_run(
                schedule_id="sch_1",
                schedule_revision=1,
                trigger_key="occ:2026-07-13T09:00:00+00:00",
                trigger_kind="scheduled",
                scheduled_for_utc=now,
                status="queued",
                now=now,
            )
        assert exc.value.code == "duplicate_trigger"


def test_parallel_policy_allows_second_run_while_first_is_running(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(tmp_path / "parallel.db") as store:
        definition = _def(tmp_path, overlap_policy="parallel")
        store.create(definition, now=now)
        first = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="parallel:first",
            trigger_kind="scheduled",
            scheduled_for_utc=now,
            now=now,
        )
        second = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="parallel:second",
            trigger_kind="scheduled",
            scheduled_for_utc=now,
            now=now,
        )
        assert store.claim_run(
            first.id,
            worker_id="worker-1",
            lease_expires_at=now,
            now=now,
        ) is not None

        claimable = store.list_claimable_runs(now=now, limit=10)
        assert second.id in {run.id for run in claimable}


def test_queue_policy_does_not_bypass_future_retry(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    retry_at = now.replace(hour=14)
    with ScheduleStore(tmp_path / "queue.db") as store:
        definition = _def(tmp_path, overlap_policy="queue")
        store.create(definition, now=now)
        first = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="queue:first",
            trigger_kind="scheduled",
            scheduled_for_utc=now,
            now=now,
        )
        second = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="queue:second",
            trigger_kind="scheduled",
            scheduled_for_utc=now.replace(hour=13),
            now=now,
        )
        claim = store.claim_run(
            first.id,
            worker_id="worker",
            lease_expires_at=now.replace(minute=5),
            now=now,
        )
        assert claim is not None
        assert second.id not in {
            run.id for run in store.list_claimable_runs(now=now, limit=10)
        }
        store.finish_run(
            first.id,
            status="retry_wait",
            now=now,
            retry_at_utc=retry_at,
            expected_worker_id="worker",
            expected_attempt=claim.attempt_count,
        )

        assert store.list_claimable_runs(now=now.replace(hour=13), limit=10) == []
        assert [
            run.id for run in store.list_claimable_runs(now=retry_at, limit=10)
        ] == [first.id]


def test_retry_wait_cancel_is_terminal(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(tmp_path / "retry-cancel.db") as store:
        definition = _def(tmp_path)
        store.create(definition, now=now)
        run = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="manual:retry-cancel",
            trigger_kind="manual",
            scheduled_for_utc=now,
            now=now,
        )
        claim = store.claim_run(
            run.id,
            worker_id="worker",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        assert claim is not None
        store.finish_run(
            run.id,
            status="retry_wait",
            now=now,
            retry_at_utc=now + timedelta(seconds=30),
            expected_worker_id="worker",
            expected_attempt=claim.attempt_count,
        )

        cancelled = store.request_cancel(run.id)
        assert cancelled.status == "cancelled"
        assert run.id not in {
            item.id
            for item in store.list_claimable_runs(
                now=now + timedelta(hours=1), limit=10
            )
        }


def test_cancellation_wins_finish_race(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(tmp_path / "cancel-race.db") as store:
        definition = _def(tmp_path)
        store.create(definition, now=now)
        run = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="manual:cancel-race",
            trigger_kind="manual",
            scheduled_for_utc=now,
            now=now,
        )
        claim = store.claim_run(
            run.id,
            worker_id="worker",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        assert claim is not None
        store.request_cancel(run.id)

        finished = store.finish_run(
            run.id,
            status="succeeded",
            now=now,
            expected_worker_id="worker",
            expected_attempt=claim.attempt_count,
        )
        assert finished.status == "cancelled"


def test_finish_requires_current_claim_fence(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(tmp_path / "fence.db") as store:
        definition = _def(tmp_path)
        store.create(definition, now=now)
        run = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="manual:fence",
            trigger_kind="manual",
            scheduled_for_utc=now,
            now=now,
        )
        first = store.claim_run(
            run.id,
            worker_id="worker-1",
            lease_expires_at=now + timedelta(seconds=5),
            now=now,
        )
        assert first is not None
        with pytest.raises(ScheduleStoreError) as missing:
            store.finish_run(run.id, status="succeeded", now=now)
        assert missing.value.code == "missing_fence"

        retry_at = now + timedelta(seconds=10)
        store.finish_run(
            run.id,
            status="retry_wait",
            now=retry_at,
            retry_at_utc=retry_at,
            expected_worker_id="worker-1",
            expected_attempt=first.attempt_count,
        )
        second = store.claim_run(
            run.id,
            worker_id="worker-2",
            lease_expires_at=retry_at + timedelta(seconds=45),
            now=retry_at,
        )
        assert second is not None and second.attempt_count == 2

        with pytest.raises(ScheduleStoreError) as stale:
            store.finish_run(
                run.id,
                status="succeeded",
                now=retry_at,
                expected_worker_id="worker-1",
                expected_attempt=first.attempt_count,
            )
        assert stale.value.code == "stale_finish"
        current = store.get_run(run.id)
        assert current is not None and current.worker_id == "worker-2"


def test_finish_updates_schedule_last_run_in_same_store_operation(tmp_path: Path) -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(tmp_path / "last-run.db") as store:
        definition = _def(tmp_path)
        store.create(definition, now=now)
        run = store.create_run(
            schedule_id=definition.id,
            schedule_revision=definition.revision,
            trigger_key="manual:last-run",
            trigger_kind="manual",
            scheduled_for_utc=now,
            now=now,
        )
        claim = store.claim_run(
            run.id,
            worker_id="worker",
            lease_expires_at=now + timedelta(seconds=45),
            now=now,
        )
        assert claim is not None
        store.finish_run(
            run.id,
            status="succeeded",
            now=now,
            expected_worker_id="worker",
            expected_attempt=claim.attempt_count,
        )
        assert store.get_last_run_at_utc(definition.id) == now


def test_connection_closed_allows_tmp_cleanup(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    store = ScheduleStore(db)
    store.create(_def(tmp_path, workspace_root=ws), now=now)
    store.close()
    # Windows：关闭后应可删除 db
    db.unlink()
    assert not db.exists()


def test_reader_does_not_block_writer(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    with ScheduleStore(db) as store:
        store.create(_def(tmp_path, workspace_root=ws), now=now)

    reader_ready = threading.Event()
    writer_done = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            with ScheduleStore(db) as r:
                reader_ready.set()
                _ = r.get("sch_1")
                writer_done.wait(timeout=5)
                _ = r.list_schedules()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    def writer() -> None:
        try:
            reader_ready.wait(timeout=5)
            with ScheduleStore(db) as w:
                w.update(
                    "sch_1",
                    name="from-writer",
                    expected_revision=1,
                    now=now,
                )
            writer_done.set()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)
            writer_done.set()

    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=writer)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors
