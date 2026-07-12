"""
tests/unit/scheduling/test_store.py - ScheduleStore schema、CRUD 与并发

验证 WAL、revision 快照、JSON 严格加载与 Windows 句柄释放。
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
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
    with ScheduleStore(db) as store:
        assert store.schema_version() == 1
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"
        fk = store._conn.execute("PRAGMA foreign_keys").fetchone()
        assert fk[0] == 1


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
        revs = store.list_revisions(created.id)
        assert len(revs) == 2
        assert revs[0].revision == 1
        assert revs[1].revision == 2


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
        assert store.unread_count() == 0


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
