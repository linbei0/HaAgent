"""
tests/integration/scheduling/test_worker_lifecycle.py - schedule-worker 生命周期

覆盖 --once 派发、空队列、无效 DB、信号停止、租约占用与崩溃恢复。
"""

from __future__ import annotations

import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from haagent import cli
from haagent.cli_parser import build_cli_parser
from haagent.cli_runtime import CliRuntime
from haagent.scheduling.models import RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore
from haagent.scheduling.worker import ScheduleWorker, run_schedule_worker


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _def(tmp_path: Path, *, schedule_id: str = "sch_1", rrule: str | None = "FREQ=HOURLY") -> ScheduleDefinition:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ScheduleDefinition(
        id=schedule_id,
        name="w",
        prompt="p",
        workspace_root=ws,
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
        rrule=rrule,
        status="active",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=30, multiplier=2.0, max_delay_seconds=900),
        revision=1,
    )


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, claim) -> object:
        self.executed.append(claim.run_id)
        return {"run_id": claim.run_id, "status": "succeeded"}


def test_schedule_worker_hidden_from_root_help() -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "schedule-worker" not in help_text
    assert "open the Textual TUI" in help_text


def test_schedule_worker_parser_accepts_once_and_db() -> None:
    runtime = CliRuntime(project_root=Path.cwd())
    parser = build_cli_parser(runtime)
    args = parser.parse_args(["schedule-worker", "--once", "--db", "C:/tmp/schedules.sqlite3"])
    assert args.command == "schedule-worker"
    assert args.once is True
    assert Path(args.db) == Path("C:/tmp/schedules.sqlite3")
    assert callable(args.handler)


def test_once_dispatches_due_run(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
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
        code = worker.run_once()
    assert code == 0
    assert len(executor.executed) == 1


def test_once_empty_queue_exits_clean(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
    executor = RecordingExecutor()
    with ScheduleStore(db) as store:
        worker = ScheduleWorker(
            store,
            owner_id="worker-empty",
            executor=executor,
            clock=lambda: now,
            sleep=lambda _s: None,
        )
        code = worker.run_once()
    assert code == 0
    assert executor.executed == []


def test_invalid_db_exits_nonzero(tmp_path: Path) -> None:
    # 目录当作 DB 路径，打开应失败
    bad = tmp_path / "not-a-db-dir"
    bad.mkdir()
    code = run_schedule_worker(db_path=bad, once=True, owner_id="w")
    assert code != 0


def test_lease_occupied_once_does_not_double_dispatch(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path),
            now=now,
            next_run_at_utc=_utc(2026, 7, 13, 10, 0, 0),
        )
        # 先让其他 owner 持有租约
        assert store.acquire_lease(owner_id="other", now=now, ttl_seconds=30) is True
        executor = RecordingExecutor()
        worker = ScheduleWorker(
            store,
            owner_id="worker-b",
            executor=executor,
            clock=lambda: now,
            sleep=lambda _s: None,
        )
        code = worker.run_once()
    assert code == 0
    assert executor.executed == []


def test_crash_recovery_via_recover_expired_runs(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
    with ScheduleStore(db) as store:
        store.create(
            _def(tmp_path, rrule=None),
            now=now,
            next_run_at_utc=None,
        )
        run = store.create_run(
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="manual:stale",
            trigger_kind="manual",
            scheduled_for_utc=now,
            status="queued",
            now=now,
        )
        expired = now - timedelta(seconds=1)
        claimed = store.claim_run(
            run.id,
            worker_id="dead-worker",
            lease_expires_at=expired,
            now=now - timedelta(seconds=60),
        )
        assert claimed is not None
        # 过期 running 应被 recover 为 interrupted/retry_wait
        executor = RecordingExecutor()
        worker = ScheduleWorker(
            store,
            owner_id="recover-w",
            executor=executor,
            clock=lambda: now,
            sleep=lambda _s: None,
        )
        code = worker.run_once()
        refreshed = store.get_run(run.id)
    assert code == 0
    assert refreshed is not None
    assert refreshed.status in {"interrupted", "retry_wait", "queued", "running", "succeeded"}
    # 至少不应仍是过期 lease 的 running
    if refreshed.status == "running":
        assert refreshed.worker_id == "recover-w"


def test_signal_handler_only_sets_stop_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "s.db"
    now = _utc(2026, 7, 13, 10, 0, 0)
    handlers: dict[int, object] = {}

    def fake_signal(sig, handler):
        handlers[sig] = handler
        return None

    monkeypatch.setattr(signal, "signal", fake_signal)
    with ScheduleStore(db) as store:
        worker = ScheduleWorker(
            store,
            owner_id="sig-w",
            executor=RecordingExecutor(),
            clock=lambda: now,
            sleep=lambda _s: None,
        )
        worker.install_signal_handlers()
        assert signal.SIGTERM in handlers or signal.SIGINT in handlers
        # 调用 handler 只置位 stop，不抛、不直接 exit
        for handler in handlers.values():
            if callable(handler):
                handler(signal.SIGTERM, None)
        assert worker.stop_event.is_set()


def test_run_loop_heartbeats_and_respects_stop(tmp_path: Path) -> None:
    db = tmp_path / "s.db"
    base = _utc(2026, 7, 13, 10, 0, 0)
    ticks = {"n": 0}
    sleeps: list[float] = []

    def clock() -> datetime:
        return base + timedelta(seconds=ticks["n"] * 5)

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        ticks["n"] += 1
        if ticks["n"] >= 3:
            worker.stop_event.set()

    with ScheduleStore(db) as store:
        worker = ScheduleWorker(
            store,
            owner_id="loop-w",
            executor=RecordingExecutor(),
            clock=clock,
            sleep=sleep,
        )
        code = worker.run_forever()
    assert code == 0
    assert sleeps
    # 至少每 10 秒会 wakeup（sleep 上限 <= 10）
    assert all(s <= 10.0 for s in sleeps)


def test_cli_handler_once_empty_db(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    # 先创建合法空库
    with ScheduleStore(db):
        pass
    runtime = CliRuntime(project_root=Path.cwd())
    parser = build_cli_parser(runtime)
    args = parser.parse_args(["schedule-worker", "--once", "--db", str(db)])
    code = args.handler(args)
    assert code == 0


def test_cli_handler_invalid_db(tmp_path: Path) -> None:
    bad = tmp_path / "dir-as-db"
    bad.mkdir()
    runtime = CliRuntime(project_root=Path.cwd())
    parser = build_cli_parser(runtime)
    args = parser.parse_args(["schedule-worker", "--once", "--db", str(bad)])
    code = args.handler(args)
    assert code != 0
