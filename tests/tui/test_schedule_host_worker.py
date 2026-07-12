"""
tests/tui/test_schedule_host_worker.py - TUI 内嵌 ScheduleCoordinator host

验证 on_mount 启动 worker、到期 tick 派发、unmount 释放租约；Fake 路径不启 host。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from haagent.app.assistant_context import AssistantContext
from haagent.app.schedule_usecases import AssistantSchedules
from haagent.scheduling.models import RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore
from haagent.tui.application.schedule_flow import ScheduleFlow
from tests.tui.support import FakeAssistantService, FakeSchedules


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


def _definition(tmp_path: Path) -> ScheduleDefinition:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ScheduleDefinition(
        id="sch_host_1",
        name="host",
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


class RecordingExecutor:
    def __init__(self) -> None:
        self.executed: list[str] = []
        self._event = threading.Event()

    def execute(self, claim) -> object:
        self.executed.append(claim.run_id)
        self._event.set()
        return {"run_id": claim.run_id, "status": "succeeded"}

    def request_cancel(self, run_id: str) -> None:
        del run_id


class _AppStub:
    def __init__(self, service: object) -> None:
        self.service = service
        self.is_mounted = True
        self._intervals: list = []

    def set_interval(self, seconds: float, callback) -> object:
        handle = SimpleNamespace(seconds=seconds, callback=callback, stopped=False)

        def stop() -> None:
            handle.stopped = True

        handle.stop = stop  # type: ignore[attr-defined]
        self._intervals.append(handle)
        return handle

    def _refresh(self) -> None:
        return None


def _real_schedules(tmp_path: Path, store: ScheduleStore) -> AssistantSchedules:
    from haagent.models.gateway_registry import gateway_from_profile
    from haagent.runtime.session.agent import AgentSession

    ctx = AssistantContext(
        workspace_root=tmp_path,
        runs_root=tmp_path / "runs",
        environ={},
        gateway_factory=gateway_from_profile,
        session_factory=AgentSession,
        max_turns=8,
        enable_web=False,
        initial_resume=None,
        initial_continue=False,
        schedule_db_path=store.path,
    )
    return AssistantSchedules(ctx, store=store)


def test_fake_schedules_does_not_require_host_worker(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.schedules = FakeSchedules(service)
    app = _AppStub(service)
    flow = ScheduleFlow(app)
    flow.start_background_polling()
    assert flow.host_worker_running is False
    flow.stop_background_polling()


def test_schedule_flow_starts_host_worker_and_dispatches(tmp_path: Path) -> None:
    db = tmp_path / "schedules.sqlite3"
    now = _utc(2026, 7, 13, 10, 0, 0)
    executor = RecordingExecutor()
    store = ScheduleStore(db)
    store.create(_definition(tmp_path), now=now, next_run_at_utc=now)

    schedules = _real_schedules(tmp_path, store)
    service = SimpleNamespace(schedules=schedules)
    app = _AppStub(service)
    flow = ScheduleFlow(app)

    flow.start_coordinator_host(
        executor=executor,
        clock=lambda: now,
        sleep=lambda _s: time.sleep(0.05),
    )
    assert flow.host_worker_running is True
    assert executor._event.wait(timeout=3.0), "host worker 应在超时前派发到期 run"
    assert len(executor.executed) == 1

    flow.stop_background_polling()
    assert flow.host_worker_running is False

    # unmount 后租约应释放，另一 owner 可 acquire
    other = ScheduleStore(db)
    assert other.acquire_lease(owner_id="other", now=now, ttl_seconds=30) is True
    other.close()
    store.close()


def test_stop_host_releases_lease_without_dispatch(tmp_path: Path) -> None:
    db = tmp_path / "schedules.sqlite3"
    now = _utc(2026, 7, 13, 10, 0, 0)
    store = ScheduleStore(db)
    schedules = _real_schedules(tmp_path, store)
    service = SimpleNamespace(schedules=schedules)
    app = _AppStub(service)
    flow = ScheduleFlow(app)
    flow.start_coordinator_host(
        executor=RecordingExecutor(),
        clock=lambda: now,
        sleep=lambda _s: time.sleep(0.2),
    )
    assert flow.host_worker_running is True
    flow.stop_background_polling()
    assert flow.host_worker_running is False
    other = ScheduleStore(db)
    assert other.acquire_lease(owner_id="after-stop", now=now, ttl_seconds=30) is True
    other.close()
    store.close()


def test_start_background_polling_starts_host_when_store_available(tmp_path: Path) -> None:
    db = tmp_path / "schedules.sqlite3"
    store = ScheduleStore(db)
    schedules = _real_schedules(tmp_path, store)
    service = SimpleNamespace(schedules=schedules)
    app = _AppStub(service)
    flow = ScheduleFlow(app)
    flow.start_background_polling()
    assert flow.host_worker_running is True
    flow.stop_background_polling()
    assert flow.host_worker_running is False
    store.close()


def test_host_status_surfaces_last_error(tmp_path: Path) -> None:
    db = tmp_path / "schedules.sqlite3"
    store = ScheduleStore(db)
    schedules = _real_schedules(tmp_path, store)
    # 注入 fatal worker 状态
    schedules._host_worker = SimpleNamespace(
        owner_id="tui-x",
        last_error="store:disk_full:boom",
        fatal=True,
        stop_event=threading.Event(),
    )
    schedules._host_thread = SimpleNamespace(is_alive=lambda: False)
    status = schedules.host_status()
    assert status.running is False
    assert status.fatal is True
    assert "disk_full" in (status.last_error or "")
    store.close()
