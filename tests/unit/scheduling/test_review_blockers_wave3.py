"""
tests/unit/scheduling/test_review_blockers_wave3.py - 延迟 stale execute 与 adapter 停止证明

P0: 晚到的旧 executor 不得冒用新 claim 的 worker/attempt。
P1: systemd 仅 disable 成功≠已停；daemon-reload 失败须抛错。
P1: launchd print 权限/通信失败≠未加载。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from haagent.scheduling.background.base import BackgroundServiceError
from haagent.scheduling.background.launchd import LaunchdBackgroundAdapter
from haagent.scheduling.background.systemd import SystemdBackgroundAdapter
from haagent.scheduling.executor import ScheduledRunExecutor
from haagent.scheduling.models import RunClaim, RetryPolicy, ScheduleDefinition
from haagent.scheduling.store import ScheduleStore, ScheduleStoreError


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


def test_late_stale_execute_cannot_impersonate_new_claim(tmp_path: Path) -> None:
    """w1 在 w2 reclaim 后才进入 execute，必须 stale_execute，不得改写 w2 run。"""
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

    # lease 过期后 w2 reclaim → attempt 2
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

    executor = ScheduledRunExecutor(store)
    with pytest.raises(ScheduleStoreError) as ei:
        executor.execute(RunClaim(run_id=run.id, worker_id="w1", attempt=1))
    assert ei.value.code == "stale_execute"

    current = store.get_run(run.id)
    assert current is not None
    assert current.status == "running"
    assert current.worker_id == "w2"
    assert current.attempt_count == 2


def test_execute_rejects_wrong_status_without_finish(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    store = ScheduleStore(tmp_path / "s.db")
    d = _def(tmp_path, id="sch_q")
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
    executor = ScheduledRunExecutor(store)
    with pytest.raises(ScheduleStoreError) as ei:
        executor.execute(RunClaim(run_id=run.id, worker_id="w1", attempt=1))
    assert ei.value.code == "stale_execute"
    assert store.get_run(run.id).status == "queued"  # type: ignore[union-attr]


def test_systemd_plain_disable_success_not_proof_when_still_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """disable --now 失败 + 普通 disable 成功 + is-active=active → 必须 error。"""
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    adapter = SystemdBackgroundAdapter(unit_dir=unit_dir)
    adapter._unit_path.write_text("x", encoding="utf-8")

    def fake_systemctl(args: list[str]):
        if "is-active" in args:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if "--now" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="disable --now failed")
        if "disable" in args:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "daemon-reload" in args:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="unknown")

    monkeypatch.setattr(adapter, "_systemctl", fake_systemctl)
    with pytest.raises(BackgroundServiceError) as ei:
        adapter.uninstall()
    assert adapter._unit_path.exists(), "must not delete unit when still active"
    # 错误已抛出即可；detail 可能来自 disable --now 的 stderr
    assert str(ei.value)


def test_systemd_daemon_reload_fail_after_unit_delete_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    adapter = SystemdBackgroundAdapter(unit_dir=unit_dir)
    adapter._unit_path.write_text("x", encoding="utf-8")

    def fake_systemctl(args: list[str]):
        if "daemon-reload" in args:
            return SimpleNamespace(returncode=1, stdout="", stderr="reload denied")
        if "is-active" in args:
            return SimpleNamespace(returncode=3, stdout="inactive\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(adapter, "_systemctl", fake_systemctl)
    with pytest.raises(BackgroundServiceError) as ei:
        adapter.uninstall()
    assert "reload" in str(ei.value).lower() or "denied" in str(ei.value).lower()


def test_launchd_print_access_denied_is_error_not_uninstalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    adapter = LaunchdBackgroundAdapter(agents_dir=agents)
    adapter._plist_path.write_text("x", encoding="utf-8")

    def fake_run(args: list[str]):
        if "print" in args:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="launchctl print failed: access denied",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="bootout failed")

    monkeypatch.setattr(adapter, "_run", fake_run)
    with pytest.raises(BackgroundServiceError) as ei:
        adapter.uninstall()
    assert adapter._plist_path.exists()
    msg = str(ei.value).lower()
    assert "access" in msg or "denied" in msg or "权限" in str(ei.value)


def test_launchd_print_not_found_allows_uninstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    adapter = LaunchdBackgroundAdapter(agents_dir=agents)
    adapter._plist_path.write_text("x", encoding="utf-8")

    def fake_run(args: list[str]):
        if "print" in args:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=f"Could not find service \"{adapter._plist_path.name}\" in domain",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="bootout failed")

    monkeypatch.setattr(adapter, "_run", fake_run)
    status = adapter.uninstall()
    assert status.state == "not_installed"
    assert not adapter._plist_path.exists()
