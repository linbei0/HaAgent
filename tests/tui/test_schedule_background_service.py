"""
tests/tui/test_schedule_background_service.py - 计划任务后台服务页 TUI

覆盖状态展示、安装/卸载确认与默认拒绝焦点。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.overlays.modals import ConfirmModal
from haagent.tui.overlays.schedule_background import ScheduleBackgroundState
from haagent.tui.overlays.schedules import SchedulesOverlay
from tests.tui.support import FakeAssistantService, FakeSchedules, _all_text


def test_background_state_render_labels() -> None:
    for state, label_hint in [
        ("not_installed", "未安装"),
        ("stopped", "已停止"),
        ("running", "运行中"),
        ("error", "异常"),
    ]:
        status = BackgroundServiceStatus(
            state=state,
            host_type="windows_task",
            detail=f"detail-{state}",
            executable="C:/haagent.exe",
            last_heartbeat_utc=datetime(2026, 7, 12, 0, 0, tzinfo=timezone.utc),
        )
        text = ScheduleBackgroundState(status=status).render()
        assert label_hint in text or state in text
        assert "windows" in text.lower() or "host" in text.lower() or "Task" in text or "任务" in text
        assert "安装" in text or "卸载" in text or "诊断" in text


def test_background_empty_and_unsupported() -> None:
    status = BackgroundServiceStatus(
        state="unsupported",
        host_type="none",
        detail="当前平台不支持后台服务",
    )
    text = ScheduleBackgroundState(status=status).render()
    assert "不支持" in text or "unsupported" in text


def test_tui_background_tab_shows_status(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    fake.background = BackgroundServiceStatus(
        state="not_installed",
        host_type="windows_task",
        detail="未安装计划任务 worker",
        executable="haagent",
    )
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="background")
            await pilot.pause()
            assert any(isinstance(s, SchedulesOverlay) for s in app.screen_stack)
            body = _all_text(app)
            assert "后台" in body or "未安装" in body or "windows" in body.lower()

    asyncio.run(_run())


def test_tui_install_requires_confirm_default_cancel(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    fake.background = BackgroundServiceStatus(
        state="not_installed",
        host_type="windows_task",
        detail="未安装",
        executable="haagent",
    )
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="background")
            await pilot.pause()
            await pilot.press("i")
            await pilot.pause()
            assert any(isinstance(s, ConfirmModal) for s in app.screen_stack)
            modal = next(s for s in app.screen_stack if isinstance(s, ConfirmModal))
            focused = modal.focused
            assert focused is not None
            assert focused.id == "confirm-no"
            await pilot.press("escape")
            await pilot.pause()
            assert fake.install_count == 0

    asyncio.run(_run())


def test_tui_install_confirm_calls_service(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    fake.background = BackgroundServiceStatus(
        state="not_installed",
        host_type="windows_task",
        detail="未安装",
        executable="haagent",
    )
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="background")
            await pilot.pause()
            await pilot.press("i")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause(0.2)
            assert fake.install_count >= 1

    asyncio.run(_run())


def test_tui_uninstall_confirm(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    fake.background = BackgroundServiceStatus(
        state="running",
        host_type="windows_task",
        detail="运行中",
        executable="haagent",
        last_heartbeat_utc=datetime(2026, 7, 12, 0, 0, tzinfo=timezone.utc),
    )
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="background")
            await pilot.pause()
            await pilot.press("u")
            await pilot.pause()
            assert any(isinstance(s, ConfirmModal) for s in app.screen_stack)
            await pilot.press("y")
            await pilot.pause(0.2)
            assert fake.uninstall_count >= 1

    asyncio.run(_run())
