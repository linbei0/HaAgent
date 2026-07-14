"""
tests/tui/test_schedule_host_worker.py - TUI 内嵌调度 host 接线

验证 TUI mount/unmount 启停 host 与 badge timer；worker 行为由 integration 测试覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace

from haagent.tui.application.schedule_flow import ScheduleFlow
from tests.tui.support import FakeAssistantService


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

    def run_worker(self, callback, **_kwargs) -> object:
        callback()
        return SimpleNamespace()

    def call_from_thread(self, callback, *args) -> None:
        callback(*args)

    def _refresh(self) -> None:
        return None


def test_schedule_flow_starts_and_stops_host_with_badge_timer(tmp_path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    app = _AppStub(service)
    flow = ScheduleFlow(app)

    flow.start_background_polling()

    assert service.schedules.host_start_count == 1
    assert service.schedules.host_status().running is True
    assert len(app._intervals) == 1
    flow.stop_background_polling()

    assert service.schedules.host_stop_count == 1
    assert service.schedules.host_status().running is False
    assert app._intervals[0].stopped is True
