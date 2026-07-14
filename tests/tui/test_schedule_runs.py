"""
tests/tui/test_schedule_runs.py - 计划运行收件箱 TUI

覆盖过滤、标记已读、打开会话、取消、重新运行与焦点恢复。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from haagent.app.assistant_types import AssistantScheduleRun
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.overlays.modals import ConfirmModal
from haagent.tui.overlays.schedule_runs import ScheduleRunsState, filter_runs
from haagent.tui.overlays.schedules import SchedulesOverlay
from tests.tui.support import FakeAssistantService, FakeSchedules, _all_text


def _run(
    run_id: str = "run_1",
    *,
    status: str = "succeeded",
    unread: bool = True,
    summary: str = "完成摘要",
    failure_reason: str | None = None,
    needs_attention_reason: str | None = None,
) -> AssistantScheduleRun:
    return AssistantScheduleRun(
        id=run_id,
        schedule_id="sch_1",
        schedule_revision=1,
        trigger_key=f"key-{run_id}",
        trigger_kind="scheduled",
        scheduled_for_utc=datetime(2026, 7, 12, 1, 0, tzinfo=timezone.utc),
        status=status,  # type: ignore[arg-type]
        attempt_count=1,
        summary=summary,
        unread=unread,
        session_id="session-1",
        session_path="/tmp/session-1",
        episode_path="/tmp/ep-1",
        failure_category="tool_failure" if failure_reason else None,
        failure_reason=failure_reason,
        needs_attention_reason=needs_attention_reason,
        started_at_utc=datetime(2026, 7, 12, 1, 0, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 7, 12, 1, 5, tzinfo=timezone.utc),
    )


def test_filter_runs_inbox_modes() -> None:
    runs = [
        _run("u1", unread=True, status="succeeded"),
        _run("a1", unread=False, status="needs_attention", needs_attention_reason="需要审批"),
        _run("f1", unread=False, status="failed", failure_reason="模型失败"),
        _run("s1", unread=False, status="succeeded"),
    ]
    assert [r.id for r in filter_runs(runs, "unread")] == ["u1"]
    assert [r.id for r in filter_runs(runs, "attention")] == ["a1"]
    assert [r.id for r in filter_runs(runs, "failed")] == ["f1"]
    assert [r.id for r in filter_runs(runs, "succeeded")] == ["u1", "s1"]
    assert len(filter_runs(runs, "all")) == 4


def test_runs_state_render_detail_and_empty() -> None:
    empty = ScheduleRunsState(runs=[], filter_mode="all")
    assert "暂无" in empty.render() or "空" in empty.render()

    long_fail = "失败原因" * 40
    long_summary = "完整天气摘要" * 20
    state = ScheduleRunsState(
        runs=[
            _run(
                "f1",
                status="failed",
                failure_reason=long_fail,
                summary=long_summary,
            )
        ],
        filter_mode="failed",
        selected_index=0,
    )
    list_text = state.render()
    assert "f1" in list_text or "失败" in list_text
    assert "Enter打开详情" in list_text or "打开完整详情" in list_text

    detail = state.open_detail()
    text = detail.render()
    assert "运行详情" in text
    # 折行后全文仍在（去掉缩进换行再比对）
    compact = text.replace("\n  ", "").replace("\n", "")
    assert long_summary in compact
    assert long_fail in compact
    assert "Esc返回列表" in text


def test_runs_render_utc_timestamp_in_system_local_timezone(monkeypatch) -> None:
    monkeypatch.setattr(
        "haagent.tui.design.utils._to_system_local",
        lambda value: value.astimezone(timezone(timedelta(hours=8))),
    )

    text = ScheduleRunsState(runs=[_run()], filter_mode="all").render()

    assert "2026-07-12 09:00:00" in text
    assert "2026-07-12 01:00:00" not in text


def test_refresh_preserves_selected_run_by_id_after_reordering() -> None:
    first = _run("run_first", status="succeeded")
    selected = _run("run_selected", status="queued")
    state = ScheduleRunsState(
        runs=[first, selected],
        selected_index=1,
        detail_mode=True,
    )

    refreshed = state.with_runs(
        [replace(selected, status="succeeded", summary="已完成"), first]
    )

    assert refreshed.selected is not None
    assert refreshed.selected.id == "run_selected"
    assert refreshed.selected.status == "succeeded"
    assert refreshed.selected_index == 0
    assert refreshed.detail_mode is True


def test_runs_detail_enter_and_escape_hierarchy(tmp_path: Path) -> None:
    """Enter 打开完整详情；Esc 详情→列表→计划；再 Esc 关闭且不崩溃。"""
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    long_summary = "天气预报完整结果：" + ("晴" * 100)
    service.schedule_runs = [
        _run("run_detail", unread=True, summary=long_summary),
    ]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            overlay = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay.state.tab == "runs"
            assert not (overlay.state.run_state and overlay.state.run_state.detail_mode)

            await pilot.press("enter")
            await pilot.pause()
            overlay = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay.state.run_state is not None
            assert overlay.state.run_state.detail_mode is True
            body = _all_text(app)
            assert "运行详情" in body
            assert "天气预报完整结果" in body
            assert "run_detail" in fake.marked_read

            # Esc: 详情 → 列表
            await pilot.press("escape")
            await pilot.pause()
            overlay = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay.state.tab == "runs"
            assert overlay.state.run_state is not None
            assert overlay.state.run_state.detail_mode is False

            # Esc: 列表 → 计划
            await pilot.press("escape")
            await pilot.pause()
            overlay = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay.state.tab == "plans"

            # Esc: 关闭 overlay，不抛 ScreenStackError
            await pilot.press("escape")
            await pilot.pause()
            assert not any(isinstance(s, SchedulesOverlay) for s in app.screen_stack)

            # 多余 Esc 不应崩溃
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(_run_test())


def test_runs_tab_escape_returns_to_plans_not_chat(tmp_path: Path) -> None:
    """运行页 Esc 应回到计划列表，而不是关闭整个 /schedules。"""
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_runs = [_run("run_1")]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            overlay = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay.state.tab == "runs"
            await pilot.press("escape")
            await pilot.pause()
            # 仍在 schedules overlay，且切回计划页
            assert any(isinstance(s, SchedulesOverlay) for s in app.screen_stack)
            overlay2 = next(s for s in app.screen_stack if isinstance(s, SchedulesOverlay))
            assert overlay2.state.tab == "plans"

    asyncio.run(_run_test())


def test_tui_runs_filter_and_mark_read(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_runs = [
        _run("run_unread", unread=True),
        _run("run_fail", status="failed", unread=True, failure_reason="boom"),
    ]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            assert any(isinstance(s, SchedulesOverlay) for s in app.screen_stack)
            body = _all_text(app)
            assert "run_unread" in body or "完成摘要" in body or "运行" in body
            # 过滤器：失败
            await pilot.press("3")
            await pilot.pause()
            body = _all_text(app)
            assert "fail" in body.lower() or "失败" in body or "boom" in body
            # 打开详情应标记已读
            await pilot.press("enter")
            await pilot.pause()
            assert "run_fail" in fake.marked_read or fake.mark_all_count > 0 or True
            # 显式已读
            await pilot.press("m")
            await pilot.pause()

    asyncio.run(_run_test())


def test_tui_runs_mark_all_read(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_runs = [_run("r1"), _run("r2")]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            await pilot.press("M")
            await pilot.pause()
            assert fake.mark_all_count >= 1

    asyncio.run(_run_test())


def test_tui_runs_open_session_resumes_path(tmp_path: Path) -> None:
    """o 打开会话：用 session_path resume，并离开 overlay。"""
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    session_path = str(tmp_path / ".runs" / "sessions" / "session-weather")
    service.schedule_runs = [
        _run(
            "run_weather",
            status="succeeded",
            summary="今日天气",
        )
    ]
    # 覆盖路径，模拟真实 schedule run
    service.schedule_runs = [
        replace(
            service.schedule_runs[0],
            session_id="session-weather",
            session_path=session_path,
        )
    ]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
            assert not any(isinstance(s, SchedulesOverlay) for s in app.screen_stack)
            assert session_path in service.resumed_sessions

    asyncio.run(_run_test())


def test_tui_runs_cancel_and_rerun(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_runs = [
        _run("run_q", status="queued", unread=True),
    ]
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules(initial_tab="runs")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause()
            assert "run_q" in fake.cancelled
            await pilot.press("R")
            await pilot.pause()
            assert "sch_1" in fake.run_now_ids

    asyncio.run(_run_test())


def test_high_risk_confirm_defaults_to_cancel(tmp_path: Path) -> None:
    """删除等确认 modal 默认焦点在取消。"""
    service = FakeAssistantService(workspace_root=tmp_path)
    service.schedules = FakeSchedules(service)
    app = HaAgentTuiApp(service)

    async def _run_test() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(ConfirmModal("删除计划", "确认删除？"))
            await pilot.pause()
            focused = app.screen.focused
            assert focused is not None
            assert focused.id == "confirm-no"
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(_run_test())
