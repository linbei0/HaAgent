"""
tests/tui/test_schedules.py - 计划任务 TUI overlay 与 /schedules

覆盖 slash 注册、宽/窄屏布局、列表动作、四步编辑、预览与空状态。
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from haagent.app.assistant_types import (
    AssistantSchedule,
    AssistantScheduleRun,
    AssistantScheduleSummary,
    BackgroundServiceStatus,
    RunQuery,
    ScheduleCreateRequest,
    SchedulePreviewRequest,
)
from haagent.scheduling.models import RetryPolicy
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.commands import SlashCommandResult, command_registry, parse_slash_command
from haagent.tui.overlays.schedule_editor import ScheduleEditorOverlay, ScheduleEditorState
from haagent.tui.overlays.schedules import SchedulesOverlay, SchedulesOverlayState
from tests.tui.support import FakeAssistantService, FakeSchedules, _all_text


def _summary(
    schedule_id: str = "sch_1",
    *,
    name: str = "日报",
    status: str = "active",
    rrule: str | None = "FREQ=DAILY",
    workspace: Path | None = None,
) -> AssistantScheduleSummary:
    root = workspace or Path("E:/ws")
    return AssistantScheduleSummary(
        id=schedule_id,
        name=name,
        status=status,  # type: ignore[arg-type]
        timezone="Asia/Shanghai",
        rrule=rrule,
        next_run_at_utc=datetime(2026, 7, 13, 1, 0, tzinfo=timezone.utc),
        last_run_at_utc=datetime(2026, 7, 12, 1, 0, tzinfo=timezone.utc),
        workspace_root=root,
        revision=1,
        model="deepseek-chat",
        connection_id="local",
    )


def _full(
    schedule_id: str = "sch_1",
    *,
    name: str = "日报",
    workspace: Path | None = None,
) -> AssistantSchedule:
    root = workspace or Path("E:/ws")
    return AssistantSchedule(
        id=schedule_id,
        name=name,
        prompt="整理今日笔记",
        workspace_root=root,
        destination_kind="new_session",
        destination_session_path=None,
        connection_id="local",
        model="deepseek-chat",
        web_enabled=False,
        allowed_tools=("file_read", "file_list", "grep"),
        approval_allowed_tools=(),
        approved_tools=(),
        permission_mode="request_approval",
        dtstart_local=datetime(2026, 7, 12, 9, 0),
        timezone="Asia/Shanghai",
        rrule="FREQ=DAILY",
        status="active",
        misfire_policy="latest",
        overlap_policy="skip",
        retry_policy=RetryPolicy(),
        revision=1,
        next_run_at_utc=datetime(2026, 7, 13, 1, 0, tzinfo=timezone.utc),
        last_run_at_utc=None,
    )


def test_schedules_is_structured_slash_command() -> None:
    registry = command_registry()
    result = parse_slash_command("/schedules", registry)
    assert result is not None
    assert result.error is None
    assert result.command is not None
    assert result.command.name == "schedules"
    assert result.command.action == "open_schedules"
    assert result == SlashCommandResult(command=registry.require("schedules"), argument="")


def test_schedules_overlay_empty_state_and_list_fields(tmp_path: Path) -> None:
    empty = SchedulesOverlayState(schedules=[], runs=[], wide=True)
    text = empty.render()
    assert "计划任务" in text
    assert "暂无" in text or "空" in text

    item = _summary(workspace=tmp_path)
    state = SchedulesOverlayState(
        schedules=[item],
        runs=[],
        selected_index=0,
        wide=True,
        detail=_full(workspace=tmp_path),
    )
    text = state.render()
    assert "日报" in text
    assert "active" in text or "运行中" in text or "启用" in text
    assert tmp_path.name in text
    assert "FREQ=DAILY" in text or "每天" in text or "日" in text


def test_schedules_overlay_narrow_tabs() -> None:
    state = SchedulesOverlayState(schedules=[], runs=[], wide=False, tab="plans")
    text = state.render()
    assert "计划" in text
    assert "运行" in text
    assert "后台" in text


def test_schedule_editor_four_pages_and_preview() -> None:
    state = ScheduleEditorState(page=0)
    text = state.render()
    assert "任务" in text
    # 任务页应引导用户真正输入名称/prompt，而不是按键追加假字
    assert "n 名称" in text or "输入名称" in text or "名称" in text
    assert "p 任务" in text or "prompt" in text.casefold() or "任务内容" in text
    state = state.with_page(1)
    assert "计划" in state.render()
    assert "时区" in state.render() or "频率" in state.render()
    state = state.with_page(2)
    assert "执行" in state.render() or "模型" in state.render() or "工具" in state.render()
    state = state.with_page(3)
    assert "确认" in state.render()

    previews = (
        datetime(2026, 7, 13, 1, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 14, 1, 0, tzinfo=timezone.utc),
        datetime(2026, 7, 15, 1, 0, tzinfo=timezone.utc),
    )
    with_preview = state.with_previews(previews)
    text = with_preview.render()
    assert "2026" in text


def test_schedule_editor_n_p_open_real_text_input(tmp_path: Path) -> None:
    """n/p 必须进入可编辑输入框，不能静默追加「计划」「执行计划任务」。"""
    service = FakeAssistantService(workspace_root=tmp_path)
    service.schedules = FakeSchedules(service)
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            editor = ScheduleEditorOverlay(
                ScheduleEditorState(page=0, workspace_root=str(tmp_path))
            )
            app.push_screen(editor)
            await pilot.pause()

            await pilot.press("n")
            await pilot.pause()
            # 应出现输入框；名称不应被静默改成「计划」
            assert editor.state.name == ""
            assert editor.input_mode in {"name", "prompt"}
            assert editor.query_one("#schedule-editor-input").display is True

            await pilot.press(*list("日报整理"))
            await pilot.press("enter")
            await pilot.pause()
            assert editor.state.name == "日报整理"
            assert editor.input_mode is None

            await pilot.press("p")
            await pilot.pause()
            assert editor.input_mode == "prompt"
            await pilot.press(*list("只读列出当前目录文件"))
            await pilot.press("enter")
            await pilot.pause()
            assert "只读列出" in editor.state.prompt
            assert editor.state.prompt != "执行计划任务"

    asyncio.run(_run())


def test_schedule_editor_frequency_forms_build_rrule() -> None:
    once = ScheduleEditorState(page=1, frequency="once")
    assert once.build_rrule() is None
    daily = ScheduleEditorState(page=1, frequency="daily")
    assert "DAILY" in (daily.build_rrule() or "")
    weekly = ScheduleEditorState(page=1, frequency="weekly", byday="MO,WE,FR")
    assert "WEEKLY" in (weekly.build_rrule() or "")
    monthly = ScheduleEditorState(page=1, frequency="monthly", bymonthday=15)
    assert "MONTHLY" in (monthly.build_rrule() or "")
    interval = ScheduleEditorState(page=1, frequency="interval", interval_value=30, interval_unit="minutes")
    rrule = interval.build_rrule() or ""
    assert "INTERVAL=30" in rrule
    custom = ScheduleEditorState(page=1, frequency="custom", custom_rrule="FREQ=WEEKLY;BYDAY=MO")
    assert custom.build_rrule() == "FREQ=WEEKLY;BYDAY=MO"


def test_schedules_overlay_long_text_and_no_color(tmp_path: Path) -> None:
    long_name = "很长计划名称" * 20
    long_prompt = "prompt " * 80
    item = _summary(name=long_name, workspace=tmp_path)
    detail = _full(name=long_name, workspace=tmp_path)
    detail = SimpleNamespace(**{**detail.__dict__, "prompt": long_prompt})
    state = SchedulesOverlayState(
        schedules=[item],
        runs=[],
        detail=detail,
        wide=True,
    )
    text = state.render()
    assert "..." in text or "[truncated]" in text or long_name[:20] in text
    old = os.environ.get("NO_COLOR")
    os.environ["NO_COLOR"] = "1"
    try:
        text2 = state.render()
        assert "日报" in text2 or long_name[:8] in text2 or "计划" in text2
    finally:
        if old is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = old


def test_tui_schedules_command_opens_overlay_120x40(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.schedules = FakeSchedules(service)
    service.schedule_summaries = [_summary(workspace=tmp_path)]
    service.schedule_details = {"sch_1": _full(workspace=tmp_path)}
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            for ch in "schedules":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert any(isinstance(screen, SchedulesOverlay) for screen in app.screen_stack)
            body = _all_text(app)
            assert "计划" in body or "日报" in body
            await pilot.press("escape")
            await pilot.pause()
            assert not any(isinstance(screen, SchedulesOverlay) for screen in app.screen_stack)

    asyncio.run(_run())


def test_tui_schedules_command_opens_overlay_80x24(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.schedules = FakeSchedules(service)
    service.schedule_summaries = [_summary(workspace=tmp_path)]
    service.schedule_details = {"sch_1": _full(workspace=tmp_path)}
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("/")
            for ch in "schedules":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert any(isinstance(screen, SchedulesOverlay) for screen in app.screen_stack)
            body = _all_text(app)
            assert "计划" in body or "日报" in body or "运行" in body

    asyncio.run(_run())


def test_tui_schedules_pause_resume_run_now(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_summaries = [_summary(workspace=tmp_path)]
    service.schedule_details = {"sch_1": _full(workspace=tmp_path)}
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules()
            await pilot.pause()
            await pilot.press("p")
            await pilot.pause()
            assert "sch_1" in fake.paused
            app.schedule_flow.open_schedules()
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            assert "sch_1" in fake.resumed
            app.schedule_flow.open_schedules()
            await pilot.pause()
            await pilot.press("!")
            await pilot.pause()
            assert "sch_1" in fake.run_now_ids

    asyncio.run(_run())


def test_tui_create_editor_four_steps_and_preview(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_summaries = []
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.open_schedules()
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert any(isinstance(screen, ScheduleEditorOverlay) for screen in app.screen_stack)
            editor = next(s for s in app.screen_stack if isinstance(s, ScheduleEditorOverlay))
            editor.state = editor.state.with_field("name", "测试计划").with_field(
                "prompt", "写日报"
            )
            await pilot.press("tab")
            await pilot.pause()
            editor.state = editor.state.with_field("frequency", "daily").with_field(
                "timezone", "Asia/Shanghai"
            )
            # 预览
            await pilot.press("v")
            await pilot.pause()
            assert fake.preview_calls >= 1
            await pilot.press("tab")
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            body = _all_text(app)
            assert "确认" in body or "测试计划" in body or "写日报" in body

    asyncio.run(_run())


def test_status_bar_badge_shows_unread(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    fake = FakeSchedules(service)
    service.schedules = fake
    service.schedule_runs = [
        AssistantScheduleRun(
            id="run_1",
            schedule_id="sch_1",
            schedule_revision=1,
            trigger_key="k1",
            trigger_kind="scheduled",
            scheduled_for_utc=datetime(2026, 7, 12, 1, 0, tzinfo=timezone.utc),
            status="succeeded",
            attempt_count=1,
            summary="ok",
            unread=True,
        )
    ]
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test(size=(120, 40)) as pilot:
            app.schedule_flow.refresh_badge()
            await pilot.pause()
            app._refresh()
            await pilot.pause()
            body = _all_text(app)
            assert "计划任务" in body
            assert "1" in body

    asyncio.run(_run())


def test_help_includes_schedules() -> None:
    from haagent.tui.design.keys import help_body

    body = help_body("chat")
    assert "/schedules" in body
