"""
haagent/tui/overlays/schedules.py - 计划任务管理 overlay shell

宽屏三栏（列表/详情/最近运行），窄屏 tabs：计划 | 运行 | 后台服务。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from textual import events
from textual.app import ComposeResult, ScreenStackError
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.tui.design.utils import safe_summary, workspace_label
from haagent.tui.overlays.schedule_background import ScheduleBackgroundState
from haagent.tui.overlays.schedule_runs import ScheduleRunsState, filter_runs

SchedulesTab = Literal["plans", "runs", "background"]

# 计划列表可视窗口行数（长列表滚动，不全量渲染）
VISIBLE_SCHEDULE_COUNT = 12

SchedulesOverlayAction = Literal[
    "create",
    "edit",
    "pause",
    "resume",
    "archive",
    "delete",
    "run_now",
    "duplicate",
    "open_run",
    "mark_run_read",
    "mark_all_read",
    "cancel_run",
    "rerun",
    "open_session",
    "install_background",
    "uninstall_background",
    "refresh_background",
    "switch_tab",
    "close",
]


@dataclass(frozen=True)
class SchedulesOverlayResult:
    action: SchedulesOverlayAction
    schedule_id: str | None = None
    run_id: str | None = None
    tab: SchedulesTab | None = None


def _rule_summary(rrule: str | None) -> str:
    if not rrule:
        return "一次"
    upper = rrule.upper()
    if "FREQ=DAILY" in upper:
        return "每天"
    if "FREQ=WEEKLY" in upper:
        return "每周"
    if "FREQ=MONTHLY" in upper:
        return "每月"
    if "FREQ=MINUTELY" in upper:
        return "间隔(分)"
    if "FREQ=HOURLY" in upper:
        return "间隔(时)"
    return safe_summary(rrule, 18)


def _fmt_dt(value: object | None) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text.replace("+00:00", "Z")[:16]


@dataclass(frozen=True)
class SchedulesOverlayState:
    schedules: list[Any]
    runs: list[Any]
    selected_index: int = 0
    scroll_offset: int = 0
    wide: bool = True
    tab: SchedulesTab = "plans"
    detail: Any | None = None
    run_state: ScheduleRunsState | None = None
    background: ScheduleBackgroundState | None = None
    message: str = ""

    @property
    def selected_schedule(self) -> Any | None:
        if not self.schedules:
            return None
        index = min(max(self.selected_index, 0), len(self.schedules) - 1)
        return self.schedules[index]

    def move(self, delta: int) -> SchedulesOverlayState:
        if not self.schedules:
            return replace(self, selected_index=0, scroll_offset=0, detail=None)
        next_index = min(max(self.selected_index + delta, 0), len(self.schedules) - 1)
        # 右侧详情跟随当前选中项，避免一直显示第一项
        selected = self.schedules[next_index]
        # 可视窗口滚动：选中行始终落在 [scroll_offset, scroll_offset+VISIBLE)
        scroll_offset = self.scroll_offset
        if next_index < scroll_offset:
            scroll_offset = next_index
        elif next_index >= scroll_offset + VISIBLE_SCHEDULE_COUNT:
            scroll_offset = next_index - VISIBLE_SCHEDULE_COUNT + 1
        max_offset = max(len(self.schedules) - VISIBLE_SCHEDULE_COUNT, 0)
        scroll_offset = min(max(scroll_offset, 0), max_offset)
        return replace(
            self,
            selected_index=next_index,
            scroll_offset=scroll_offset,
            detail=selected,
        )

    def with_tab(self, tab: SchedulesTab) -> SchedulesOverlayState:
        return replace(self, tab=tab)

    def render(self) -> str:
        # 宽屏从收件箱/后台入口进入时，仍展示对应面板，避免键位与内容脱节。
        if self.tab == "runs":
            return self._render_narrow()
        if self.tab == "background":
            return self._render_narrow()
        if not self.wide:
            return self._render_narrow()
        return self._render_wide()

    def _render_narrow(self) -> str:
        tabs = [
            ("plans", "计划"),
            ("runs", "运行"),
            ("background", "后台服务"),
        ]
        header = " | ".join(f"[{label}]" if key == self.tab else label for key, label in tabs)
        lines = ["计划任务", header, ""]
        if self.tab == "plans":
            lines.extend(self._list_lines())
            lines.extend(self._detail_lines())
            lines.extend(
                [
                    "",
                    "↑/↓  n新建  e编辑  y复制  p暂停  r恢复  a归档  x删除  !立即运行",
                    "Tab 切换页  Esc 关闭",
                ]
            )
        elif self.tab == "runs":
            run_state = self.run_state or ScheduleRunsState(runs=self.runs)
            lines.append(run_state.render())
        else:
            bg = self.background or ScheduleBackgroundState()
            lines.append(bg.render())
        if self.message:
            lines.extend(["", self.message])
        return "\n".join(lines)

    def _render_wide(self) -> str:
        lines = ["计划任务（列表 | 详情 | 最近运行）", ""]
        list_lines = self._list_lines()
        detail_lines = self._detail_lines()
        recent = filter_runs(self.runs, "all")[:8]
        run_lines = ["最近运行:"]
        if not recent:
            run_lines.append("  （暂无）")
        for run in recent:
            unread = "*" if getattr(run, "unread", False) else " "
            run_lines.append(
                f"  {unread}{getattr(run, 'id', '?')[:10]} "
                f"{getattr(run, 'status', '?')} "
                f"{safe_summary(str(getattr(run, 'summary', '') or ''), 24)}"
            )
        # 三栏用分隔拼接，窄宽都能读
        lines.append("── 列表 ──")
        lines.extend(list_lines)
        lines.append("")
        lines.append("── 详情 ──")
        lines.extend(detail_lines)
        lines.append("")
        lines.extend(run_lines)
        lines.extend(
            [
                "",
                "↑/↓  n新建  e编辑  y复制  p暂停  r恢复  a归档  x删除  !立即运行",
                "t运行收件箱  b后台服务  Esc 关闭计划任务",
            ]
        )
        if self.message:
            lines.extend(["", self.message])
        return "\n".join(lines)

    def _list_lines(self) -> list[str]:
        if not self.schedules:
            return ["（暂无计划）按 n 创建"]
        total = len(self.schedules)
        scroll_offset = min(self.scroll_offset, max(total - VISIBLE_SCHEDULE_COUNT, 0))
        window = self.schedules[scroll_offset : scroll_offset + VISIBLE_SCHEDULE_COUNT]
        lines: list[str] = []
        if total > VISIBLE_SCHEDULE_COUNT:
            lines.append(
                f"计划 {min(self.selected_index, total - 1) + 1}/{total}  "
                f"（↑/↓ 滚动窗口）"
            )
        for offset, item in enumerate(window):
            index = scroll_offset + offset
            marker = ">" if index == min(self.selected_index, total - 1) else " "
            name = safe_summary(str(getattr(item, "name", "?")), 16)
            status = str(getattr(item, "status", "?"))
            rule = _rule_summary(getattr(item, "rrule", None))
            next_run = _fmt_dt(getattr(item, "next_run_at_utc", None))
            last_run = _fmt_dt(getattr(item, "last_run_at_utc", None))
            root = getattr(item, "workspace_root", None)
            if isinstance(root, Path):
                ws = workspace_label(root, 12)
            else:
                ws = safe_summary(str(root or "?"), 12)
            lines.append(
                f"{marker} {name:<16} {status:<9} {rule:<8} next:{next_run} last:{last_run} ws:{ws}"
            )
        return lines

    def _detail_lines(self) -> list[str]:
        detail = self.detail or self.selected_schedule
        if detail is None:
            return ["（未选择计划）"]
        lines = [
            f"名称: {getattr(detail, 'name', '-')}",
            f"状态: {getattr(detail, 'status', '-')}",
            f"规则: {_rule_summary(getattr(detail, 'rrule', None))} ({getattr(detail, 'rrule', None) or 'once'})",
            f"时区: {getattr(detail, 'timezone', '-')}",
            f"Workspace: {getattr(detail, 'workspace_root', '-')}",
            f"模型: {getattr(detail, 'connection_id', '-')}/{getattr(detail, 'model', '-')}",
            f"下次: {_fmt_dt(getattr(detail, 'next_run_at_utc', None))}",
            f"上次: {_fmt_dt(getattr(detail, 'last_run_at_utc', None))}",
        ]
        prompt = getattr(detail, "prompt", None)
        if prompt:
            lines.append(f"Prompt: {safe_summary(str(prompt), 70)}")
        return lines


class SchedulesOverlay(ModalScreen[SchedulesOverlayResult | None]):
    def __init__(self, state: SchedulesOverlayState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        # 窄屏/80x24：内容可滚动，避免 footer 被裁切
        with VerticalScroll(id="schedules-scroll"):
            yield Static(self.state.render(), id="schedules-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        ch = event.character
        if key == "escape":
            event.stop()
            self._handle_escape()
            return
        # 窄屏 tab 切换：原地切换，避免 dismiss/repush 导致 screen stack 错乱
        if key == "tab" and not self.state.wide:
            event.stop()
            order: list[SchedulesTab] = ["plans", "runs", "background"]
            idx = order.index(self.state.tab) if self.state.tab in order else 0
            self._switch_tab(order[(idx + 1) % len(order)])
            return
        if ch == "t" and self.state.wide:
            event.stop()
            self._switch_tab("runs")
            return
        if ch == "b" and self.state.wide:
            event.stop()
            self._switch_tab("background")
            return

        if self.state.tab == "runs":
            self._handle_runs_key(event)
            return
        if self.state.tab == "background":
            self._handle_background_key(event)
            return
        self._handle_plans_key(event)

    def _handle_escape(self) -> None:
        # 详情 → 列表 → 计划页 → 关闭 overlay
        run_state = self.state.run_state
        if self.state.tab == "runs" and run_state is not None and run_state.detail_mode:
            self._set_state(replace(self.state, run_state=run_state.close_detail()))
            return
        if self.state.tab in {"runs", "background"}:
            self._switch_tab("plans")
            return
        self._safe_dismiss(None)

    def _switch_tab(self, tab: SchedulesTab) -> None:
        run_state = self.state.run_state or ScheduleRunsState(runs=self.state.runs)
        if tab != "runs":
            run_state = run_state.close_detail()
        self._set_state(replace(self.state.with_tab(tab), run_state=run_state))

    def _handle_plans_key(self, event: events.Key) -> None:
        key = event.key
        ch = event.character
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if ch == "n":
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="create"))
            return
        selected = self.state.selected_schedule
        sid = str(getattr(selected, "id", "")) if selected is not None else None
        if ch == "e" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="edit", schedule_id=sid))
            return
        if ch == "y" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="duplicate", schedule_id=sid))
            return
        if ch == "p" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="pause", schedule_id=sid))
            return
        if ch == "r" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="resume", schedule_id=sid))
            return
        if ch == "a" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="archive", schedule_id=sid))
            return
        if ch == "x" and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="delete", schedule_id=sid))
            return
        if (ch == "!" or key == "exclamation_mark") and sid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="run_now", schedule_id=sid))
            return

    def _handle_runs_key(self, event: events.Key) -> None:
        key = event.key
        ch = event.character
        run_state = self.state.run_state or ScheduleRunsState(runs=self.state.runs)
        if key == "up":
            event.stop()
            self._set_state(replace(self.state, run_state=run_state.move(-1)))
            return
        if key == "down":
            event.stop()
            self._set_state(replace(self.state, run_state=run_state.move(1)))
            return
        if ch in {"1", "2", "3", "4", "5"} and not run_state.detail_mode:
            event.stop()
            modes = ["unread", "attention", "failed", "succeeded", "all"]
            mode = modes[int(ch) - 1]
            self._set_state(
                replace(self.state, run_state=run_state.with_filter(mode))  # type: ignore[arg-type]
            )
            return
        selected = run_state.selected
        rid = str(getattr(selected, "id", "")) if selected is not None else None
        sid = str(getattr(selected, "schedule_id", "")) if selected is not None else None
        if key == "enter" and rid:
            event.stop()
            # 原地打开完整详情，并标记已读；不 dismiss，避免 stack 错乱
            self._mark_run_read_inplace(rid)
            refreshed = self.state.run_state or run_state
            self._set_state(replace(self.state, run_state=refreshed.open_detail()))
            return
        if ch == "m" and rid:
            event.stop()
            self._mark_run_read_inplace(rid)
            return
        if ch == "M":
            event.stop()
            self._mark_all_read_inplace()
            return
        if ch == "o" and rid:
            event.stop()
            self._safe_dismiss(
                SchedulesOverlayResult(action="open_session", run_id=rid, schedule_id=sid)
            )
            return
        if ch == "c" and rid:
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="cancel_run", run_id=rid))
            return
        if ch == "R" and sid:
            event.stop()
            self._safe_dismiss(
                SchedulesOverlayResult(action="rerun", schedule_id=sid, run_id=rid)
            )
            return

    def _handle_background_key(self, event: events.Key) -> None:
        ch = event.character
        if ch == "i":
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="install_background"))
            return
        if ch == "u":
            event.stop()
            self._safe_dismiss(SchedulesOverlayResult(action="uninstall_background"))
            return
        if ch == "d":
            event.stop()
            # 原地刷新后台状态
            self._refresh_background_inplace()
            return

    def _mark_run_read_inplace(self, run_id: str) -> None:
        schedules = getattr(getattr(self.app, "service", None), "schedules", None)
        if schedules is None:
            return
        try:
            schedules.mark_run_read(run_id)
        except Exception:
            return
        run_state = self.state.run_state or ScheduleRunsState(runs=self.state.runs)
        updated: list[Any] = []
        for run in run_state.runs:
            if str(getattr(run, "id", "")) == run_id and hasattr(run, "__dataclass_fields__"):
                try:
                    updated.append(replace(run, unread=False))
                    continue
                except Exception:
                    pass
            if str(getattr(run, "id", "")) == run_id:
                try:
                    object.__setattr__(run, "unread", False)
                except Exception:
                    pass
            updated.append(run)
        self._set_state(
            replace(
                self.state,
                runs=updated,
                run_state=run_state.with_runs(updated),
            )
        )
        refresh_badge = getattr(getattr(self.app, "schedule_flow", None), "refresh_badge", None)
        if callable(refresh_badge):
            try:
                refresh_badge()
            except Exception:
                pass

    def _mark_all_read_inplace(self) -> None:
        schedules = getattr(getattr(self.app, "service", None), "schedules", None)
        if schedules is None:
            return
        try:
            schedules.mark_all_runs_read()
        except Exception:
            return
        run_state = self.state.run_state or ScheduleRunsState(runs=self.state.runs)
        updated: list[Any] = []
        for run in run_state.runs:
            if hasattr(run, "__dataclass_fields__"):
                try:
                    updated.append(replace(run, unread=False))
                    continue
                except Exception:
                    pass
            try:
                object.__setattr__(run, "unread", False)
            except Exception:
                pass
            updated.append(run)
        self._set_state(
            replace(
                self.state,
                runs=updated,
                run_state=run_state.with_runs(updated),
            )
        )
        refresh_badge = getattr(getattr(self.app, "schedule_flow", None), "refresh_badge", None)
        if callable(refresh_badge):
            try:
                refresh_badge()
            except Exception:
                pass

    def _refresh_background_inplace(self) -> None:
        schedules = getattr(getattr(self.app, "service", None), "schedules", None)
        if schedules is None:
            return
        try:
            status = schedules.background_status()
        except Exception:
            return
        host = None
        try:
            if hasattr(schedules, "host_status"):
                host = schedules.host_status()
        except Exception:
            host = None
        bg = ScheduleBackgroundState(status=status, host=host)
        self._set_state(replace(self.state, background=bg))

    def _safe_dismiss(self, result: SchedulesOverlayResult | None) -> None:
        # 焦点恢复/重复 Esc：仅在当前 screen 是自己时 dismiss，吞掉 stack 错误
        try:
            if self.app.screen is not self:
                return
            self.dismiss(result)
        except ScreenStackError:
            return

    def _set_state(self, state: SchedulesOverlayState) -> None:
        self.state = state
        try:
            self.query_one("#schedules-dialog", Static).update(state.render())
        except NoMatches:
            return
