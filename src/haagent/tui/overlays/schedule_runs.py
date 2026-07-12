"""
haagent/tui/overlays/schedule_runs.py - 计划运行收件箱状态与渲染

过滤未读/需要关注/失败/成功/全部；展示摘要、失败原因与动作提示。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from haagent.tui.design.utils import safe_summary

RunFilterMode = Literal["unread", "attention", "failed", "succeeded", "all"]

FILTER_LABELS: dict[RunFilterMode, str] = {
    "unread": "未读",
    "attention": "需要关注",
    "failed": "失败",
    "succeeded": "成功",
    "all": "全部",
}

FILTER_KEYS: tuple[RunFilterMode, ...] = (
    "unread",
    "attention",
    "failed",
    "succeeded",
    "all",
)


def filter_runs(runs: list[Any], mode: RunFilterMode) -> list[Any]:
    if mode == "unread":
        return [r for r in runs if getattr(r, "unread", False)]
    if mode == "attention":
        return [r for r in runs if getattr(r, "status", "") == "needs_attention"]
    if mode == "failed":
        return [r for r in runs if getattr(r, "status", "") == "failed"]
    if mode == "succeeded":
        return [r for r in runs if getattr(r, "status", "") == "succeeded"]
    return list(runs)


def _fmt_dt(value: object | None) -> str:
    if value is None:
        return "-"
    text = str(value)
    if "T" in text:
        return text.replace("+00:00", "Z")[:19]
    return text[:19]


def _wrap_text(text: str, *, width: int = 72) -> list[str]:
    """按宽度折行，保留完整摘要供详情页阅读。"""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return ["-"]
    lines: list[str] = []
    for paragraph in raw.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = paragraph
        while len(current) > width:
            lines.append(current[:width])
            current = current[width:]
        lines.append(current)
    return lines


@dataclass(frozen=True)
class ScheduleRunsState:
    runs: list[Any]
    filter_mode: RunFilterMode = "all"
    selected_index: int = 0
    detail_mode: bool = False

    @property
    def visible(self) -> list[Any]:
        return filter_runs(self.runs, self.filter_mode)

    @property
    def selected(self) -> Any | None:
        items = self.visible
        if not items:
            return None
        index = min(max(self.selected_index, 0), len(items) - 1)
        return items[index]

    def with_filter(self, mode: RunFilterMode) -> ScheduleRunsState:
        return replace(self, filter_mode=mode, selected_index=0, detail_mode=False)

    def move(self, delta: int) -> ScheduleRunsState:
        if self.detail_mode:
            return self
        items = self.visible
        if not items:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(items) - 1)
        return replace(self, selected_index=next_index)

    def open_detail(self) -> ScheduleRunsState:
        if self.selected is None:
            return self
        return replace(self, detail_mode=True)

    def close_detail(self) -> ScheduleRunsState:
        return replace(self, detail_mode=False)

    def with_runs(self, runs: list[Any]) -> ScheduleRunsState:
        return replace(self, runs=list(runs))

    def render(self) -> str:
        if self.detail_mode and self.selected is not None:
            return self._render_detail(self.selected)
        lines = [
            "运行收件箱",
            "过滤: "
            + " | ".join(
                f"[{FILTER_LABELS[m]}]" if m == self.filter_mode else FILTER_LABELS[m]
                for m in FILTER_KEYS
            ),
            "",
        ]
        items = self.visible
        if not items:
            lines.append("（暂无运行记录）")
        else:
            for index, run in enumerate(items):
                marker = ">" if index == min(self.selected_index, len(items) - 1) else " "
                unread = "*" if getattr(run, "unread", False) else " "
                status = str(getattr(run, "status", "?"))
                summary = safe_summary(str(getattr(run, "summary", "") or "-"), 36)
                lines.append(
                    f"{marker}{unread} {getattr(run, 'id', '?'):<12} {status:<14} "
                    f"{_fmt_dt(getattr(run, 'scheduled_for_utc', None))}  {summary}"
                )
            selected = self.selected
            if selected is not None:
                lines.extend(["", "── 详情预览 ──"])
                lines.append(f"计划: {getattr(selected, 'schedule_id', '-')}")
                lines.append(f"状态: {getattr(selected, 'status', '-')}")
                lines.append(
                    f"摘要: {safe_summary(str(getattr(selected, 'summary', '') or '-'), 80)}"
                )
                lines.append("（按 Enter 打开完整详情）")
        lines.extend(
            [
                "",
                "1未读 2需关注 3失败 4成功 5全部  ↑/↓  m标为已读  M全部已读",
                "Enter打开详情  o打开会话  R重新运行  c取消  Esc返回计划列表",
            ]
        )
        return "\n".join(lines)

    def _render_detail(self, selected: Any) -> str:
        lines = [
            "运行详情",
            f"run: {getattr(selected, 'id', '-')}",
            f"计划: {getattr(selected, 'schedule_id', '-')}",
            f"revision: {getattr(selected, 'schedule_revision', '-')}",
            f"触发: {getattr(selected, 'trigger_kind', '-')} / {getattr(selected, 'trigger_key', '-')}",
            (
                f"时间: 计划 {_fmt_dt(getattr(selected, 'scheduled_for_utc', None))}  "
                f"开始 {_fmt_dt(getattr(selected, 'started_at_utc', None))}  "
                f"结束 {_fmt_dt(getattr(selected, 'finished_at_utc', None))}"
            ),
            f"尝试: {getattr(selected, 'attempt_count', 0)}",
            f"状态: {getattr(selected, 'status', '-')}",
            f"未读: {'是' if getattr(selected, 'unread', False) else '否'}",
        ]
        cat = getattr(selected, "failure_category", None)
        if cat:
            lines.append(f"类别: {cat}")
        fail = getattr(selected, "failure_reason", None)
        if fail:
            lines.append("失败:")
            lines.extend(f"  {part}" for part in _wrap_text(str(fail)))
        need = getattr(selected, "needs_attention_reason", None)
        if need:
            lines.append("需关注:")
            lines.extend(f"  {part}" for part in _wrap_text(str(need)))
        lines.append("摘要:")
        lines.extend(f"  {part}" for part in _wrap_text(str(getattr(selected, "summary", "") or "-")))
        session_path = getattr(selected, "session_path", None)
        episode_path = getattr(selected, "episode_path", None)
        if session_path:
            lines.append(f"会话: {session_path}")
        if episode_path:
            lines.append(f"episode: {episode_path}")
        lines.extend(
            [
                "",
                "Esc返回列表  o打开会话  R重新运行  c取消  m标为已读",
            ]
        )
        return "\n".join(lines)
