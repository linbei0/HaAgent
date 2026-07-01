"""
haagent/tui/tool_timeline.py - 工具时间线视图状态

把 ChatEvent 转换为可导航、可脱敏展示的工具调用时间线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from haagent.runtime.chat_session import ChatEvent
from haagent.tui.copy import MODAL_TITLES
from haagent.tui.theme import status_badge
from haagent.tui.utils import safe_summary


TOOL_EVENT_TYPES = {
    "tool_started",
    "tool_finished",
    "tool_failed",
    "approval_requested",
    "approval_granted",
    "approval_denied",
    "edit_diff_requested",
    "edit_diff_granted",
    "edit_diff_denied",
}


@dataclass
class ToolTimelineItem:
    tool_name: str
    status: str
    turn_index: int
    reason: str = ""
    args_summary: dict[str, object] = field(default_factory=dict)
    result_summary: dict[str, object] = field(default_factory=dict)
    impact: str = ""
    stdout_summary: str = ""
    stderr_summary: str = ""
    episode_path: str = ""
    error_summary: str = ""

    def line_text(self, *, selected: bool = False) -> str:
        marker = ">" if selected else " "
        impact = f"  {safe_summary(self.impact, 48)}" if self.impact else ""
        display_status = "pending approval" if self.status == "pending" else self.status
        return f"{marker} {self.tool_name} {status_badge(display_status)} ({display_status}){impact}".rstrip()

    def detail_text(self) -> str:
        lines = [
            MODAL_TITLES["tool_details"],
            f"tool name: {safe_summary(self.tool_name, 120)}",
            f"status: {status_badge(self.status)} ({safe_summary(self.status, 80)})",
            f"reason: {safe_summary(self.reason or 'none', 240)}",
            f"args: {redact_mapping_for_display(self.args_summary)}",
            f"impact: {safe_summary(self.impact or '影响范围以工具参数摘要为准', 240)}",
            f"stdout: {safe_summary(self.stdout_summary or 'none', 600)}",
            f"stderr: {safe_summary(self.stderr_summary or 'none', 600)}",
            f"episode: {safe_summary(self.episode_path or 'none', 300)}",
            f"error: {safe_summary(self.error_summary or 'none', 300)}",
        ]
        if self.result_summary:
            lines.append(f"result: {redact_mapping_for_display(self.result_summary)}")
        return "\n".join(lines)


@dataclass
class ToolTimelineState:
    items: list[ToolTimelineItem] = field(default_factory=list)
    selected_index: int = 0

    def apply_event(self, event: ChatEvent) -> None:
        if event.event_type not in TOOL_EVENT_TYPES:
            return
        payload = event.payload
        tool_name = str(payload.get("tool_name", "unknown"))
        if event.event_type in {"tool_started", "approval_requested", "edit_diff_requested"}:
            self.items.append(_item_from_event(event, tool_name))
            if len(self.items) == 1:
                self.selected_index = 0
            return
        item = self._latest_for_tool(tool_name)
        if item is None:
            item = _item_from_event(event, tool_name)
            self.items.append(item)
            if len(self.items) == 1:
                self.selected_index = 0
        _update_item_from_event(item, event)

    def move(self, delta: int) -> None:
        if not self.items:
            return
        self.selected_index = min(max(self.selected_index + delta, 0), len(self.items) - 1)

    def selected_item(self) -> ToolTimelineItem | None:
        if not self.items:
            return None
        self.selected_index = min(max(self.selected_index, 0), len(self.items) - 1)
        return self.items[self.selected_index]

    def render(self, *, limit: int | None = None) -> str:
        if not self.items:
            return "  none"
        items = self.items[-limit:] if limit else self.items
        first_index = len(self.items) - len(items)
        return "\n".join(
            item.line_text(selected=(first_index + index == self.selected_index))
            for index, item in enumerate(items)
        )

    def _latest_for_tool(self, tool_name: str) -> ToolTimelineItem | None:
        for item in reversed(self.items):
            if item.tool_name == tool_name and item.status in {"running", "pending", "approved", "denied"}:
                return item
        return None


def redact_mapping_for_display(mapping: dict[str, object]) -> str:
    if not mapping:
        return "none"
    pieces: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, list):
            value_text = "[" + ", ".join(safe_summary(str(item), 80) for item in value[:4])
            if len(value) > 4:
                value_text += ", ..."
            value_text += "]"
        else:
            value_text = safe_summary(str(value), 180)
        pieces.append(f"{key}={value_text}")
    return "; ".join(pieces)


def _item_from_event(event: ChatEvent, tool_name: str) -> ToolTimelineItem:
    payload = event.payload
    args_summary = _mapping(payload.get("args_summary"))
    reason = str(payload.get("reason") or payload.get("question") or event.message or "")
    status = "pending" if event.event_type in {"approval_requested", "edit_diff_requested"} else "running"
    item = ToolTimelineItem(
        tool_name=tool_name,
        status=status,
        turn_index=event.turn_index,
        reason=reason,
        args_summary=args_summary,
        impact=_impact_summary(tool_name, args_summary),
        episode_path=str(payload.get("episode_path", "")),
    )
    _update_item_from_event(item, event)
    return item


def _update_item_from_event(item: ToolTimelineItem, event: ChatEvent) -> None:
    payload = event.payload
    if event.event_type == "tool_finished":
        item.status = "done" if str(payload.get("status", "success")) != "error" else "failed"
    elif event.event_type == "tool_failed":
        if item.status != "denied":
            item.status = "failed"
    elif event.event_type in {"approval_granted", "edit_diff_granted"}:
        item.status = "approved"
    elif event.event_type in {"approval_denied", "edit_diff_denied"}:
        item.status = "denied"
    if payload.get("args_summary"):
        item.args_summary = _mapping(payload.get("args_summary"))
        item.impact = _impact_summary(item.tool_name, item.args_summary)
    if payload.get("result_summary"):
        item.result_summary = _mapping(payload.get("result_summary"))
        item.stdout_summary = str(item.result_summary.get("stdout_excerpt") or item.result_summary.get("stdout") or "")
        item.stderr_summary = str(item.result_summary.get("stderr_excerpt") or item.result_summary.get("stderr") or "")
    if payload.get("episode_path"):
        item.episode_path = str(payload.get("episode_path"))
    if payload.get("message") or payload.get("error_type") or payload.get("error"):
        item.error_summary = str(payload.get("message") or payload.get("error_type") or payload.get("error") or "")
    if payload.get("reason"):
        item.reason = str(payload.get("reason"))


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _impact_summary(tool_name: str, args_summary: dict[str, object]) -> str:
    if tool_name in {"file_write", "apply_patch"}:
        return f"file={args_summary.get('path', 'unknown')}"
    if tool_name == "apply_patch_set":
        paths = args_summary.get("paths")
        if isinstance(paths, list) and paths:
            return f"files={', '.join(str(path) for path in paths[:3])}"
        return "files=unknown"
    if tool_name == "shell":
        return f"command={args_summary.get('command', 'unknown')}"
    if tool_name == "code_run":
        return f"cwd={args_summary.get('cwd', '.')}"
    if args_summary.get("path"):
        return f"path={args_summary['path']}"
    return ""
