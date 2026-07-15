"""
src/haagent/tui/presentation/progress.py - TUI 进度事件展示投影

把 runtime task/tool 事件转换为用户可理解的临时状态、时间线摘要和可展开详情。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from haagent.runtime.events.types import ApprovalStateEvent, TaskProgressEvent, ToolActivityEvent, UserInputStateEvent
from haagent.tui.design.copy import tool_display_name

Severity = Literal["info", "warning", "error"]
TimelineKind = Literal["activity", "notice", "effect"]

TEXT_LIMIT = 160
DETAIL_LIMIT = 240

_READ_ONLY_TOOL_STATUS = {
    "file_read": "正在阅读文件...",
    "web_search": "正在搜索资料...",
    "search": "正在搜索资料...",
    "web_fetch": "正在阅读资料...",
    "list_dir": "正在查看目录...",
}
_SIDE_EFFECT_TOOL_TITLES = {
    "file_write": "已写入文件",
    "apply_patch": "已修改文件",
}
_SIDE_EFFECT_STARTED_STATUS = {
    "file_write": "正在修改文件...",
    "apply_patch": "正在修改文件...",
    "shell": "正在运行命令...",
    "code_run": "正在执行代码...",
}
_POTENTIAL_SIDE_EFFECT_TOOLS = {"shell", "code_run"}


@dataclass(frozen=True)
class ProgressStatusState:
    text: str
    severity: Severity
    turn_index: int
    source: Literal["task", "tool", "approval", "input"]


@dataclass(frozen=True)
class TimelinePresentationItem:
    kind: TimelineKind
    title: str
    summary: str
    severity: Severity
    turn_index: int
    detail_id: str | None = None


@dataclass(frozen=True)
class ExpandableDetail:
    detail_id: str
    lines: list[str]


@dataclass(frozen=True)
class ProgressPresentation:
    status_line: ProgressStatusState | None = None
    timeline_item: TimelinePresentationItem | None = None
    details: ExpandableDetail | None = None


def present_task_progress(event: TaskProgressEvent) -> ProgressPresentation:
    # 受阻/恢复建议对普通用户是内部噪音（resume_or_replan 等），不进主对话过程组。
    if event.event_name in {"task_recovery_suggested", "task_step_blocked"}:
        return ProgressPresentation()
    if event.event_name == "task_budget_warning":
        return _task_budget_notice(event)
    return ProgressPresentation()


def present_tool_activity(event: ToolActivityEvent) -> ProgressPresentation:
    if event.status == "failed":
        return _tool_failure_notice(event)
    if event.status == "started" and _has_status_line(event):
        return _tool_status_line(event)
    if _is_side_effect_tool(event):
        return _tool_effect_summary(event)
    return ProgressPresentation()


def present_grouped_tool_failure(event: ToolActivityEvent, *, count: int) -> ProgressPresentation:
    detail = ExpandableDetail(
        detail_id=_tool_detail_id(event),
        lines=[
            f"工具：{_bounded(event.tool_name, 80)}",
            "状态：failed",
            f"失败次数：{count}",
            f"错误类型：{_bounded(_meaningful(event.error_type) or 'unknown', 80)}",
        ],
    )
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="activity",
            title=f"{tool_display_name(event.tool_name)}失败 {count} 次，已使用已有上下文继续",
            summary="",
            severity="warning",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def present_approval_state(event: ApprovalStateEvent) -> ProgressPresentation:
    if event.state == "requested":
        return _approval_requested_notice(event)
    if event.state == "denied":
        return _approval_denied_notice(event)
    return ProgressPresentation()


def present_user_input_state(event: UserInputStateEvent) -> ProgressPresentation:
    if event.state == "requested":
        return _user_input_requested_notice(event)
    if event.approved is False:
        return _user_input_cancelled_notice(event)
    return ProgressPresentation()


def _task_budget_notice(event: TaskProgressEvent) -> ProgressPresentation:
    detail = _task_detail(event)
    action = _meaningful(event.suggested_action) or "继续当前会话，我会从当前步骤接着处理"
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="notice",
            title="任务快到本轮上限",
            summary=f"建议：{_bounded(action)}",
            severity="warning",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _tool_status_line(event: ToolActivityEvent) -> ProgressPresentation:
    text = _READ_ONLY_TOOL_STATUS.get(event.tool_name) or _SIDE_EFFECT_STARTED_STATUS.get(event.tool_name)
    if text is None:
        text = "正在处理..."
    return ProgressPresentation(
        status_line=ProgressStatusState(
            text=text,
            severity="info",
            turn_index=event.turn_index,
            source="tool",
        )
    )


def _tool_effect_summary(event: ToolActivityEvent) -> ProgressPresentation:
    detail = _tool_detail(event)
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="effect",
            title=_SIDE_EFFECT_TOOL_TITLES.get(event.tool_name, "已执行操作"),
            summary=_effect_summary_text(event),
            severity="info",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _tool_failure_notice(event: ToolActivityEvent) -> ProgressPresentation:
    detail = _tool_detail(event)
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="activity",
            title=f"{tool_display_name(event.tool_name)}失败",
            summary="建议：查看错误摘要后重试或调整命令",
            severity="error",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _approval_requested_notice(event: ApprovalStateEvent) -> ProgressPresentation:
    detail = _approval_detail(event)
    title = "需要确认文件改动" if event.approval_kind == "edit_diff" else f"需要确认：{tool_display_name(event.tool_name)}"
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="activity",
            title=title,
            summary="建议：在弹窗中确认或拒绝",
            severity="warning",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _approval_denied_notice(event: ApprovalStateEvent) -> ProgressPresentation:
    detail = _approval_detail(event)
    label = "文件改动已拒绝" if event.approval_kind == "edit_diff" else f"已拒绝：{tool_display_name(event.tool_name)}"
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="activity",
            title=label,
            summary="建议：调整请求或选择其他方案",
            severity="warning",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _user_input_requested_notice(event: UserInputStateEvent) -> ProgressPresentation:
    detail = _user_input_detail(event)
    question = _bounded(event.question)
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="notice",
            title="需要补充信息",
            summary=f"{question}\n建议：请在输入框回答",
            severity="info",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _user_input_cancelled_notice(event: UserInputStateEvent) -> ProgressPresentation:
    detail = _user_input_detail(event)
    return ProgressPresentation(
        timeline_item=TimelinePresentationItem(
            kind="notice",
            title=f"回答已取消：{tool_display_name(event.tool_name)}",
            summary="建议：补充信息后重试或调整任务",
            severity="warning",
            turn_index=event.turn_index,
            detail_id=detail.detail_id,
        ),
        details=detail,
    )


def _task_detail(event: TaskProgressEvent) -> ExpandableDetail:
    lines = [
        f"步骤：{_bounded(event.step_id, 80)}",
        f"事件：{_bounded(event.event_name, 80)}",
        f"状态：{_bounded(event.status, 80)}",
    ]
    if _meaningful(event.category):
        lines.append(f"类别：{_bounded(event.category, 80)}")
    if event.evidence_count:
        lines.append(f"证据：{event.evidence_count}")
    if event.checkpoint_count:
        lines.append(f"检查点：{event.checkpoint_count}")
    if event.reason_chars:
        lines.append(f"原因摘要长度：{event.reason_chars} 字符")
    return ExpandableDetail(detail_id=_task_detail_id(event), lines=lines)


def _tool_detail(event: ToolActivityEvent) -> ExpandableDetail:
    lines = [
        f"工具：{_bounded(event.tool_name, 80)}",
        f"状态：{_bounded(event.status, 80)}",
    ]
    if _meaningful(event.result_status):
        lines.append(f"结果：{_bounded(event.result_status, 80)}")
    if _meaningful(event.error_type):
        lines.append(f"错误类型：{_bounded(event.error_type, 80)}")
    if _meaningful(event.error_message):
        lines.append(f"错误摘要：{_bounded(event.error_message, DETAIL_LIMIT)}")
    return ExpandableDetail(detail_id=_tool_detail_id(event), lines=lines)


def _approval_detail(event: ApprovalStateEvent) -> ExpandableDetail:
    return ExpandableDetail(
        detail_id=f"approval:{event.turn_index}:{event.tool_name}:{event.state}",
        lines=[
            f"工具：{_bounded(event.tool_name, 80)}",
            f"状态：{_bounded(event.state, 80)}",
            f"类型：{_bounded(event.approval_kind, 80)}",
        ],
    )


def _user_input_detail(event: UserInputStateEvent) -> ExpandableDetail:
    return ExpandableDetail(
        detail_id=f"input:{event.turn_index}:{event.tool_name}:{event.state}",
        lines=[
            f"工具：{_bounded(event.tool_name, 80)}",
            f"状态：{_bounded(event.state, 80)}",
        ],
    )


def _is_side_effect_tool(event: ToolActivityEvent) -> bool:
    if event.status != "finished":
        return False
    if event.tool_name in _SIDE_EFFECT_TOOL_TITLES:
        return True
    return event.tool_name in _POTENTIAL_SIDE_EFFECT_TOOLS and _has_file_effect_signal(event)


def _has_status_line(event: ToolActivityEvent) -> bool:
    return event.tool_name in _READ_ONLY_TOOL_STATUS or event.tool_name in _SIDE_EFFECT_STARTED_STATUS


def _effect_summary_text(event: ToolActivityEvent) -> str:
    count = _file_effect_count(event)
    if count:
        return f"{count} 个文件有变更"
    if event.tool_name == "file_write":
        return "文件已写入"
    return "操作已完成"


def _has_file_effect_signal(event: ToolActivityEvent) -> bool:
    return _file_effect_count(event) is not None or bool(
        re.search(r"\b(file|files)\s+(modified|written|changed|patched)\b", _effect_source_text(event), flags=re.IGNORECASE)
    )


def _file_effect_count(event: ToolActivityEvent) -> str | None:
    match = re.search(
        r"\b(?:modified|wrote|written|changed|patched)\s+(\d+)\s+files?\b",
        _effect_source_text(event),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return None


def _effect_source_text(event: ToolActivityEvent) -> str:
    return f"{event.summary}\n{event.result_status}"


def _task_detail_id(event: TaskProgressEvent) -> str:
    return f"task:{event.turn_index}:{event.step_id}:{event.event_name}"


def _tool_detail_id(event: ToolActivityEvent) -> str:
    return f"tool:{event.turn_index}:{event.tool_name}:{event.status}"


def _meaningful(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "n/a"}:
        return ""
    return value.strip()


def _bounded(text: str, limit: int = TEXT_LIMIT) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."
