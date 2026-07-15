"""
haagent/tui/design/renderers.py - TUI 文本渲染逻辑

集中生成状态栏、审批、失败和记忆候选文本，保持组件层轻量。
"""

from __future__ import annotations

from pathlib import Path

from rich.cells import cell_len
from rich.text import Text

from haagent.app.assistant_types import AssistantWorkspaceStatus
from haagent.memory import MemoryCandidate
from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.tui.design.copy import EMPTY_LABELS, PANEL_TITLES
from haagent.tui.design.utils import safe_summary


def status_line(
    status: AssistantWorkspaceStatus,
    *,
    ui_state: str,
    width: int,
) -> Text:
    """只投影普通用户需要的四类状态，并按终端 cell 宽度安全压缩。"""

    width = max(1, width or 120)
    state_label = _WORK_STATE_LABELS.get(ui_state, "需留意")
    web_label = "联网已开" if status.web_enabled else "联网已关"
    workspace_value = _workspace_status_value(status.workspace_root, width=width)

    fixed_right = f"模型  · {web_label} · {state_label}"
    remaining_for_values = max(2, width - cell_len(f"工作区 {workspace_value}  {fixed_right}"))
    model_limit = max(2, min(cell_len(status.model or "未配置"), remaining_for_values))
    model_value = _truncate_cells_end(status.model or "未配置", model_limit)
    right = f"模型 {model_value} · {web_label} · {state_label}"

    workspace_limit = max(2, width - cell_len(right) - cell_len("工作区 ") - 2)
    workspace_value = _truncate_cells_start(workspace_value, workspace_limit)
    left = f"工作区 {workspace_value}"
    gap = max(1, width - cell_len(left) - cell_len(right))

    rendered = Text()
    rendered.append(left, style="dim")
    rendered.append(" " * gap)
    rendered.append("模型 ", style="dim")
    rendered.append(model_value, style="bold")
    rendered.append(" · ", style="dim")
    rendered.append(web_label, style="bold" if status.web_enabled else "dim")
    rendered.append(" · ", style="dim")
    rendered.append(state_label, style="bold")
    return rendered


_WORK_STATE_LABELS = {
    "idle": "空闲",
    "running": "正在工作",
    "waiting approval": "待确认",
    "waiting input": "待补充",
    "done": "已完成",
    "completed": "已完成",
    "failed": "失败",
    "cancelling": "正在取消",
    "cancelled": "已取消",
}


def _workspace_status_value(path: Path, *, width: int) -> str:
    if width < 120:
        return path.name or str(path)
    return _truncate_cells_start(str(path), min(40, max(12, width // 3)))


def _truncate_cells_end(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width == 1:
        return "…"
    kept = ""
    for char in value:
        if cell_len(kept + char) > width - 1:
            break
        kept += char
    return kept + "…"


def _truncate_cells_start(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    if width == 1:
        return "…"
    kept = ""
    for char in reversed(value):
        if cell_len(char + kept) > width - 1:
            break
        kept = char + kept
    return "…" + kept


def approval_body(request: HumanInteractionRequest) -> str:
    lines = [
        "工具请求需要确认",
        "",
        f"工具      {safe_summary(request.tool_name, 80)}",
        f"请求      {safe_summary(request.question, 160)}",
    ]
    if request.reason:
        lines.append(f"原因      {safe_summary(request.reason, 160)}")
    if request.risk_level:
        lines.append(f"风险      {_value_label(request.risk_level)}")
    lines.extend(
        [
            f"参数      {format_args_summary(request.args_summary)}",
            f"影响      {impact_summary(request.tool_name, request.args_summary)}",
            "",
            "高风险内容只展示摘要，不展示完整差异、标准输出或标准错误。",
        ],
    )
    return "\n".join(lines)


def edit_diff_body(request: HumanInteractionRequest, *, max_lines: int = 40) -> str:
    args = request.args_summary
    diff_preview = str(args.get("diff_preview", ""))
    diff_lines = diff_preview.splitlines()
    if len(diff_lines) > max_lines:
        diff_lines = [*diff_lines[:max_lines], f"… 差异预览已截断至 {max_lines} 行"]
    paths = args.get("paths") if isinstance(args.get("paths"), list) else []
    lines = [
        "文件改动需要确认",
        "",
        f"工具      {safe_summary(request.tool_name, 80)}",
        f"路径      {safe_summary(str(args.get('path') or ', '.join(str(path) for path in paths) or '未知'), 160)}",
        f"变更      {_value_label(str(args.get('change_type', 'modified')))}",
        f"统计      +{args.get('additions', 0)} -{args.get('deletions', 0)}",
    ]
    if request.reason:
        lines.append(f"原因      {safe_summary(request.reason, 160)}")
    lines.extend(["", "差异预览", *diff_lines, "", "按 y 允许本次，a 始终允许当前会话内同类改动，n 拒绝。"])
    return "\n".join(lines)


def memory_panel_text(
    *,
    candidates: list[MemoryCandidate],
    selected_index: int,
    detail_mode: bool,
    notice: str | None,
    error: str | None,
) -> str:
    prefix = [PANEL_TITLES["memory"], f"  {notice}", ""] if notice else []
    if error:
        return "\n".join([*prefix, PANEL_TITLES["memory"], f"  记忆候选不可用：{error}"])
    if not candidates:
        return "\n".join([*prefix, PANEL_TITLES["memory"], f"  {EMPTY_LABELS['no_pending_candidates']}"])
    selected_index = min(max(selected_index, 0), len(candidates) - 1)
    if detail_mode:
        return "\n".join([*prefix, memory_candidate_detail(candidates[selected_index])])
    lines = [*prefix, PANEL_TITLES["memory"]]
    for index, candidate in enumerate(candidates):
        marker = ">" if index == selected_index else " "
        lines.append(f"{marker} {candidate.candidate_id} [{candidate.scope}/{candidate.category}] {candidate.title}")
    lines.extend(["", "↑/↓ 移动  g/G 首尾  Enter 详情  a/y 确认  r 拒绝  Esc 返回"])
    return "\n".join(lines)


def memory_candidate_detail(candidate: MemoryCandidate) -> str:
    evidence = candidate.evidence
    lines = [
        "记忆候选详情",
        f"候选编号：{candidate.candidate_id}",
        f"状态：{_value_label(candidate.status)}",
        f"范围：{_value_label(candidate.scope)}",
        f"类别：{_value_label(candidate.category)}",
        f"标题：{candidate.title}",
        f"内容：{candidate.body}",
        f"来源：{_value_label(candidate.source)}",
        f"创建时间：{candidate.created_at}",
        f"标签：{', '.join(candidate.tags) if candidate.tags else '无'}",
        f"风险标记：{', '.join(candidate.risk_flags) if candidate.risk_flags else '无'}",
        "",
        "证据",
        f"来源类型：{evidence.source_type}",
        f"来源摘要：{evidence.source_summary or '无'}",
        f"依据：{evidence.basis or '无'}",
        f"分类理由：{evidence.category_rationale or '无'}",
        f"Episode 路径：{evidence.episode_path or '无'}",
    ]
    return "\n".join(lines)


def format_args_summary(args_summary: dict[str, object]) -> str:
    if not args_summary:
        return "无"
    pieces = []
    for key, value in args_summary.items():
        if isinstance(value, list):
            safe_items = ", ".join(safe_summary(str(item), 80) for item in value[:3])
            if len(value) > 3:
                safe_items += ", ..."
            pieces.append(f"{_ARG_LABELS.get(key, key)}=[{safe_items}]")
        else:
            pieces.append(f"{_ARG_LABELS.get(key, key)}={safe_summary(str(value), 120)}")
    return "; ".join(pieces)


def impact_summary(tool_name: str, args_summary: dict[str, object]) -> str:
    if tool_name in {"file_write", "apply_patch"}:
        path = safe_summary(str(args_summary.get("path", "unknown")), 120)
        return f"会修改本地文件；path={path}"
    if tool_name == "apply_patch_set":
        paths = args_summary.get("paths")
        if isinstance(paths, list) and paths:
            return f"会修改本地文件；paths={safe_summary(', '.join(str(path) for path in paths[:3]), 160)}"
        return "会修改本地文件；paths=unknown"
    if tool_name == "shell":
        command = safe_summary(str(args_summary.get("command", "unknown")), 160)
        return f"会执行本地命令；是否修改本地文件取决于命令；command={command}"
    if tool_name == "code_run":
        return "会执行本地代码；可能读取或修改 workspace 内文件"
    return "影响范围以工具参数摘要为准"


_ARG_LABELS = {
    "path": "路径",
    "paths": "路径",
    "command": "命令",
    "cwd": "工作目录",
    "timeout_seconds": "超时秒数",
    "change_type": "变更类型",
    "additions": "新增行",
    "deletions": "删除行",
}

_VALUE_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "modified": "修改",
    "created": "新建",
    "deleted": "删除",
    "pending": "待确认",
    "approved": "已确认",
    "rejected": "已拒绝",
    "workspace": "工作区",
    "user": "用户",
    "session": "会话",
}


def _value_label(value: str) -> str:
    return _VALUE_LABELS.get(value.strip().casefold(), value)
