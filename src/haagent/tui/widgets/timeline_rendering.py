"""
src/haagent/tui/widgets/timeline_rendering.py - 对话时间线纯渲染函数

把 timeline item 转为文本摘要，并计算渲染体量指标。
"""

from __future__ import annotations

from haagent.tui.widgets.timeline_models import (
    DETAIL_COLLAPSE_HINT,
    DETAIL_EXPAND_HINT,
    PRESENTATION_DETAIL_LINE_LIMIT,
    TOOL_DETAIL_VISIBLE_LIMIT,
    TOOL_DIAGNOSTIC_VISIBLE_LIMIT,
    TimelineItem,
    TimelineRenderMetrics,
    TimelineRole,
    TimelineStatus,
    ToolActivity,
    ToolStatus,
)


def render_timeline_item(item: TimelineItem, *, show_tool_details: bool) -> str:
    label = item.title or role_label(item.role)
    marker = role_marker(item.role, item.status)
    lines = [f"{marker} [{label}]"]
    if item.tools:
        lines.extend(render_tool_summary(item.tools, show_details=show_tool_details))
    body = timeline_item_body(item)
    if body:
        lines.extend(body.splitlines())
    if item.role == "assistant" and item.status == "streaming":
        lines.append("  生成中 · HaAgent")
    return "\n".join(lines)


def timeline_item_body(item: TimelineItem) -> str:
    content = item.content or ""
    if not item.detail_lines:
        return content
    lines = content.splitlines() if content else []
    lines.append(DETAIL_COLLAPSE_HINT if item.expanded else DETAIL_EXPAND_HINT)
    if not item.expanded:
        return "\n".join(lines)
    lines.append("")
    lines.extend(bounded_detail_line(line) for line in item.detail_lines)
    return "\n".join(lines)


def bounded_detail_line(line: str) -> str:
    value = line.strip()
    if len(value) <= PRESENTATION_DETAIL_LINE_LIMIT:
        return value
    return value[: PRESENTATION_DETAIL_LINE_LIMIT - 3].rstrip() + "..."


def timeline_render_metrics(items: list[TimelineItem], *, show_tool_details: bool) -> TimelineRenderMetrics:
    rendered_items = [render_timeline_item(item, show_tool_details=show_tool_details) for item in items]
    tool_count = sum(len(item.tools) for item in items)
    diagnostic_count = sum(len(tool.diagnostics) for item in items for tool in item.tools)
    detail_line_count = 0
    if show_tool_details:
        detail_line_count = sum(len(render_tool_summary(item.tools, show_details=True)) for item in items if item.tools)
    return TimelineRenderMetrics(
        item_count=len(items),
        tool_count=tool_count,
        diagnostic_count=diagnostic_count,
        detail_line_count=detail_line_count,
        rendered_character_count=sum(len(item) for item in rendered_items),
    )


def render_tool_summary(tools: list[ToolActivity], *, show_details: bool) -> list[str]:
    if not tools:
        return []
    counts = {
        "running": sum(1 for item in tools if item.status == "running"),
        "approval": sum(1 for item in tools if item.status == "approval"),
        "done": sum(1 for item in tools if item.status == "done"),
        "failed": sum(1 for item in tools if item.status == "failed"),
    }
    summary_parts = [f"工具 {len(tools)} 项"]
    if counts["done"]:
        summary_parts.append(f"{counts['done']} 成功")
    if counts["running"]:
        summary_parts.append(f"{counts['running']} 运行中")
    if counts["approval"]:
        summary_parts.append(f"{counts['approval']} 待确认")
    if counts["failed"]:
        summary_parts.append(f"{counts['failed']} 失败")
    compact_names = unique_tool_names(tools, limit=3)
    if compact_names:
        summary_parts.append(f"当前：{'、'.join(compact_names)}")
    lines = [f"  [工具] {' · '.join(summary_parts)}"]
    if show_details:
        collapsed_tools = max(0, len(tools) - TOOL_DETAIL_VISIBLE_LIMIT)
        visible_tools = tools[-TOOL_DETAIL_VISIBLE_LIMIT:]
        if collapsed_tools:
            lines.append(f"    ... 已折叠 {collapsed_tools} 条较早工具详情")
        for item in visible_tools:
            lines.append(f"    - 工具 {item.tool_name} {tool_status_label(item.status)} · {item.summary}")
            collapsed_diagnostics = max(0, len(item.diagnostics) - TOOL_DIAGNOSTIC_VISIBLE_LIMIT)
            if collapsed_diagnostics:
                lines.append(f"      ... 已折叠 {collapsed_diagnostics} 条较早诊断")
            lines.extend(f"      诊断：{diagnostic}" for diagnostic in item.diagnostics[-TOOL_DIAGNOSTIC_VISIBLE_LIMIT:])
    return lines


def role_label(role: TimelineRole) -> str:
    return {
        "user": "你",
        "assistant": "HaAgent",
        "system": "系统",
        "failure": "失败",
        "activity": "动态",
        "notice": "提示",
        "effect": "操作",
        "process": "过程",
    }[role]


def role_marker(role: TimelineRole, status: TimelineStatus) -> str:
    if role == "user":
        return ">"
    if role == "failure" or status == "failed":
        return "!"
    if role == "notice":
        return "!"
    if role == "activity":
        return "·"
    if role == "effect":
        return "+"
    if role == "process":
        return "·"
    if role == "assistant" and status == "streaming":
        return ">>"
    return "|"


def tool_status_label(status: ToolStatus) -> str:
    return {
        "running": "... 运行中 (running)",
        "approval": "? 待审批",
        "done": "ok 成功",
        "failed": "! 失败 (failed)",
    }[status]


def unique_tool_names(items: list[ToolActivity], *, limit: int) -> list[str]:
    names: list[str] = []
    for item in reversed(items):
        if item.tool_name not in names:
            names.append(item.tool_name)
        if len(names) == limit:
            break
    return list(reversed(names))
