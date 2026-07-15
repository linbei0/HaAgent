"""
src/haagent/tui/widgets/timeline_rendering.py - 对话时间线纯渲染函数

把 timeline item 转为文本摘要。
"""

from __future__ import annotations

from haagent.tui.design.copy import tool_display_name
from haagent.tui.widgets.timeline_models import (
    DETAIL_COLLAPSE_HINT,
    DETAIL_EXPAND_HINT,
    PRESENTATION_DETAIL_LINE_LIMIT,
    TOOL_DETAIL_VISIBLE_LIMIT,
    TOOL_DIAGNOSTIC_VISIBLE_LIMIT,
    TimelineItem,
    TimelineRole,
    TimelineStatus,
    ToolActivity,
    ToolStatus,
)


def render_timeline_item(item: TimelineItem, *, show_tool_details: bool) -> str:
    lines: list[str] = []
    label = timeline_item_label(item)
    if label:
        marker = role_marker(item.role, item.status)
        lines.append(f"{marker} {label}")
    if item.tools:
        # 过程子项默认露出中文工具名，避免再套一层「已完成 N 项」折叠。
        lines.extend(
            render_tool_summary(
                item.tools,
                show_details=show_tool_details,
                list_names=item.role == "process",
            )
        )
    body = timeline_item_body(item)
    if body:
        lines.extend(body.splitlines())
    return "\n".join(lines)


def timeline_item_label(item: TimelineItem) -> str:
    """返回时间线条目标题；过程子项无标题时不回退到空泛的「过程」。"""

    if item.role in {"user", "assistant"}:
        return ""
    # 过程分组头用「已完成 N 项」；子项无 chrome 标题，工具名只出现在工具摘要行。
    if item.role == "process":
        return item.title or ""
    return item.title or role_label(item.role)


def process_group_title(
    process_items: list[TimelineItem],
    *,
    expanded: bool,
    elapsed_seconds: float | None = None,
) -> str:
    """折叠头显示步骤数和本轮真实耗时；旧会话没有耗时数据时只显示步骤数。"""

    arrow = "⌄" if expanded else "›"
    title = f"已完成 {len(process_items)} 步"
    if elapsed_seconds is not None:
        title = f"{title} · {_elapsed_time_label(elapsed_seconds)}"
    return f"{title} {arrow}"


def _elapsed_time_label(elapsed_seconds: float) -> str:
    total_seconds = max(0, round(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def unique_tool_display_names(tools: list[ToolActivity], *, limit: int) -> list[str]:
    names: list[str] = []
    for tool in tools:
        name = tool_display_name(tool.tool_name)
        if name not in names:
            names.append(name)
        if len(names) == limit:
            break
    return names


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


def render_tool_summary(
    tools: list[ToolActivity],
    *,
    show_details: bool,
    list_names: bool = False,
) -> list[str]:
    if not tools:
        return []
    active = [item for item in tools if item.status in {"running", "approval"}]
    failed = [item for item in tools if item.status == "failed"]
    completed = [item for item in tools if item.status == "done"]
    lines: list[str] = []
    if active:
        current = active[-1]
        verb = "等待确认" if current.status == "approval" else f"正在{tool_display_name(current.tool_name)}"
        lines.append(f"  {verb} · {len(active)} 项")
    for item in failed:
        lines.append(f"  {tool_display_name(item.tool_name)}失败")
    if completed and not active and not failed:
        if list_names and not show_details:
            # 过程流内默认露出工具中文名；完整参数/诊断仍按详情展开。
            names = unique_tool_display_names(completed, limit=8)
            lines.append(f"  {' · '.join(names)} ›")
        else:
            lines.append(f"  已完成 {len(completed)} 项 {'⌄' if show_details else '›'}")
    elif completed:
        if list_names and not show_details:
            names = unique_tool_display_names(completed, limit=8)
            lines.append(f"  {' · '.join(names)} ›")
        else:
            lines.append(f"  已完成 {len(completed)} 项 ›")
    if show_details:
        collapsed_tools = max(0, len(tools) - TOOL_DETAIL_VISIBLE_LIMIT)
        visible_tools = tools[-TOOL_DETAIL_VISIBLE_LIMIT:]
        if collapsed_tools:
            lines.append(f"    ... 已折叠 {collapsed_tools} 条较早工具详情")
        for item in visible_tools:
            display_name = tool_display_name(item.tool_name)
            lines.append(f"    - {display_name}（{item.tool_name}）{tool_status_label(item.status)} · {item.summary}")
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
        "running": "运行中",
        "approval": "? 待审批",
        "done": "成功",
        "failed": "! 失败",
    }[status]
