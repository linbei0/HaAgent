"""
src/haagent/tui/widgets/tool_activity.py - timeline 工具活动合并逻辑

维护同一 assistant turn 内工具状态更新与诊断附加的纯逻辑。
"""

from __future__ import annotations

from haagent.tui.widgets.timeline_models import ToolActivity, ToolStatus


def merge_tool_activity(tools: list[ToolActivity], activity: ToolActivity) -> None:
    existing_activity = matching_open_tool_activity(tools, activity)
    if existing_activity is None:
        tools.append(activity)
        return
    existing_activity.status = activity.status
    existing_activity.summary = activity.summary
    for diagnostic in activity.diagnostics:
        if diagnostic not in existing_activity.diagnostics:
            existing_activity.diagnostics.append(diagnostic)


def matching_open_tool_activity(tools: list[ToolActivity], activity: ToolActivity) -> ToolActivity | None:
    if activity.status == "running":
        candidate_statuses: set[ToolStatus] = {"approval"}
    else:
        candidate_statuses = {"running", "approval"}
        if activity.status == "failed":
            candidate_statuses.add("failed")
    for item in reversed(tools):
        if item.tool_name == activity.tool_name and item.status in candidate_statuses:
            return item
    return None


def matching_latest_tool_activity(tools: list[ToolActivity], tool_name: str) -> ToolActivity | None:
    for item in reversed(tools):
        if item.tool_name == tool_name:
            return item
    return None
