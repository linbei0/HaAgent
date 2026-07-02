"""
tests/unit/tui/test_tool_activity.py - TUI 工具活动状态归并测试

验证工具事件摘要按一次工具调用归并，而不是按 started/finished/approval 事件条数重复计数。
"""

from __future__ import annotations

from haagent.tui.widgets import ToolActivity, merge_tool_activity


def test_merge_tool_activity_updates_running_call_to_terminal_status() -> None:
    tools = [ToolActivity("web_search", "running", "starting tool web_search", 1)]

    merge_tool_activity(tools, ToolActivity("web_search", "done", "finished tool web_search", 1))

    assert tools == [ToolActivity("web_search", "done", "finished tool web_search", 1)]


def test_merge_tool_activity_resolves_pending_confirmation_without_duplicate_call() -> None:
    tools = [ToolActivity("shell", "running", "starting tool shell", 1)]

    merge_tool_activity(tools, ToolActivity("shell", "approval", "等待审批", 1))
    merge_tool_activity(tools, ToolActivity("shell", "failed", "审批已拒绝", 1))

    assert tools == [ToolActivity("shell", "failed", "审批已拒绝", 1)]


def test_merge_tool_activity_keeps_repeated_same_tool_calls_distinct() -> None:
    tools = [ToolActivity("web_fetch", "failed", "timed out", 1)]

    merge_tool_activity(tools, ToolActivity("web_fetch", "running", "starting tool web_fetch", 1))

    assert tools == [
        ToolActivity("web_fetch", "failed", "timed out", 1),
        ToolActivity("web_fetch", "running", "starting tool web_fetch", 1),
    ]
