"""
src/haagent/tui/widgets/__init__.py - TUI 基础 widget 包

集中导出 conversation timeline、prompt input 和状态栏等基础 widget。
"""

from haagent.tui.widgets.timeline import (
    ConversationTimeline,
    ConversationView,
    FooterBar,
    PromptInput,
    ResizeMessage,
    StatusBar,
    ToolActivity,
    ToolStatus,
    _end_location,
    merge_tool_activity,
)

__all__ = [
    "ConversationTimeline",
    "ConversationView",
    "FooterBar",
    "PromptInput",
    "ResizeMessage",
    "StatusBar",
    "ToolActivity",
    "ToolStatus",
    "_end_location",
    "merge_tool_activity",
]

