"""
src/haagent/tui/widgets/timeline.py - 对话时间线兼容再导出

时间线已按职责拆分为独立模块：
- timeline_models：数据模型与常量
- timeline_rendering：纯渲染函数
- tool_activity：工具活动合并逻辑
- timeline_block：单条时间线渲染块
- conversation_timeline：ConversationTimeline 主组件
- prompt_input / status：输入区与状态栏组件

本模块只负责向后兼容的再导出，保持既有 import 路径可用。
"""

from __future__ import annotations

from haagent.tui.widgets.conversation_timeline import ConversationTimeline, ConversationView
from haagent.tui.widgets.prompt_input import PromptInput, _end_location
from haagent.tui.widgets.status import FooterBar, ProgressStatusLine, ResizeMessage, StatusBar
from haagent.tui.widgets.timeline_block import AssistantMarkdown, TimelineBlock, ToolActivityLog
from haagent.tui.widgets.timeline_models import (
    MARKDOWN_DELTA_FLUSH_INTERVAL_MS,
    SELECTION_RESUME_DELAY_MS,
    TOOL_ACTIVITY_FLUSH_INTERVAL_MS,
    TimelineItem,
    TimelineRenderMetrics,
    ToolActivity,
    ToolStatus,
)
from haagent.tui.widgets.timeline_rendering import (
    render_timeline_item as _render_timeline_item,
    render_tool_summary as _render_tool_summary,
    timeline_render_metrics,
)
from haagent.tui.widgets.tool_activity import merge_tool_activity

__all__ = [
    "MARKDOWN_DELTA_FLUSH_INTERVAL_MS",
    "SELECTION_RESUME_DELAY_MS",
    "TOOL_ACTIVITY_FLUSH_INTERVAL_MS",
    "AssistantMarkdown",
    "ConversationTimeline",
    "ConversationView",
    "FooterBar",
    "ProgressStatusLine",
    "PromptInput",
    "ResizeMessage",
    "StatusBar",
    "TimelineBlock",
    "TimelineItem",
    "TimelineRenderMetrics",
    "ToolActivity",
    "ToolActivityLog",
    "ToolStatus",
    "_end_location",
    "_render_timeline_item",
    "_render_tool_summary",
    "merge_tool_activity",
    "timeline_render_metrics",
]
