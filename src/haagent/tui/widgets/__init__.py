"""
src/haagent/tui/widgets/__init__.py - TUI 基础 widget 包

集中导出 conversation timeline、prompt input、状态栏和输入停靠区等基础 widget。
时间线相关实现已拆分为独立模块（conversation_timeline / timeline_block /
prompt_input / status），此处只做统一导出入口。
"""

from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.input_dock import InputDock
from haagent.tui.widgets.prompt_input import PromptInput, _end_location
from haagent.tui.widgets.question_prompt import QuestionPrompt
from haagent.tui.widgets.plan_confirmation import PlanConfirmationPanel
from haagent.tui.widgets.todo_panel import TodoPanel
from haagent.tui.widgets.request_history_rail import RequestHistoryPreview, RequestHistoryRail
from haagent.tui.widgets.status import ContextUsageLine, FooterBar, ProgressStatusLine, ResizeMessage, StatusBar
from haagent.tui.widgets.timeline_block import AssistantMarkdown, TimelineBlock, ToolActivityLog
from haagent.tui.widgets.timeline_models import ToolActivity, ToolStatus
from haagent.tui.widgets.tool_activity import merge_tool_activity

__all__ = [
    "AssistantMarkdown",
    "ConversationTimeline",
    "ContextUsageLine",
    "FooterBar",
    "InputDock",
    "ProgressStatusLine",
    "PromptInput",
    "QuestionPrompt",
    "PlanConfirmationPanel",
    "RequestHistoryPreview",
    "RequestHistoryRail",
    "ResizeMessage",
    "StatusBar",
    "TimelineBlock",
    "ToolActivity",
    "ToolActivityLog",
    "ToolStatus",
    "TodoPanel",
    "_end_location",
    "merge_tool_activity",
]
