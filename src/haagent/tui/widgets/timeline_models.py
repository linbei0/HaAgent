"""
src/haagent/tui/widgets/timeline_models.py - 对话时间线数据模型

集中定义 timeline 使用的状态、常量和 dataclass，供 widget、渲染与测试复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TimelineRole = Literal["user", "assistant", "system", "failure", "activity", "notice", "effect", "process"]
TimelineStatus = Literal["streaming", "done", "failed"]
ToolStatus = Literal["running", "approval", "done", "failed"]
TOOL_DETAIL_VISIBLE_LIMIT = 8
TOOL_DIAGNOSTIC_VISIBLE_LIMIT = 2
TOOL_ACTIVITY_FLUSH_INTERVAL_MS = 50
MARKDOWN_DELTA_FLUSH_INTERVAL_MS = 33
SELECTION_RESUME_DELAY_MS = 120
DETAILS_REFRESH_RECENT_TURNS = 3
PRESENTATION_DETAIL_LINE_LIMIT = 240
PROCESS_GROUP_ID_BASE = -1_000_000
DETAIL_EXPAND_HINT = "详情：点击展开"
DETAIL_COLLAPSE_HINT = "详情：点击收起"


@dataclass
class ToolActivity:
    tool_name: str
    status: ToolStatus
    summary: str
    turn_index: int
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class TimelineItem:
    item_id: int
    role: TimelineRole
    turn_index: int
    content: str
    status: TimelineStatus = "done"
    title: str | None = None
    tools: list[ToolActivity] = field(default_factory=list)
    detail_id: str | None = None
    detail_lines: list[str] = field(default_factory=list)
    expanded: bool = False
    pinned: bool = False
    elapsed_seconds: float | None = None
