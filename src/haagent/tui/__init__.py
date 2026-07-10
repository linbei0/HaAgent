"""
haagent/tui/__init__.py - HaAgent TUI adapter 包

放置终端界面适配层，所有运行能力通过 AssistantService 获取。
"""

from haagent.tui.application import HaAgentTuiApp, run_tui

__all__ = [
    "HaAgentTuiApp",
    "run_tui",
]
