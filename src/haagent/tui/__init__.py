"""
haagent/tui/__init__.py - HaAgent TUI adapter 包

放置终端界面适配层，所有运行能力通过 AssistantService 获取。
"""

from haagent.tui.application import HaAgentTuiApp, find_untrusted_absolute_paths, is_wide_external_root, run_tui

__all__ = [
    "HaAgentTuiApp",
    "find_untrusted_absolute_paths",
    "is_wide_external_root",
    "run_tui",
]
