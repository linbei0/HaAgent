"""
src/haagent/tui/application/__init__.py - TUI 应用入口包

集中导出 Textual App、启动函数和 runtime 事件 adapter。
"""

from haagent.tui.application.app import HaAgentTuiApp, run_tui

__all__ = [
    "HaAgentTuiApp",
    "run_tui",
]
