"""
haagent/tui/typography/__init__.py - TUI 排版适配入口

集中导出终端文本换行相关能力，避免散落 patch Rich/Textual 内部实现。
"""

from __future__ import annotations

from haagent.tui.typography.wrap import (
    compute_uax14_wrap_offsets,
    divide_uax14_line,
    install_textual_line_breaking,
    is_textual_line_breaking_installed,
)

__all__ = [
    "compute_uax14_wrap_offsets",
    "divide_uax14_line",
    "install_textual_line_breaking",
    "is_textual_line_breaking_installed",
]
