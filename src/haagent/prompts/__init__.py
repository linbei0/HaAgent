"""
haagent/prompts/__init__.py - 显式提示词包接口

提供内置提示词包和显式命令解析的公共导出。
"""

from __future__ import annotations

from haagent.prompts.commands import PromptCommandResult, parse_prompt_command
from haagent.prompts.packs import (
    PromptMode,
    PromptPack,
    get_prompt_mode,
    get_prompt_pack,
    is_prompt_mode_command,
    iter_prompt_modes,
)

__all__ = [
    "PromptCommandResult",
    "PromptMode",
    "PromptPack",
    "get_prompt_mode",
    "get_prompt_pack",
    "is_prompt_mode_command",
    "iter_prompt_modes",
    "parse_prompt_command",
]
