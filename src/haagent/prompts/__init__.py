"""
haagent/prompts/__init__.py - 显式提示词包接口

提供内置提示词包和显式命令解析的公共导出。
"""

from __future__ import annotations

from haagent.prompts.commands import PromptCommandResult, parse_prompt_command
from haagent.prompts.packs import PromptPack, get_prompt_pack

__all__ = [
    "PromptCommandResult",
    "PromptPack",
    "get_prompt_pack",
    "parse_prompt_command",
]
