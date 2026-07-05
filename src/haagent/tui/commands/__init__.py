"""
src/haagent/tui/commands/__init__.py - TUI 命令包

集中导出 slash command registry、解析器和命令建议 overlay。
"""

from haagent.tui.commands.registry import (
    CommandRegistry,
    SlashCommand,
    SlashCommandResult,
    command_registry,
    is_prompt_mode_command,
    parse_slash_command,
)
from haagent.tui.commands.suggestions import CommandSuggestionOverlay, CommandSuggestionState

__all__ = [
    "CommandSuggestionOverlay",
    "CommandSuggestionState",
    "SlashCommand",
    "CommandRegistry",
    "SlashCommandResult",
    "command_registry",
    "is_prompt_mode_command",
    "parse_slash_command",
]
