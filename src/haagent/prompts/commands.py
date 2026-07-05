"""
haagent/prompts/commands.py - 显式提示词命令解析

解析 /review、/debug、/verify 等用户显式任务模式命令。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.prompts.packs import get_prompt_mode


@dataclass(frozen=True)
class PromptCommandResult:
    command: str | None
    prompt_pack_ids: list[str]
    normalized_prompt: str


def parse_prompt_command(text: str) -> PromptCommandResult:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)

    command_text = stripped[1:]
    command, separator, body = command_text.partition(" ")
    normalized_command = command.strip().lower()
    mode = get_prompt_mode(normalized_command)
    if separator == "" and mode is None:
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)
    if mode is None:
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)

    normalized_body = body.strip() or mode.default_goal
    return PromptCommandResult(
        command=normalized_command,
        prompt_pack_ids=[mode.pack.id],
        normalized_prompt=normalized_body,
    )
