"""
haagent/prompts/commands.py - 显式提示词命令解析

解析 /review、/debug、/verify 等用户显式任务模式命令。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptCommandResult:
    command: str | None
    prompt_pack_ids: list[str]
    normalized_prompt: str


_COMMAND_PACKS: dict[str, list[str]] = {
    "review": ["code-review"],
    "debug": ["debugging"],
    "verify": ["verification"],
}

_DEFAULT_GOALS: dict[str, str] = {
    "review": "Review the current workspace changes.",
    "debug": "Debug the described failure.",
    "verify": "Verify the current result with concrete evidence.",
}


def parse_prompt_command(text: str) -> PromptCommandResult:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)

    command_text = stripped[1:]
    command, separator, body = command_text.partition(" ")
    normalized_command = command.strip().lower()
    if separator == "" and normalized_command not in _COMMAND_PACKS:
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)
    if normalized_command not in _COMMAND_PACKS:
        return PromptCommandResult(command=None, prompt_pack_ids=[], normalized_prompt=stripped)

    normalized_body = body.strip() or _DEFAULT_GOALS[normalized_command]
    return PromptCommandResult(
        command=normalized_command,
        prompt_pack_ids=list(_COMMAND_PACKS[normalized_command]),
        normalized_prompt=normalized_body,
    )
