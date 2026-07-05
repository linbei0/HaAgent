"""
haagent/prompts/packs.py - 内置任务提示词包

定义 HaAgent 显式命令可加载的短提示词包。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPack:
    id: str
    title: str
    content: str
    placement: str = "system"
    hard_required: bool = True
    max_chars: int = 4000


@dataclass(frozen=True)
class PromptMode:
    command: str
    pack: PromptPack
    default_goal: str
    tui_description: str


_PROMPT_MODES: tuple[PromptMode, ...] = (
    PromptMode(
        command="review",
        pack=PromptPack(
            id="code-review",
            title="Code Review",
            content=(
                "Task mode: code review.\n"
                "Goal: identify actionable defects before proposing or making fixes.\n"
                "Workflow: inspect the requested diff or files first; compare behavior "
                "against the user's requirements; check correctness, security, data loss, "
                "regression, edge cases, and missing tests; ignore style-only nits unless "
                "they hide real risk.\n"
                "Evidence: ground every finding in file:line references, observed behavior, "
                "or command output. Do not report guesses as facts.\n"
                "Output: lead with findings ordered by severity. For each finding, state "
                "what is wrong, why it matters, and the smallest practical fix. If no "
                "issues are found, say that clearly and name any remaining test gaps or "
                "residual risk.\n"
                "Boundary: review is read-only unless the user explicitly asks for fixes."
            ),
        ),
        default_goal="Review the current workspace changes.",
        tui_description="使用代码审查提示词包",
    ),
    PromptMode(
        command="debug",
        pack=PromptPack(
            id="debugging",
            title="Debugging",
            content=(
                "Task mode: debugging.\n"
                "Goal: find and fix the root cause, not just the visible symptom.\n"
                "Workflow: read the exact error, failing output, or reproduction steps; "
                "reproduce or inspect the failure before changing code; trace where the "
                "bad state enters the system; compare with nearby working patterns; form "
                "one hypothesis at a time; make the smallest focused change.\n"
                "Evidence: cite the failure signal, relevant files or state transitions, "
                "and the verification command or observation that proves the fix.\n"
                "Output: summarize the root cause, the focused change, and the verification "
                "result. If the failure cannot be reproduced, say what evidence is missing "
                "and what was inspected.\n"
                "Boundary: do not add broad fallbacks, silent degradation, or blind retries "
                "to make the symptom disappear."
            ),
        ),
        default_goal="Debug the described failure.",
        tui_description="使用调试提示词包",
    ),
    PromptMode(
        command="verify",
        pack=PromptPack(
            id="verification",
            title="Verification",
            content=(
                "Task mode: verification.\n"
                "Goal: establish whether the requested result is actually supported by "
                "fresh, concrete evidence.\n"
                "Workflow: identify the claim being verified; choose the smallest relevant "
                "check, such as a focused test, command, file inspection, rendered output, "
                "or manifest record; run or inspect it before making a success claim; read "
                "the full result, including failures and warnings.\n"
                "Evidence: report command names, pass/fail counts, file paths, observed "
                "outputs, or exact manifest facts. Separate verified facts from inference.\n"
                "Output: state Verified, Not verified, or Partially verified; list the "
                "evidence; list any checks not run and the residual risk.\n"
                "Boundary: do not claim success from code reading alone when a relevant "
                "check can be run."
            ),
        ),
        default_goal="Verify the current result with concrete evidence.",
        tui_description="使用验证提示词包",
    ),
)

_PACKS: dict[str, PromptPack] = {mode.pack.id: mode.pack for mode in _PROMPT_MODES}
_MODES_BY_COMMAND: dict[str, PromptMode] = {mode.command: mode for mode in _PROMPT_MODES}


def get_prompt_pack(pack_id: str) -> PromptPack:
    pack = _PACKS[pack_id]
    if len(pack.content) > pack.max_chars:
        raise ValueError(f"prompt pack exceeds max_chars: {pack.id}")
    return pack


def get_prompt_mode(command: str) -> PromptMode | None:
    return _MODES_BY_COMMAND.get(command)


def iter_prompt_modes() -> tuple[PromptMode, ...]:
    return _PROMPT_MODES


def is_prompt_mode_command(text: str) -> bool:
    if not text.startswith("/"):
        return False
    command_text, _, _argument = text[1:].partition(" ")
    return command_text in _MODES_BY_COMMAND
