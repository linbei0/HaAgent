"""
haagent/tools/contributions/core.py - 核心会话静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.memory.prompts import START_MEMORY_UPDATE_TOOL_DESCRIPTION
from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.catalog import ToolContribution
from haagent.tools.contribution_helpers import (
    compact_excerpt,
    first_present_string,
    interaction_summary_value,
)


def _request_user_input_interaction(args: dict[str, Any]) -> dict[str, object]:
    return {
        "question": interaction_summary_value(str(args.get("question", "")), 240),
        "reason": interaction_summary_value(str(args.get("reason", "")), 240),
    }


def _request_user_input_observation(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    answer = first_present_string(result.get("answer"), result.get("answer_excerpt"))
    return {
        "status": result.get("status", "unknown"),
        "question": first_present_string(result.get("question"), args.get("question")),
        "answer_excerpt": compact_excerpt(answer)[0],
        "answer_chars": result.get("answer_chars", len(answer)),
    }


CORE_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="fake_tool",
        description="deterministic test tool",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        router_owned=True,
    ),
    ToolContribution(
        name="load_image_attachment",
        description=(
            "load a previously attached session image by image_id so the next model call "
            "can inspect it as visual input"
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "id from Image Attachment History, for example img-123abc",
                },
            },
            "required": ["image_id"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        router_owned=True,
    ),
    ToolContribution(
        name="request_user_input",
        description=(
            "Ask the user only when execution requires a preference, requirement, choice, or information that "
            "tools cannot discover. Do not ask for file paths, project facts, or runtime state that file_list, "
            "grep, file_read, or other tools can determine. Continue with the returned answer."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "one concrete question whose answer is required to continue",
                },
                "reason": {
                    "type": "string",
                    "description": "briefly explain what decision or missing requirement blocks execution",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        execution_effect="interaction",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default"}),
        router_owned=True,
        interaction_args_summary=_request_user_input_interaction,
        project_observation=_request_user_input_observation,
    ),
    ToolContribution(
        name="start_memory_update",
        description=START_MEMORY_UPDATE_TOOL_DESCRIPTION,
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "short reason describing the durable information that may be worth settlement",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default"}),
        router_owned=True,
    ),
]
