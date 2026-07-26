"""
haagent/tools/contributions/core.py - 核心会话静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.memory.prompts import START_MEMORY_UPDATE_TOOL_DESCRIPTION
from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.catalog import ToolContribution
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.session_history import SESSION_HISTORY_USAGE_GUIDANCE, session_history
from haagent.tools.contribution_helpers import interaction_summary_value


def _request_user_input_interaction(args: dict[str, Any]) -> dict[str, object]:
    questions = args.get("questions") if isinstance(args.get("questions"), list) else []
    return {
        "headers": [
            interaction_summary_value(str(question.get("header", "")), 48)
            for question in questions
            if isinstance(question, dict)
        ],
        "question_count": len(questions),
        "reason": interaction_summary_value(str(args.get("reason", "")), 240),
    }


def _request_user_input_observation(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "outcome": result.get("outcome", "unknown"),
        "question_count": result.get("question_count", 0),
        "answered_count": result.get("answered_count", 0),
        "answer_chars": result.get("answer_chars", 0),
    }


def _bind_session_history(deps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        del context
        return session_history(args, deps.session_path, runs_root=deps.runs_root)

    return handler


def _session_history_args(args: dict[str, Any]) -> dict[str, object]:
    return {"query": str(args.get("query", ""))[:240], "limit": args.get("limit", 3)}


def _session_history_result(result: dict[str, Any]) -> dict[str, object]:
    items = result.get("results") if isinstance(result.get("results"), list) else []
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    return {
        "matched_turn_count": diagnostics.get("matched_turn_count", 0),
        "selected_turns": diagnostics.get("selected_turns", []),
        "result_count": len(items),
    }


CORE_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="todo_update",
        description=(
            "Replace the complete session Todo list. Use it for multiple independent tasks, usually three or "
            "more meaningful steps, or long-running work. Keep at most one item in_progress and update it "
            "immediately after a milestone completes. Do not split every file read or tool call into a Todo."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "explanation": {"type": "string", "description": "optional short reason for this update"},
                "items": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "maxLength": 80},
                            "content": {"type": "string", "maxLength": 240},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "cancelled"],
                            },
                        },
                        "required": ["id", "content", "status"],
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        execution_effect="session_state",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "main_only"}),
        router_owned=True,
        display_name_zh="更新任务清单",
    ),
    ToolContribution(
        name="submit_plan",
        description="Submit the complete structured Plan proposal for user confirmation. Only available in Plan Mode.",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "maxLength": 240},
                "summary": {"type": "string", "maxLength": 1200},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "content": {"type": "string", "maxLength": 240},
                            "completion_condition": {"type": "string", "maxLength": 240},
                        },
                        "required": ["content", "completion_condition"],
                    },
                },
                "verification": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "required": {"type": "boolean"},
                        "description": {"type": "string", "maxLength": 480},
                    },
                    "required": ["required", "description"],
                },
                "assumptions": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string", "maxLength": 240},
                },
            },
            "required": ["goal", "summary", "steps", "verification", "assumptions"],
            "additionalProperties": False,
        },
        execution_effect="interaction",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"plan_mode", "main_only"}),
        router_owned=True,
        display_name_zh="提交实施方案",
    ),
    ToolContribution(
        name="session_history",
        description="Search dialogue evidence already persisted for the current session. "
        + SESSION_HISTORY_USAGE_GUIDANCE,
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "specific words, paths, decisions, or constraints to recall"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "optional result count; defaults to 3"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        bind_handler=_bind_session_history,
        summarize_args=_session_history_args,
        summarize_result=_session_history_result,
        display_name_zh="检索会话历史",
    ),
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
            "Ask one to three short structured questions only when execution requires a preference, requirement, "
            "choice, or information that "
            "tools cannot discover. Do not ask for file paths, project facts, or runtime state that file_list, "
            "grep, file_read, or other tools can determine. Prefer one question. Put a recommended option first "
            "and suffix its label with （推荐）. Do not ask for secrets. Continue with the returned answers."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "description": "one to three questions; prefer one unless independent decisions block progress",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string", "description": "stable unique snake_case question id"},
                            "header": {"type": "string", "description": "very short label shown in the UI"},
                            "question": {"type": "string", "description": "complete question text"},
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                    "required": ["label", "description"],
                                },
                            },
                            "multiple": {"type": "boolean"},
                            "custom": {"type": "boolean"},
                            "placeholder": {"type": "string"},
                        },
                        "required": ["id", "header", "question"],
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "briefly explain what decision or missing requirement blocks execution",
                },
            },
            "required": ["questions"],
            "additionalProperties": False,
        },
        execution_effect="interaction",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default"}),
        router_owned=True,
        interaction_args_summary=_request_user_input_interaction,
        project_observation=_request_user_input_observation,
        display_name_zh="询问用户",
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
