"""
haagent/runtime/loop_guidance.py - Agent loop 推进策略

根据最新工具结果或无工具回复生成短小的下一轮模型提示。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


GUIDANCE_CHAR_LIMIT = 420


@dataclass(frozen=True)
class LoopGuidance:
    status: str
    message: str
    trigger: str
    tool_name: str | None = None


@dataclass
class LoopGuidanceState:
    consecutive_failures: int = 0
    failed_signatures: list[str] = field(default_factory=list)
    successful_tool_count: int = 0
    successful_tool_names: list[str] = field(default_factory=list)
    successful_read_only_signatures: list[str] = field(default_factory=list)
    has_file_change: bool = False
    has_verification_evidence: bool = False


def guidance_for_observation(
    observation: dict[str, object],
    state: LoopGuidanceState,
    goal: str = "",
) -> LoopGuidance | None:
    tool_name = str(observation.get("tool_name", "unknown"))
    args = _dict_or_empty(observation.get("args"))
    result = _dict_or_empty(observation.get("result"))
    status = str(result.get("status", ""))
    if status == "error":
        state.consecutive_failures += 1
        state.failed_signatures.append(_failure_signature(tool_name, args, result))
        guidance = _error_guidance(tool_name, args, result)
        if state.consecutive_failures >= 2:
            guidance = (
                "Do not repeat the same failing tool call. Change strategy, inspect current "
                "state with file_read/file_search, or ask the user with request_user_input."
            )
        return LoopGuidance(
            status="handle_error",
            message=_limit(guidance),
            trigger="tool_error",
            tool_name=tool_name,
        )
    if status == "success":
        had_file_change = state.has_file_change
        state.consecutive_failures = 0
        state.successful_tool_count += 1
        state.successful_tool_names.append(tool_name)
        if tool_name in {"file_write", "apply_patch", "apply_patch_set"}:
            state.has_file_change = True
        if tool_name in {"shell", "code_run"} and result.get("exit_code") == 0:
            state.has_verification_evidence = True
        guidance = _success_guidance(tool_name, args, result)
        if tool_name == "file_read" and had_file_change:
            guidance = (
                "If the read-back content satisfies the request, produce the final answer now; "
                "do not keep editing or repeat file_read. Otherwise make one specific next fix."
            )
        read_only_guidance = _read_only_completion_guidance(tool_name, args, result, state, goal)
        if read_only_guidance is not None:
            guidance = read_only_guidance
        return LoopGuidance(
            status="final_answer_required" if read_only_guidance is not None else "continue",
            message=_limit(guidance),
            trigger="repeated_read_only_exploration" if read_only_guidance is not None else "tool_success",
            tool_name=tool_name,
        )
    return None


def guidance_for_no_tool_response(
    content: str,
    goal: str,
    state: LoopGuidanceState,
) -> LoopGuidance | None:
    normalized_goal = goal.lower()
    normalized_content = content.lower()
    if (
        _goal_needs_edit(normalized_goal)
        and not state.has_file_change
        and _content_mentions_code_or_file(normalized_content)
    ):
        return LoopGuidance(
            status="continue",
            message=(
                "The task appears to require changing files. Do not stop with prose or code only; "
                "choose file_write/apply_patch/apply_patch_set/code_run as appropriate."
            ),
            trigger="no_tool_edit_needed",
        )
    if (
        _goal_needs_verification(normalized_goal)
        and not state.has_verification_evidence
        and _content_claims_done(normalized_content)
    ):
        return LoopGuidance(
            status="continue",
            message=(
                "The response claims completion without fresh verify evidence. Continue by "
                "running verification with shell/code_run, or explicitly explain why verification cannot run."
            ),
            trigger="no_tool_unverified_completion",
        )
    return None


def guidance_observation(guidance: LoopGuidance) -> dict[str, object]:
    return {
        "tool_name": "loop_guidance",
        "args": {"trigger": guidance.trigger, "tool_name": guidance.tool_name},
        "result": {
            "status": guidance.status,
            "message": guidance.message,
            "trigger": guidance.trigger,
            "tool_name": guidance.tool_name,
        },
    }


def _success_guidance(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    if tool_name == "file_read":
        return "Use the file_read content already provided to continue analysis or locate the next specific action."
    if tool_name == "file_search":
        path = _first_match_path(result)
        if path:
            return f"Choose the most relevant search hit and read it next with file_read: {path}."
        return "Review the search result summary and choose the next file_read or refine the search query."
    if tool_name == "context_find":
        read_args = _first_context_read_args(result)
        if read_args:
            return f"context_find found candidates. Choose the most relevant one and read it next with file_read: {read_args}."
        return "context_find found no candidates. change keywords, adjust file_glob, or ask the user with request_user_input."
    if tool_name in {"file_write", "apply_patch", "apply_patch_set"}:
        path = str(result.get("path") or args.get("path") or "")
        if tool_name == "apply_patch_set":
            paths = result.get("paths")
            path = ", ".join(str(item) for item in paths) if isinstance(paths, list) else path
        return f"File change succeeded. Read back {path} or run verification before claiming completion."
    if tool_name == "shell":
        exit_code = result.get("exit_code")
        return f"Shell completed with exit_code={exit_code}. Use stdout/stderr summary to decide if the task is verified."
    if tool_name == "code_run":
        exit_code = result.get("exit_code")
        return f"code_run completed with exit_code={exit_code}. Use the output summary to decide if the task is complete."
    if tool_name == "request_user_input":
        return "Use the user's answer to continue the task with the appropriate tool; do not ask the same question again."
    return "Use the successful tool result to choose the next concrete step or produce a final answer if criteria are satisfied."


def _read_only_completion_guidance(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    state: LoopGuidanceState,
    goal: str,
) -> str | None:
    if not _goal_is_read_only_summary(goal) or state.has_file_change:
        return None
    signature = _read_only_signature(tool_name, args, result)
    if signature is None:
        return None
    repeated = signature in state.successful_read_only_signatures
    state.successful_read_only_signatures.append(signature)
    enough_context = len(state.successful_read_only_signatures) >= 4
    if not repeated and not enough_context:
        return None
    return (
        "You already have enough read-only context for this summary/description task. "
        "Produce the final answer now. Do not call tools again, do not repeat the same "
        "file_read/file_list, and summarize the gathered evidence for the user."
    )


def _read_only_signature(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str | None:
    if tool_name == "file_read":
        path = _first_present_string(args.get("path"), result.get("path"))
        if not path:
            return None
        offset = _first_present(args.get("offset"), result.get("offset"))
        keyword = _first_present(args.get("keyword"), result.get("keyword"))
        return f"file_read:{path}:offset={offset}:keyword={keyword}"
    if tool_name == "file_list":
        path = _first_present_string(args.get("path"), result.get("path"), ".")
        max_depth = _first_present(args.get("max_depth"), result.get("max_depth"))
        return f"file_list:{path}:max_depth={max_depth}"
    return None


def _goal_is_read_only_summary(goal: str) -> bool:
    normalized = goal.lower()
    if _goal_needs_edit(normalized) or _goal_needs_verification(normalized):
        return False
    markers = [
        "介绍",
        "总结",
        "概述",
        "说明",
        "解释",
        "describe",
        "summarize",
        "summary",
        "explain",
        "overview",
    ]
    return any(marker in normalized for marker in markers)


def _error_guidance(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    suggestions = result.get("suggestions")
    if tool_name in {"file_read", "file_list", "file_search"} and isinstance(suggestions, list) and suggestions:
        return f"File path failed; try the suggested path with file_read: {suggestions[0]}."
    error = _dict_or_empty(result.get("error"))
    error_type = str(error.get("type", ""))
    message = str(error.get("message", ""))
    if tool_name == "apply_patch" and _patch_miss(error_type, message):
        path = str(args.get("path", ""))
        return f"Patch did not match. First file_read {path}, then narrow old_text to an exact current snippet."
    if tool_name == "apply_patch_set" and error_type == "patch_text_not_unique":
        path = _first_patch_set_error_path(result)
        return f"Patch text was ambiguous. First file_read {path}, then retry apply_patch_set with expand old_text context."
    if tool_name == "apply_patch_set" and _patch_miss(error_type, message):
        path = _first_patch_set_error_path(result)
        return f"Patch set did not match. First file_read {path}, then retry apply_patch_set with exact current snippets."
    if tool_name in {"shell", "code_run"}:
        return "Use stderr/stdout summary to fix the command or code; do not rerun the same command unchanged."
    return "Use the latest structured error to adjust arguments or ask the user for missing information."


def _first_match_path(result: dict[str, Any]) -> str | None:
    matches = result.get("matches")
    if not isinstance(matches, list):
        return None
    for match in matches:
        if isinstance(match, dict) and isinstance(match.get("path"), str):
            return str(match["path"])
    return None


def _first_context_read_args(result: dict[str, Any]) -> dict[str, Any] | None:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        read_args = candidate.get("recommended_file_read")
        if isinstance(read_args, dict) and isinstance(read_args.get("path"), str):
            return read_args
    return None


def _first_patch_set_error_path(result: dict[str, Any]) -> str:
    replacements = result.get("replacements")
    if isinstance(replacements, list):
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            if replacement.get("status") == "error":
                return str(replacement.get("path") or "")
    return ""


def _patch_miss(error_type: str, message: str) -> bool:
    combined = f"{error_type} {message}".lower()
    return "not found" in combined or "not_applied" in combined or "no match" in combined


def _goal_needs_edit(goal: str) -> bool:
    return any(
        marker in goal
        for marker in [
            "修改",
            "创建",
            "写入",
            "edit",
            "modify",
            "update",
            "create",
            "write",
        ]
    )


def _goal_needs_verification(goal: str) -> bool:
    return any(marker in goal for marker in ["验证", "测试", "test", "verify", "pytest", "脚本"])


def _content_mentions_code_or_file(content: str) -> bool:
    return "```" in content or ".py" in content or ".md" in content or "file" in content or "文件" in content


def _content_claims_done(content: str) -> bool:
    return any(marker in content for marker in ["done", "完成", "pass", "通过", "tests pass"])


def _failure_signature(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    error = _dict_or_empty(result.get("error"))
    return f"{tool_name}:{sorted(args)}:{error.get('type', '')}"


def _dict_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _first_present_string(*values: object) -> str:
    for value in values:
        if value is not None:
            return str(value)
    return ""


def _limit(message: str) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= GUIDANCE_CHAR_LIMIT:
        return normalized
    return normalized[:GUIDANCE_CHAR_LIMIT] + "... [truncated]"
