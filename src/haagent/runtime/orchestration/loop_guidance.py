"""
src/haagent/runtime/orchestration/loop_guidance.py - Agent loop 工具建议生成器

根据工具执行结果生成下一步建议。

职责边界：
- 为 Agent 提供"下一步建议"（非强制指令）
- 不判断任务类型，不猜测意图，不强制终止循环
- 死循环和连续失败由 ProgressGuard 负责
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUGGESTION_CHAR_LIMIT = 420


@dataclass(frozen=True)
class ToolSuggestion:
    message: str
    trigger: str
    tool_name: str | None = None


def suggestion_for_observation(
    observation: dict[str, object],
) -> ToolSuggestion | None:
    """根据工具执行结果，为 Agent 生成可选的下一步建议。"""
    tool_name = str(observation.get("tool_name", "unknown"))
    args = _dict_or_empty(observation.get("args"))
    result = _dict_or_empty(observation.get("result"))
    status = str(result.get("status", ""))

    if status == "error":
        message = _error_suggestion(tool_name, args, result)
        if message is None:
            return None
        return ToolSuggestion(
            message=_limit(message),
            trigger="tool_error",
            tool_name=tool_name,
        )

    if status == "success":
        message = _success_suggestion(tool_name, args, result)
        if message is None:
            return None
        return ToolSuggestion(
            message=_limit(message),
            trigger="tool_success",
            tool_name=tool_name,
        )

    return None


def suggestion_observation(suggestion: ToolSuggestion) -> dict[str, object]:
    return {
        "tool_name": "loop_suggestion",
        "args": {"trigger": suggestion.trigger, "tool_name": suggestion.tool_name},
        "result": {
            "status": "suggestion",
            "message": suggestion.message,
            "trigger": suggestion.trigger,
            "tool_name": suggestion.tool_name,
        },
    }


def safety_violation_observation(violation_message: str, recovery_suggestion: str) -> dict[str, object]:
    return {
        "tool_name": "safety_warning",
        "args": {},
        "result": {
            "status": "warning",
            "message": violation_message,
            "recovery_suggestion": recovery_suggestion,
        },
    }


def _success_suggestion(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str | None:
    if tool_name == "grep":
        path = _first_match_path(result)
        if path:
            return f"Choose the most relevant search hit and read it next with file_read: {path}."
        return "No matches found. Refine the grep pattern or use file_list to explore the directory structure."

    if tool_name in {"file_write", "apply_patch", "apply_patch_set"}:
        path = str(result.get("path") or args.get("path") or "")
        if tool_name == "apply_patch_set":
            paths = result.get("paths")
            path = ", ".join(str(p) for p in paths) if isinstance(paths, list) else path
        return f"File change succeeded. Consider reading back {path} or running verification before completing."

    if tool_name == "request_user_input":
        return "Use the user's answer to continue the task; do not ask the same question again."

    return None


def _error_suggestion(tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> str | None:
    if result.get("execution_state") == "unknown":
        return (
            "Inspect the file, process, or workspace state before retrying. "
            "Only then run a narrower idempotent verification or replan the task."
        )
    suggestions = result.get("suggestions")
    if tool_name in {"file_read", "file_list", "grep"} and isinstance(suggestions, list) and suggestions:
        return f"File path failed; try the suggested path with file_read: {suggestions[0]}."

    suggested_tool = result.get("suggested_tool")
    if tool_name in {"file_read", "file_list", "grep"} and isinstance(suggested_tool, dict):
        name = suggested_tool.get("name")
        args = suggested_tool.get("args")
        if name == "file_list" and isinstance(args, dict):
            path = args.get("path", ".")
            max_depth = args.get("max_depth", 1)
            if tool_name == "file_read":
                return f"file_read received a directory. Use file_list with path={path!r} and max_depth={max_depth}."
            return f"file_list path was unavailable. List the nearest existing parent with path={path!r} and max_depth={max_depth}, then choose a real child path."
        if name == "file_read" and tool_name == "grep" and isinstance(args, dict):
            path = args.get("path", "")
            keyword = args.get("keyword")
            if keyword:
                return f"grep root was a file. Use file_read with path={path!r} and keyword={keyword!r}."
            return f"grep root was a file. Use file_read with path={path!r}."

    error = _dict_or_empty(result.get("error"))
    error_type = str(error.get("type", ""))
    message = str(error.get("message", ""))

    if tool_name == "apply_patch" and _patch_miss(error_type, message):
        path = str(args.get("path", ""))
        return f"Patch did not match. First file_read {path!r}, then narrow old_text to an exact current snippet."

    if tool_name == "apply_patch_set" and error_type == "patch_text_not_unique":
        path = _first_patch_set_error_path(result)
        return f"Patch text is not unique in {path}. Use a longer old_text that uniquely identifies the target location."

    if tool_name == "apply_patch_set" and _patch_miss(error_type, message):
        path = _first_patch_set_error_path(result)
        return f"Patch did not match in {path}. Read the file first to get the exact current content."

    if tool_name == "code_run" and error_type == "tool_argument_invalid":
        return "code_run argument invalid. Check that the script path is correct and the file exists."

    return None


# --- helpers ---

def _first_match_path(result: dict[str, Any]) -> str | None:
    matches = result.get("matches")
    if not isinstance(matches, list) or not matches:
        return None
    first = matches[0]
    if isinstance(first, dict):
        return str(first.get("path") or "") or None
    return None


def _first_patch_set_error_path(result: dict[str, Any]) -> str:
    for key in ("patch_results", "replacements"):
        items = result.get(key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("status") == "error":
                    return str(item.get("path") or "")
    return str(result.get("path") or "")


def _patch_miss(error_type: str, message: str) -> bool:
    combined = f"{error_type} {message}".lower()
    return "not found" in combined or "not_applied" in combined or "no match" in combined


def _dict_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _limit(message: str) -> str:
    normalized = " ".join(message.split())
    if len(normalized) <= SUGGESTION_CHAR_LIMIT:
        return normalized
    return normalized[:SUGGESTION_CHAR_LIMIT] + "... [truncated]"
