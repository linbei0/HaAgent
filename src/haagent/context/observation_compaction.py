"""
haagent/context/observation_compaction.py - 工具 observation 摘要与压缩

把工具 observation 转成稳定、紧凑的模型输入摘要。
"""

from __future__ import annotations

from typing import Any


OBSERVATION_EXCERPT_CHAR_LIMIT = 240


def observation_tool_name(observation: dict[str, object]) -> str:
    tool_name = observation.get("tool_name", "unknown_tool")
    return str(tool_name)


def observation_summary(observation: dict[str, object]) -> dict[str, object]:
    tool_name = observation_tool_name(observation)
    args = _dict_or_empty(observation.get("args"))
    result = _dict_or_empty(observation.get("result"))
    if tool_name == "file_list":
        return _file_list_observation_summary(args, result)
    if tool_name == "file_read":
        return _file_read_observation_summary(args, result)
    if tool_name == "request_user_input":
        return _request_user_input_observation_summary(args, result)
    if tool_name == "file_write":
        return _file_write_observation_summary(args, result)
    if tool_name == "file_search":
        return _file_search_observation_summary(args, result)
    if tool_name == "context_find":
        return _context_find_observation_summary(args, result)
    if tool_name == "code_run":
        return _code_run_observation_summary(args, result)
    if tool_name == "shell":
        return _shell_observation_summary(args, result)
    if tool_name == "apply_patch":
        return _apply_patch_observation_summary(args, result)
    if tool_name == "apply_patch_set":
        return _apply_patch_set_observation_summary(args, result)
    if tool_name == "verification":
        return _verification_observation_summary(args, result)
    if tool_name == "loop_guidance":
        return _loop_guidance_observation_summary(args, result)
    return _generic_observation_summary(args, result)


def raw_observation_summary(observation: dict[str, object]) -> dict[str, object]:
    return {
        "args": observation.get("args", {}),
        "result": observation.get("result", {}),
    }


def _file_read_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    content = _string_value(result.get("content"))
    excerpt, truncated = _compact_excerpt(content)
    selected_line_count = len(content.splitlines())
    offset = _first_present(args.get("offset"), result.get("offset"))
    start_line = result.get("start_line")
    if start_line is None and isinstance(offset, int) and selected_line_count:
        start_line = offset + 1
    end_line = result.get("end_line")
    if end_line is None and isinstance(start_line, int) and selected_line_count:
        end_line = start_line + selected_line_count - 1
    return {
        "status": _string_value(result.get("status")),
        "path": _first_present_string(args.get("path"), result.get("path")),
        "offset": offset,
        "limit": _first_present(args.get("limit"), result.get("limit")),
        "keyword": _first_present(args.get("keyword"), result.get("keyword")),
        "start_line": start_line,
        "end_line": end_line,
        "line_count": _first_present(result.get("line_count"), selected_line_count),
        "excerpt": excerpt,
        "truncated": bool(result.get("truncated")) or truncated,
    }


def _file_write_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    return {
        "status": _string_value(result.get("status")),
        "path": _first_present_string(args.get("path"), result.get("path")),
        "mode": _first_present_string(args.get("mode"), result.get("mode")),
        "bytes_written": result.get("bytes_written"),
        "created": result.get("created"),
        "truncated": False,
    }


def _request_user_input_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    answer = _string_value(result.get("answer"))
    answer_excerpt, answer_truncated = _compact_excerpt(answer)
    return {
        "status": _string_value(result.get("status")),
        "question": _first_present_string(args.get("question"), result.get("question")),
        "reason": _first_present_string(args.get("reason")),
        "answer_excerpt": answer_excerpt,
        "answer_chars": _first_present(result.get("answer_chars"), len(answer)),
        "truncated": answer_truncated,
    }


def _file_list_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    tree_excerpt, tree_truncated = _compact_excerpt(_string_value(result.get("tree")))
    return {
        "status": _string_value(result.get("status")),
        "path": _first_present_string(args.get("path"), result.get("path"), "."),
        "max_depth": _first_present(args.get("max_depth"), result.get("max_depth")),
        "max_entries": _first_present(args.get("max_entries"), result.get("max_entries")),
        "entry_count": result.get("entry_count"),
        "tree_excerpt": tree_excerpt,
        "skipped_dirs": result.get("skipped_dirs", []),
        "truncated": bool(result.get("truncated")) or tree_truncated,
    }


def _file_search_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    matches = result.get("matches")
    if not isinstance(matches, list):
        matches = []
    excerpt_source = "\n".join(_format_search_match(match) for match in matches)
    excerpt, truncated = _compact_excerpt(excerpt_source)
    return {
        "status": _string_value(result.get("status")),
        "query": _first_present_string(args.get("query"), args.get("pattern")),
        "pattern": _first_present_string(args.get("pattern"), args.get("query")),
        "match_count": len(matches),
        "excerpt": excerpt,
        "truncated": truncated,
    }


def _context_find_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    compact_candidates: list[dict[str, object]] = []
    truncated = bool(result.get("truncated"))
    for candidate in candidates[:5]:
        if not isinstance(candidate, dict):
            continue
        excerpt, excerpt_truncated = _compact_excerpt(_string_value(candidate.get("excerpt")))
        compact_candidates.append(
            {
                "path": _string_value(candidate.get("path")),
                "line": candidate.get("line"),
                "score": candidate.get("score"),
                "reasons": candidate.get("reasons", []),
                "excerpt": excerpt,
                "recommended_file_read": candidate.get("recommended_file_read", {}),
            },
        )
        truncated = truncated or excerpt_truncated
    return {
        "status": _string_value(result.get("status")),
        "query": _first_present_string(args.get("query"), result.get("query")),
        "keywords": result.get("keywords", []),
        "file_glob": _first_present_string(args.get("file_glob"), result.get("file_glob")),
        "candidate_count": result.get("candidate_count", len(candidates)),
        "candidates": compact_candidates,
        "truncated": truncated or len(candidates) > len(compact_candidates),
    }


def _shell_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    stdout_excerpt, stdout_truncated = _compact_excerpt(_string_value(result.get("stdout")))
    stderr_excerpt, stderr_truncated = _compact_excerpt(_string_value(result.get("stderr")))
    return {
        "status": _string_value(result.get("status")),
        "command": _string_value(args.get("command")),
        "cwd": _string_value(args.get("cwd"), default="."),
        "exit_code": result.get("exit_code"),
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "truncated": stdout_truncated or stderr_truncated,
    }


def _code_run_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    stdout_excerpt, stdout_truncated = _compact_excerpt(_string_value(result.get("stdout_excerpt")))
    stderr_excerpt, stderr_truncated = _compact_excerpt(_string_value(result.get("stderr_excerpt")))
    return {
        "status": _string_value(result.get("status")),
        "exit_code": result.get("exit_code"),
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "script_path": _string_value(result.get("script_path")),
        "truncated": bool(result.get("truncated")) or stdout_truncated or stderr_truncated,
    }


def _apply_patch_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    old_text = _string_value(args.get("old_text"))
    new_text = _string_value(args.get("new_text"))
    old_text_excerpt, old_text_truncated = _compact_excerpt(old_text)
    new_text_excerpt, new_text_truncated = _compact_excerpt(new_text)
    return {
        "status": _string_value(result.get("status")),
        "path": _first_present_string(args.get("path"), result.get("path")),
        "old_text_excerpt": old_text_excerpt,
        "new_text_excerpt": new_text_excerpt,
        "old_text_length": len(old_text),
        "new_text_length": len(new_text),
        "replacements": result.get("replacements"),
        "truncated": old_text_truncated or new_text_truncated,
    }


def _apply_patch_set_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    error = _dict_or_empty(result.get("error"))
    replacements = result.get("replacements")
    compact_replacements: list[dict[str, object]] = []
    if isinstance(replacements, list):
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            compact_replacements.append(
                {
                    "index": replacement.get("index"),
                    "path": _string_value(replacement.get("path")),
                    "status": _string_value(replacement.get("status")),
                    "reason": _string_value(replacement.get("reason")),
                    "match_count": replacement.get("match_count"),
                    "old_text_chars": replacement.get("old_text_chars"),
                    "new_text_chars": replacement.get("new_text_chars"),
                },
            )
    paths = result.get("paths")
    if not isinstance(paths, list):
        paths = sorted({item["path"] for item in compact_replacements if item.get("path")})
    failure_reason, reason_truncated = _compact_excerpt(_string_value(error.get("message")))
    return {
        "status": _string_value(result.get("status")),
        "replacement_count": result.get("replacement_count", _patch_set_arg_count(args)),
        "paths": paths,
        "failure_type": _string_value(error.get("type")),
        "failure_reason": failure_reason,
        "replacements": compact_replacements,
        "truncated": reason_truncated,
    }


def _verification_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    stdout_excerpt, stdout_truncated = _compact_excerpt(_string_value(result.get("stdout")))
    stderr_excerpt, stderr_truncated = _compact_excerpt(_string_value(result.get("stderr")))
    return {
        "status": _string_value(result.get("status")),
        "command": _first_present_string(args.get("command"), result.get("command")),
        "exit_code": result.get("exit_code"),
        "failure_reason": _string_value(result.get("failure_reason")),
        "timeout": bool(result.get("timeout")),
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "truncated": stdout_truncated or stderr_truncated,
    }


def _loop_guidance_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    message_excerpt, message_truncated = _compact_excerpt(_string_value(result.get("message")))
    return {
        "status": _string_value(result.get("status")),
        "trigger": _first_present_string(args.get("trigger"), result.get("trigger")),
        "tool_name": _first_present_string(args.get("tool_name"), result.get("tool_name")),
        "message": message_excerpt,
        "truncated": message_truncated,
    }


def _format_search_match(match: object) -> str:
    if not isinstance(match, dict):
        return str(match)
    path = _string_value(match.get("path"))
    line = _string_value(match.get("line"))
    column = _string_value(match.get("column"))
    text = _string_value(match.get("text"))
    location = path
    if line:
        location = f"{location}:{line}"
    if column:
        location = f"{location}:{column}"
    return f"{location}: {text}"


def _generic_observation_summary(
    args: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, object]:
    return {
        "status": _string_value(result.get("status")),
        "args_keys": sorted(str(key) for key in args),
        "result_keys": sorted(str(key) for key in result),
        "truncated": False,
    }


def _compact_excerpt(value: str) -> tuple[str, bool]:
    truncated = len(value) > OBSERVATION_EXCERPT_CHAR_LIMIT
    return value[:OBSERVATION_EXCERPT_CHAR_LIMIT], truncated


def _dict_or_empty(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _string_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


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


def _patch_set_arg_count(args: dict[str, Any]) -> int:
    replacements = args.get("replacements")
    if isinstance(replacements, list):
        return len(replacements)
    return 0
