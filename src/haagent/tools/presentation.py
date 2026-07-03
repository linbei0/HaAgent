"""
haagent/tools/presentation.py - 工具展示摘要

集中生成工具参数和工具结果的轻量摘要，供 runtime/TUI 展示使用，不执行工具或做权限判断。
"""

from __future__ import annotations

from haagent.runtime.execution.human_interaction import interaction_args_summary


def summarize_tool_args(tool_name: str, args: dict[str, object]) -> dict[str, object]:
    if tool_name in {"file_write", "code_run", "apply_patch", "apply_patch_set", "shell", "request_user_input"}:
        return interaction_args_summary(tool_name, args)
    if tool_name == "file_read":
        return {
            "path": _summary_value(str(args.get("path", "")), 160),
            "offset": args.get("offset"),
            "limit": args.get("limit"),
            "keyword": _summary_value(str(args.get("keyword", "")), 80),
        }
    return {"args_keys": sorted(str(key) for key in args)}


def summarize_tool_result(tool_name: str, result: dict[str, object]) -> dict[str, object]:
    if tool_name == "file_read":
        return {
            "path": _summary_value(str(result.get("path", "")), 160),
            "start_line": result.get("start_line"),
            "end_line": result.get("end_line"),
            "line_count": result.get("line_count"),
            "truncated": bool(result.get("truncated")),
        }
    if tool_name == "file_write":
        return {
            "path": _summary_value(str(result.get("path", "")), 160),
            "mode": result.get("mode"),
            "bytes_written": result.get("bytes_written"),
            "created": result.get("created"),
            "changed_files": _changed_files_summary(result),
        }
    if tool_name == "apply_patch":
        return {
            "path": _summary_value(str(result.get("path", "")), 160),
            "replacements": result.get("replacements"),
            "changed_files": _changed_files_summary(result),
        }
    if tool_name == "apply_patch_set":
        paths = result.get("paths") if isinstance(result.get("paths"), list) else []
        return {
            "paths": [_summary_value(str(path), 160) for path in paths],
            "replacement_count": result.get("replacement_count"),
            "changed_files": _changed_files_summary(result),
        }
    if tool_name == "code_run":
        return {
            "exit_code": result.get("exit_code"),
            "stdout_excerpt": _summary_value(str(result.get("stdout_excerpt", "")), 300),
            "stderr_excerpt": _summary_value(str(result.get("stderr_excerpt", "")), 300),
            "stdout_chars": len(str(result.get("stdout_excerpt", ""))),
            "stderr_chars": len(str(result.get("stderr_excerpt", ""))),
            "truncated": bool(result.get("truncated")),
        }
    if tool_name == "shell":
        return {
            "exit_code": result.get("exit_code"),
            "stdout_excerpt": _summary_value(str(result.get("stdout_excerpt", "")), 300),
            "stderr_excerpt": _summary_value(str(result.get("stderr_excerpt", "")), 300),
            "stdout_chars": len(str(result.get("stdout_excerpt", ""))),
            "stderr_chars": len(str(result.get("stderr_excerpt", ""))),
            "timeout": bool(result.get("timeout")),
            "truncated": bool(result.get("truncated")),
        }
    return {
        "status": str(result.get("status", "unknown")),
        "result_keys": sorted(str(key) for key in result),
    }


def _summary_value(value: str, limit: int = 300) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        normalized = "none"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _changed_files_summary(result: dict[str, object]) -> list[dict[str, object]]:
    changed_files = result.get("changed_files")
    if not isinstance(changed_files, list):
        return []
    summaries: list[dict[str, object]] = []
    for item in changed_files:
        if not isinstance(item, dict):
            continue
        summary: dict[str, object] = {
            "path": _summary_value(str(item.get("path", "")), 160),
            "change_type": str(item.get("change_type", "modified")),
            "additions": item.get("additions"),
            "deletions": item.get("deletions"),
        }
        if "bytes_written" in item:
            summary["bytes_written"] = item.get("bytes_written")
        if "replacements" in item:
            summary["replacements"] = item.get("replacements")
        summaries.append(summary)
    return summaries
