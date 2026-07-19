"""
haagent/tools/contributions/files.py - 工作区文件静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.contribution_helpers import (
    changed_files_summary,
    compact_excerpt,
    first_present_string,
    format_search_match,
    interaction_summary_value,
    secret_write_guardrail,
    summary_value,
    patch_set_secret_guardrail,
)
from haagent.tools.file_tools import (
    apply_patch,
    apply_patch_set,
    file_list,
    file_read,
    file_write,
    grep,
)


def _bind_file_list(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return file_list(args, deps.workspace_root, deps.path_policy, context)

    return handler


def _bind_grep(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return grep(args, deps.workspace_root, deps.path_policy, context)

    return handler


def _bind_file_read(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return file_read(args, deps.workspace_root, deps.path_policy, context)

    return handler


def _bind_file_write(deps: ToolRuntimeDeps) -> ToolHandler:
    # 逐次 interaction_handler 经 ToolExecutionContext 注入，Router 不再按名旁路。
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return file_write(
            args,
            deps.workspace_root,
            deps.path_policy,
            context,
        )

    return handler


def _bind_apply_patch(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return apply_patch(
            args,
            deps.workspace_root,
            deps.path_policy,
            context,
        )

    return handler


def _bind_apply_patch_set(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return apply_patch_set(
            args,
            deps.workspace_root,
            deps.path_policy,
            context,
        )

    return handler


def _file_read_args(args: dict[str, Any]) -> dict[str, object]:
    return {
        "path": summary_value(str(args.get("path", "")), 160),
        "offset": args.get("offset"),
        "limit": args.get("limit"),
        "keyword": summary_value(str(args.get("keyword", "")), 80),
    }


def _file_read_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "path": summary_value(str(result.get("path", "")), 160),
        "start_line": result.get("start_line"),
        "end_line": result.get("end_line"),
        "line_count": result.get("line_count"),
        "truncated": bool(result.get("truncated")),
    }


def _file_read_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "path": first_present_string(result.get("path"), args.get("path")),
        "start_line": result.get("start_line"),
        "end_line": result.get("end_line"),
        "line_count": result.get("line_count"),
        "truncated": result.get("truncated", False),
        "content": compact_excerpt(
            first_present_string(result.get("content"), result.get("excerpt")),
        )[0],
    }


def _file_list_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    tree = first_present_string(result.get("tree"), result.get("content"))
    return {
        "status": result.get("status", "unknown"),
        "path": first_present_string(result.get("path"), args.get("path"), "."),
        "entry_count": result.get("entry_count"),
        "truncated": result.get("truncated", False),
        "tree": compact_excerpt(tree)[0],
    }


def _grep_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    matches = result.get("matches")
    formatted_matches = []
    if isinstance(matches, list):
        formatted_matches = [format_search_match(match) for match in matches[:8]]
    warnings = result.get("warnings") if isinstance(result.get("warnings"), list) else []
    skipped_paths = result.get("skipped_paths") if isinstance(result.get("skipped_paths"), list) else []
    return {
        "status": result.get("status", "unknown"),
        "pattern": first_present_string(result.get("pattern"), args.get("pattern")),
        "match_count": result.get("match_count", len(formatted_matches)),
        "truncated": result.get("truncated", False),
        "partial": result.get("partial", False),
        "warnings": warnings[:4],
        "skipped_paths": skipped_paths[:8],
        "skipped_count": len(skipped_paths),
        "guidance": result.get("guidance"),
        "matches": formatted_matches,
    }


def _file_write_interaction(args: dict[str, Any]) -> dict[str, object]:
    content = str(args.get("content", ""))
    return {
        "content_chars": len(content),
        "mode": str(args.get("mode", "")),
        "path": interaction_summary_value(str(args.get("path", "")), 160),
    }


def _file_write_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "path": summary_value(str(result.get("path", "")), 160),
        "mode": result.get("mode"),
        "bytes_written": result.get("bytes_written"),
        "created": result.get("created"),
        "changed_files": changed_files_summary(result),
    }


def _file_write_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "path": first_present_string(result.get("path"), args.get("path")),
        "mode": args.get("mode"),
        "bytes_written": result.get("bytes_written"),
        "created": result.get("created"),
        "truncated": result.get("truncated", False),
    }


def _apply_patch_interaction(args: dict[str, Any]) -> dict[str, object]:
    old_text = str(args.get("old_text", ""))
    new_text = str(args.get("new_text", ""))
    return {
        "new_text_chars": len(new_text),
        "old_text_chars": len(old_text),
        "path": interaction_summary_value(str(args.get("path", "")), 160),
    }


def _apply_patch_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "path": summary_value(str(result.get("path", "")), 160),
        "replacements": result.get("replacements"),
        "changed_files": changed_files_summary(result),
    }


def _apply_patch_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    patch = first_present_string(result.get("patch"), args.get("patch"))
    return {
        "status": result.get("status", "unknown"),
        "path": first_present_string(result.get("path"), args.get("path")),
        "changed": result.get("changed"),
        "patch": compact_excerpt(patch)[0],
    }


def _apply_patch_set_interaction(args: dict[str, Any]) -> dict[str, object]:
    replacements = args.get("replacements")
    if not isinstance(replacements, list):
        return {"replacement_count": 0, "paths": []}
    paths = [
        interaction_summary_value(str(replacement.get("path", "")), 160)
        for replacement in replacements
        if isinstance(replacement, dict)
    ]
    return {"replacement_count": len(replacements), "paths": paths}


def _apply_patch_set_result(result: dict[str, Any]) -> dict[str, object]:
    paths = result.get("paths") if isinstance(result.get("paths"), list) else []
    return {
        "paths": [summary_value(str(path), 160) for path in paths],
        "replacement_count": result.get("replacement_count"),
        "changed_files": changed_files_summary(result),
    }


def _apply_patch_set_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    replacements = args.get("replacements")
    count = len(replacements) if isinstance(replacements, list) else 0
    return {
        "status": result.get("status", "unknown"),
        "replacement_count": result.get("replacement_count", count),
        "changed_paths": result.get("changed_paths", []),
        "summary": compact_excerpt(first_present_string(result.get("summary")))[0],
    }


FILE_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="file_list",
        description=(
            "List a compact directory tree. Use this first when the project structure or exact path is unknown. "
            "Use the narrowest useful path and max_depth. Use grep for file contents and file_read for a known file."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": 'optional absolute or workspace-relative directory; external paths require permission; defaults to "."',
                },
                "max_depth": {
                    "type": "integer",
                    "description": "optional maximum directory depth; defaults to 2",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "optional maximum entries to return; defaults to 100",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_default"}),
        bind_handler=_bind_file_list,
        project_observation=_file_list_observation,
        observation_long_text_keys=("tree",),
    ),
    ToolContribution(
        name="grep",
        description=(
            "Fast regular expression search across file contents. Use it for a known phrase, symbol, error, or "
            "candidate location; narrow with path and include when possible. Returns matching paths and lines. "
            "For open-ended multi-step investigation, delegate with agent instead of repeatedly guessing searches."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "regular expression, for example 'class\\s+Foo' or 'timeout.*error'",
                },
                "path": {
                    "type": "string",
                    "description": "optional absolute or workspace-relative directory or file to search; external paths require permission",
                },
                "include": {
                    "type": "string",
                    "description": "optional file pattern such as '*.py' or '*.{ts,tsx}'; omit to search all visible files",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "optional case sensitivity flag; defaults to true",
                },
                "max_matches": {
                    "type": "integer",
                    "description": "optional total match limit; defaults to 200",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "optional search timeout from 1 to 60 seconds; defaults to 15",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_default"}),
        bind_handler=_bind_grep,
        project_observation=_grep_observation,
    ),
    ToolContribution(
        name="file_read",
        description=(
            "Read a known text file with offset, limit, or keyword context. Use file_list when the path is unknown "
            "and grep when searching for content. Read an existing file before file_write, apply_patch, or "
            "apply_patch_set. Prefer one useful window over many tiny repeated reads; directories require file_list."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "absolute or workspace-relative file path; external paths require permission",
                },
                "offset": {
                    "type": "integer",
                    "description": "optional zero-based line offset",
                },
                "limit": {
                    "type": "integer",
                    "description": "optional maximum number of lines",
                },
                "keyword": {
                    "type": "string",
                    "description": "optional keyword; read lines near the first match",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_default"}),
        bind_handler=_bind_file_read,
        summarize_args=_file_read_args,
        summarize_result=_file_read_result,
        project_observation=_file_read_observation,
        display_name_zh="读取文件",
        observation_long_text_keys=("content",),
    ),
    ToolContribution(
        name="file_write",
        description=(
            "Create, overwrite, or append a text file. Use this for genuinely new files, full replacements, or "
            "appends. Read an existing file before overwriting it. Prefer apply_patch/apply_patch_set for targeted "
            "changes, and do not create extra files or documentation unless the user requested them."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "absolute or workspace-relative file path; external paths require permission",
                },
                "content": {
                    "type": "string",
                    "description": "text content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "write mode: create, overwrite, or append",
                },
            },
            "required": ["path", "content", "mode"],
            "additionalProperties": False,
        },
        execution_effect="workspace_write",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_file_write,
        interaction_args_summary=_file_write_interaction,
        summarize_result=_file_write_result,
        project_observation=_file_write_observation,
        guardrail=lambda args: secret_write_guardrail("file_write", args),
        display_name_zh="写入文件",
    ),
    ToolContribution(
        name="apply_patch",
        description=(
            "Replace one exact, unique text fragment in one existing file. Read the file first and copy exact "
            "indentation and context. If old_text is missing or repeated, read the current file and retry with a "
            "larger unique fragment. Use apply_patch_set for related or multi-file changes."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "absolute or workspace-relative file path; external paths require permission",
                },
                "old_text": {
                    "type": "string",
                    "description": "exact text that must occur once; include surrounding lines when needed for uniqueness",
                },
                "new_text": {
                    "type": "string",
                    "description": "replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        execution_effect="workspace_write",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_apply_patch,
        interaction_args_summary=_apply_patch_interaction,
        summarize_result=_apply_patch_result,
        project_observation=_apply_patch_observation,
        guardrail=lambda args: secret_write_guardrail("apply_patch", args),
        display_name_zh="修改文件",
        observation_long_text_keys=("patch",),
    ),
    ToolContribution(
        name="apply_patch_set",
        description=(
            "Apply one or more exact text replacements atomically across one or multiple files. Read every target "
            "file first. Each old_text must match exactly once; if any replacement is missing or repeated, no file "
            "is written. Prefer this for related multi-file or multi-site edits and verify changed files afterward."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "replacements": {
                    "type": "array",
                    "description": (
                        "non-empty list of replacements; each item has an absolute or workspace-relative path, "
                        "old_text, and new_text"
                    ),
                },
            },
            "required": ["replacements"],
            "additionalProperties": False,
        },
        execution_effect="workspace_write",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_apply_patch_set,
        interaction_args_summary=_apply_patch_set_interaction,
        summarize_result=_apply_patch_set_result,
        project_observation=_apply_patch_set_observation,
        guardrail=patch_set_secret_guardrail,
        display_name_zh="修改文件",
    ),
]
