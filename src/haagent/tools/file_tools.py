"""
haagent/tools/file_tools.py - 文件类本地工具

实现文件发现、读取、写入和补丁类工具；外部路径统一进入目录权限审批。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from difflib import SequenceMatcher, unified_diff
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from haagent.memory.path_policy import (
    MEMORY_STORE_PATH_ERROR,
    MEMORY_STORE_PATH_MESSAGE,
    is_workspace_memory_store_path,
)
from haagent.runtime.execution.path_policy import PathAccess, PathPolicy, default_path_policy
from haagent.runtime.execution.human_interaction import HumanInteractionHandler, HumanInteractionRequest
from haagent.tools.base import RecoveryAction, ToolExecutionContext, tool_error
from haagent.tools.path_access import resolve_tool_paths


PATH_GUIDANCE = "path is relative to workspace_root"
SEARCH_PATH_GUIDANCE = 'path is relative to workspace_root and may be a directory or file; use "." or omit path'
NOISE_DIRECTORIES = {
    ".git",
    ".runs",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}
GREP_NOISE_DIRECTORIES = NOISE_DIRECTORIES | {".tmp", ".haagent-tmp", ".pytest_cache"}
FILE_READ_MODEL_VISIBLE_CHAR_BUDGET = 12000
GREP_DEFAULT_TIMEOUT_SECONDS = 15
GREP_MAX_TIMEOUT_SECONDS = 60
GREP_NARROW_GUIDANCE = "Search was incomplete; narrow path or include and retry."
GREP_TRUNCATED_GUIDANCE = "Search results were truncated; narrow path or include and retry."
PERMISSION_ERROR_MARKERS = (
    "access is denied",
    "access denied",
    "permission denied",
    "operation not permitted",
    "拒绝访问",
    "(os error 5)",
    "(os error 13)",
)


def _use_tool_recovery(tool_name: str, args: dict[str, Any], reason: str) -> RecoveryAction:
    return RecoveryAction("use_tool", reason, tool_name=tool_name, args=args)


def file_list(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    path_arg = args.get("path", ".")
    if not isinstance(path_arg, str):
        return tool_error("tool_argument_invalid", "path must be a string")
    policy = path_policy or default_path_policy(workspace_root)
    root = _resolve_tool_path(path_arg, policy, "read", execution_context)
    if isinstance(root, dict):
        return root
    if not root.exists():
        suggested_tool = _suggest_file_list_parent(root, workspace_root)
        return tool_error(
            "tool_argument_invalid",
            f"path does not exist: {path_arg}; {PATH_GUIDANCE}",
            recovery=(
                _use_tool_recovery(
                    str(suggested_tool["name"]),
                    dict(suggested_tool["args"]),
                    "列出最近存在的父目录，再选择真实子路径。",
                )
                if suggested_tool is not None
                else None
            ),
        )
    if not root.is_dir():
        return tool_error("tool_argument_invalid", f"path must be a directory: {path_arg}; {PATH_GUIDANCE}")

    max_depth = args.get("max_depth", 2)
    max_entries = args.get("max_entries", 100)
    if max_depth < 0:
        return tool_error("tool_argument_invalid", "max_depth must be non-negative")
    if max_entries <= 0:
        return tool_error("tool_argument_invalid", "max_entries must be positive")

    entries: list[str] = []
    skipped_dirs: set[str] = set()
    truncated = _collect_file_tree(
        root=root,
        current=root,
        current_depth=0,
        max_depth=max_depth,
        max_entries=max_entries,
        entries=entries,
        skipped_dirs=skipped_dirs,
    )
    return {
        "status": "success",
        "path": path_arg,
        "max_depth": max_depth,
        "max_entries": max_entries,
        "entry_count": len(entries),
        "truncated": truncated,
        "tree": "\n".join(entries),
        "skipped_dirs": sorted(skipped_dirs),
    }


def grep(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """优先使用 ripgrep 搜索文本；rg 不可用时退回 Python 遍历。"""
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return tool_error("tool_argument_invalid", "pattern must be a non-empty string")

    path_arg = args.get("path", ".")
    if not isinstance(path_arg, str):
        return tool_error("tool_argument_invalid", "path must be a string")
    include = args.get("include")
    if include is not None and (not isinstance(include, str) or not include):
        return tool_error("tool_argument_invalid", "include must be a non-empty string")
    case_sensitive = args.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        return tool_error("tool_argument_invalid", "case_sensitive must be a boolean")
    max_matches = args.get("max_matches", 200)
    if isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches <= 0:
        return tool_error("tool_argument_invalid", "max_matches must be a positive integer")
    timeout_seconds = args.get("timeout_seconds", GREP_DEFAULT_TIMEOUT_SECONDS)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= GREP_MAX_TIMEOUT_SECONDS
    ):
        return tool_error(
            "tool_argument_invalid",
            f"timeout_seconds must be an integer between 1 and {GREP_MAX_TIMEOUT_SECONDS}",
        )

    policy = path_policy or default_path_policy(workspace_root)
    root = _resolve_tool_path(path_arg, policy, "read", execution_context)
    if isinstance(root, dict):
        return root
    if not root.exists():
        return tool_error("tool_argument_invalid", f"path does not exist: {path_arg}; {SEARCH_PATH_GUIDANCE}")
    if access_error := _search_root_access_error(root, path_arg):
        return access_error

    rg = shutil.which("rg")
    if rg:
        return _grep_with_ripgrep(
            rg=rg,
            pattern=pattern,
            root=root,
            workspace_root=workspace_root,
            include=include,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
            timeout_seconds=timeout_seconds,
        )

    return _grep_with_python(
        pattern=pattern,
        root=root,
        workspace_root=workspace_root,
        include=include,
        case_sensitive=case_sensitive,
        max_matches=max_matches,
        timeout_seconds=timeout_seconds,
    )


def file_read(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    path_arg = args.get("path")
    if not isinstance(path_arg, str):
        return tool_error("tool_argument_invalid", "path must be a string")
    policy = path_policy or default_path_policy(workspace_root)
    path = _resolve_tool_path(path_arg, policy, "read", execution_context)
    if isinstance(path, dict):
        return path
    if not path.exists():
        suggestions = _similar_workspace_paths(path_arg, workspace_root)
        return tool_error(
            "tool_argument_invalid",
            f"path does not exist: {path_arg}; {PATH_GUIDANCE}",
            recovery=(
                _use_tool_recovery(
                    "file_read",
                    {"path": suggestions[0]},
                    "使用最接近的现有文件路径重试读取。",
                )
                if suggestions
                else _use_tool_recovery(
                    "file_list",
                    {"path": ".", "max_depth": 2},
                    "先列出目录，再选择真实文件路径。",
                )
            ),
        )
    if not path.is_file():
        if path.is_dir():
            return tool_error(
                "tool_argument_invalid",
                f"path must be a file: {path_arg}; {PATH_GUIDANCE}",
                recovery=_use_tool_recovery(
                    "file_list",
                    {"path": _display_path(path, workspace_root), "max_depth": 1},
                    "目标是目录，改用 file_list 查看内容。",
                ),
            )
        return tool_error("tool_argument_invalid", f"path must be a file: {path_arg}; {PATH_GUIDANCE}")

    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", 200))
    if offset < 0 or limit < 0:
        return tool_error("tool_argument_invalid", "offset and limit must be non-negative")

    keyword = args.get("keyword")
    if keyword is not None and (not isinstance(keyword, str) or not keyword):
        return tool_error("tool_argument_invalid", "keyword must be a non-empty string")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(lines)
    if keyword is None:
        start_index = offset
    else:
        match_index = _first_keyword_line(lines, keyword)
        if match_index is None:
            return tool_error("keyword_not_found", f"keyword not found in {path_arg}: {keyword}")
        start_index = max(0, match_index - (limit // 2))
        if start_index + limit > total_lines:
            start_index = max(0, total_lines - limit)

    end_index = min(start_index + limit, total_lines)
    selected = lines[start_index:end_index]
    content = "".join(selected)
    range_truncated = start_index > 0 or end_index < total_lines
    visible_content, content_collapsed = _bounded_visible_text(content, FILE_READ_MODEL_VISIBLE_CHAR_BUDGET)
    return {
        "status": "success",
        "path": _display_path(path, workspace_root),
        "offset": offset,
        "limit": limit,
        "keyword": keyword,
        "start_line": start_index + 1 if selected else start_index,
        "end_line": end_index,
        "line_count": total_lines,
        "content": content,
        "truncated": range_truncated,
        "model_visible": {
            "path": _display_path(path, workspace_root),
            "offset": offset,
            "limit": limit,
            "keyword": keyword,
            "start_line": start_index + 1 if selected else start_index,
            "end_line": end_index,
            "line_count": total_lines,
            "content": visible_content,
            "truncated": range_truncated or content_collapsed,
            "truncation_reason": _file_read_truncation_reason(range_truncated, content_collapsed),
        },
    }


def file_write(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    path_arg = args.get("path")
    content = args.get("content")
    mode = args.get("mode")
    if not all(isinstance(value, str) for value in (path_arg, content, mode)):
        return tool_error("tool_argument_invalid", "path, content, and mode must be strings")
    if mode not in {"create", "overwrite", "append"}:
        return tool_error("tool_argument_invalid", "mode must be create, overwrite, or append")

    policy = path_policy or default_path_policy(workspace_root)
    path = _resolve_tool_path(path_arg, policy, "full", execution_context)
    if isinstance(path, dict):
        return path
    if is_workspace_memory_store_path(path, workspace_root):
        return tool_error(MEMORY_STORE_PATH_ERROR, MEMORY_STORE_PATH_MESSAGE)
    if not path.parent.exists():
        return tool_error("tool_argument_invalid", f"parent directory does not exist: {path_arg}; {PATH_GUIDANCE}")
    if not path.parent.is_dir():
        return tool_error("tool_argument_invalid", f"parent path must be a directory: {path_arg}; {PATH_GUIDANCE}")
    if path.exists() and not path.is_file():
        return tool_error("tool_argument_invalid", f"path must be a file: {path_arg}; {PATH_GUIDANCE}")

    existed = path.exists()
    if mode == "create" and existed:
        return tool_error("file_exists", f"path already exists: {path_arg}")
    if mode == "append" and not existed:
        return tool_error("file_not_found", f"path does not exist for append: {path_arg}")

    old_text = path.read_text(encoding="utf-8") if existed else ""
    new_text = old_text + content if mode == "append" else content
    change = _file_change_summary(
        path=path,
        old_text=old_text,
        new_text=new_text,
        change_type="added" if not existed else "modified",
        bytes_written=len(content.encode("utf-8")),
        additions=_append_additions(content) if mode == "append" else None,
        deletions=0 if mode == "append" else None,
    )
    if approval_error := _request_edit_approval(
        execution_context.interaction_handler if execution_context is not None else None,
        tool_name="file_write",
        question=f"Approve file edit for {path_arg}?",
        reason=f"file_write will {mode} {path_arg}",
        args_summary={
            "path": path_arg,
            "mode": mode,
            "change_type": change["change_type"],
            "additions": change["additions"],
            "deletions": change["deletions"],
            "bytes_written": change["bytes_written"],
            "diff_preview": _diff_preview(path_arg, old_text, new_text),
        },
    ):
        return approval_error

    path.write_text(new_text, encoding="utf-8")

    return {
        "status": "success",
        "path": str(path),
        "mode": mode,
        "bytes_written": len(content.encode("utf-8")),
        "created": not existed,
        "changed_files": [change],
    }


def apply_patch(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """仅允许工作区内文件，并要求 old_text 唯一匹配后再写回。"""
    path_arg = args.get("path")
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if not all(isinstance(value, str) for value in (path_arg, old_text, new_text)):
        return tool_error("tool_argument_invalid", "path, old_text, and new_text must be strings")

    policy = path_policy or default_path_policy(workspace_root)
    path = _resolve_tool_path(path_arg, policy, "full", execution_context)
    if isinstance(path, dict):
        return path
    if is_workspace_memory_store_path(path, workspace_root):
        return tool_error(MEMORY_STORE_PATH_ERROR, MEMORY_STORE_PATH_MESSAGE)
    if not path.exists():
        return tool_error("tool_argument_invalid", f"path does not exist: {path_arg}; {PATH_GUIDANCE}")
    if not path.is_file():
        return tool_error("tool_argument_invalid", f"path must be a file: {path_arg}; {PATH_GUIDANCE}")

    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        return tool_error(
            "patch_text_not_found",
            "old_text was not found",
            recovery=_use_tool_recovery(
                "file_read",
                {"path": path_arg},
                "先读取文件当前内容，再使用精确片段重试补丁。",
            ),
        )
    if count > 1:
        return tool_error(
            "patch_text_not_unique",
            "old_text must match exactly once",
            recovery=_use_tool_recovery(
                "file_read",
                {"path": path_arg},
                "读取文件并扩大 old_text 上下文，使其唯一匹配。",
            ),
        )

    updated_text = text.replace(old_text, new_text, 1)
    change = _file_change_summary(
        path=path,
        old_text=text,
        new_text=updated_text,
        change_type="modified",
        replacements=1,
    )
    if approval_error := _request_edit_approval(
        execution_context.interaction_handler if execution_context is not None else None,
        tool_name="apply_patch",
        question=f"Approve file edit for {path_arg}?",
        reason=f"apply_patch will modify {path_arg}",
        args_summary={
            "path": path_arg,
            "change_type": change["change_type"],
            "additions": change["additions"],
            "deletions": change["deletions"],
            "replacements": 1,
            "diff_preview": _diff_preview(path_arg, text, updated_text),
        },
    ):
        return approval_error

    path.write_text(updated_text, encoding="utf-8")
    return {"status": "success", "path": str(path), "replacements": 1, "changed_files": [change]}


def apply_patch_set(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """原子校验多个唯一文本替换；全部可应用后才写文件。"""
    replacements = args.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        return tool_error("tool_argument_invalid", "replacements must be a non-empty list")

    validated: list[tuple[str, str, str]] = []
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict):
            return _patch_set_error(
                "tool_argument_invalid",
                "each replacement must be an object",
                [],
                replacements,
                index,
            )
        path_arg = replacement.get("path")
        old_text = replacement.get("old_text")
        new_text = replacement.get("new_text")
        if not all(isinstance(value, str) for value in (path_arg, old_text, new_text)):
            return _patch_set_error(
                "tool_argument_invalid",
                "path, old_text, and new_text must be strings",
                [],
                replacements,
                index,
                path_arg,
            )
        validated.append((path_arg, old_text, new_text))

    policy = path_policy or default_path_policy(workspace_root)
    resolved_paths = resolve_tool_paths(
        [item[0] for item in validated],
        policy,
        "full",
        execution_context,
    )
    if isinstance(resolved_paths, dict):
        return resolved_paths

    staged_texts: dict[Path, str] = {}
    original_texts: dict[Path, str] = {}
    summaries: list[dict[str, object]] = []
    for index, ((path_arg, old_text, new_text), path) in enumerate(zip(validated, resolved_paths)):
        if is_workspace_memory_store_path(path, workspace_root):
            return _patch_set_error(
                MEMORY_STORE_PATH_ERROR,
                MEMORY_STORE_PATH_MESSAGE,
                summaries,
                replacements,
                index,
                path_arg,
            )
        if not path.exists():
            return _patch_set_error(
                "tool_argument_invalid",
                f"path does not exist: {path_arg}; {PATH_GUIDANCE}",
                summaries,
                replacements,
                index,
                path_arg,
            )
        if not path.is_file():
            return _patch_set_error(
                "tool_argument_invalid",
                f"path must be a file: {path_arg}; {PATH_GUIDANCE}",
                summaries,
                replacements,
                index,
                path_arg,
            )

        text = staged_texts.get(path)
        if text is None:
            text = path.read_text(encoding="utf-8")
            original_texts[path] = text
        match_count = text.count(old_text)
        if match_count == 0:
            return _patch_set_error(
                "patch_text_not_found",
                "old_text was not found",
                summaries,
                replacements,
                index,
                path_arg,
                match_count,
            )
        if match_count > 1:
            return _patch_set_error(
                "patch_text_not_unique",
                "old_text must match exactly once",
                summaries,
                replacements,
                index,
                path_arg,
                match_count,
            )

        staged_texts[path] = text.replace(old_text, new_text, 1)
        summaries.append(
            {
                "index": index,
                "path": path_arg,
                "status": "ready",
                "match_count": match_count,
                "old_text_chars": len(old_text),
                "new_text_chars": len(new_text),
            },
        )

    changed_files = [
        _file_change_summary(
            path=path,
            old_text=original_texts[path],
            new_text=text,
            change_type="modified",
            replacements=sum(1 for summary in summaries if summary["path"] == _display_path(path, workspace_root) or summary["path"] == str(path)),
        )
        for path, text in sorted(staged_texts.items(), key=lambda item: str(item[0]))
    ]
    if approval_error := _request_edit_approval(
        execution_context.interaction_handler if execution_context is not None else None,
        tool_name="apply_patch_set",
        question="Approve file edits?",
        reason=f"apply_patch_set will modify {len(changed_files)} file(s)",
        args_summary={
            "replacement_count": len(summaries),
            "paths": [_display_path(path, workspace_root) for path in sorted(staged_texts)],
            "changed_files": changed_files,
            "additions": sum(int(item["additions"]) for item in changed_files),
            "deletions": sum(int(item["deletions"]) for item in changed_files),
            "diff_preview": "\n".join(
                _diff_preview(_display_path(path, workspace_root), original_texts[path], staged_texts[path])
                for path in sorted(staged_texts)
            ),
        },
    ):
        return approval_error

    for path, text in staged_texts.items():
        path.write_text(text, encoding="utf-8")

    success_summaries = [{**summary, "status": "success"} for summary in summaries]
    changed_paths = sorted({str(summary["path"]) for summary in success_summaries})
    return {
        "status": "success",
        "replacement_count": len(success_summaries),
        "paths": changed_paths,
        "replacements": success_summaries,
        "changed_files": changed_files,
    }


def _display_path(path: Path, workspace_root: Path) -> str:
    root = workspace_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _bounded_visible_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = "\n...[model-visible content truncated]...\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[:max_chars], True
    head = keep // 2
    tail = keep - head
    return f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}", True


def _file_read_truncation_reason(range_truncated: bool, content_collapsed: bool) -> str | None:
    if range_truncated and content_collapsed:
        return "requested_range_excludes_file_lines_and_content_over_budget"
    if range_truncated:
        return "requested_range_excludes_file_lines"
    if content_collapsed:
        return "content_over_model_visible_budget"
    return None


def _file_change_summary(
    *,
    path: Path,
    old_text: str,
    new_text: str,
    change_type: str,
    bytes_written: int | None = None,
    replacements: int | None = None,
    additions: int | None = None,
    deletions: int | None = None,
) -> dict[str, object]:
    diff_additions, diff_deletions = _diff_counts(old_text, new_text)
    summary: dict[str, object] = {
        "path": str(path),
        "change_type": change_type,
        "additions": diff_additions if additions is None else additions,
        "deletions": diff_deletions if deletions is None else deletions,
    }
    if bytes_written is not None:
        summary["bytes_written"] = bytes_written
    if replacements is not None:
        summary["replacements"] = replacements
    return summary


def _request_edit_approval(
    interaction_handler: HumanInteractionHandler | None,
    *,
    tool_name: str,
    question: str,
    reason: str,
    args_summary: dict[str, object],
) -> dict[str, Any] | None:
    if interaction_handler is None:
        return None
    response = interaction_handler(
        HumanInteractionRequest(
            interaction_type="edit_diff",
            tool_name=tool_name,
            question=question,
            reason=reason,
            risk_level="high",
            args_summary=args_summary,
        ),
    )
    if response.approved:
        return None
    return tool_error("approval_denied", f"edit approval denied for {tool_name}")


def _resolve_tool_path(
    path: str,
    policy: PathPolicy,
    access: PathAccess,
    execution_context: ToolExecutionContext | None,
) -> Path | dict[str, Any]:
    resolved = resolve_tool_paths([path], policy, access, execution_context)
    if isinstance(resolved, dict):
        return resolved
    return resolved[0]


def _diff_preview(path: str, old_text: str, new_text: str, *, max_lines: int = 80) -> str:
    lines = list(
        unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
            lineterm="",
        ),
    )
    if len(lines) <= max_lines:
        return "".join(lines)
    return "".join(lines[:max_lines]) + f"\n... diff truncated after {max_lines} lines"


def _diff_counts(old_text: str, new_text: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in unified_diff(old_text.splitlines(keepends=True), new_text.splitlines(keepends=True), lineterm=""):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _append_additions(content: str) -> int:
    return max(1, content.count("\n"))


def _patch_set_error(
    error_type: str,
    message: str,
    summaries: list[dict[str, object]],
    replacements: list[object],
    failed_index: int,
    path: object = "",
    match_count: int | None = None,
) -> dict[str, Any]:
    recovery = None
    if error_type in {"patch_text_not_found", "patch_text_not_unique"} and path:
        recovery = _use_tool_recovery(
            "file_read",
            {"path": str(path)},
            "读取失败文件的当前内容，再构造唯一且精确的替换片段。",
        )
    result = tool_error(error_type, message, recovery=recovery)
    failed_summary: dict[str, object] = {
        "index": failed_index,
        "path": str(path or ""),
        "status": "error",
        "reason": message,
    }
    if match_count is not None:
        failed_summary["match_count"] = match_count
    replacement_summaries = [*summaries, failed_summary]
    for skipped_index in range(failed_index + 1, len(replacements)):
        skipped = replacements[skipped_index]
        skipped_path = skipped.get("path", "") if isinstance(skipped, dict) else ""
        replacement_summaries.append(
            {
                "index": skipped_index,
                "path": str(skipped_path or ""),
                "status": "skipped",
                "reason": "previous replacement failed",
            },
        )
    result["replacement_count"] = len(replacements)
    result["replacements"] = replacement_summaries
    return result


def _collect_file_tree(
    *,
    root: Path,
    current: Path,
    current_depth: int,
    max_depth: int,
    max_entries: int,
    entries: list[str],
    skipped_dirs: set[str],
) -> bool:
    try:
        children = sorted(current.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower()))
    except OSError:
        skipped_dirs.add(_relative_tree_path(current, root).rstrip("/"))
        return False
    for child in children:
        if child.is_dir() and child.name in NOISE_DIRECTORIES:
            skipped_dirs.add(_relative_tree_path(child, root).rstrip("/"))
            continue
        if len(entries) >= max_entries:
            return True
        entries.append(_relative_tree_path(child, root))
        if child.is_dir() and current_depth < max_depth - 1:
            if _collect_file_tree(
                root=root,
                current=child,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                max_entries=max_entries,
                entries=entries,
                skipped_dirs=skipped_dirs,
            ):
                return True
    return False


def _relative_tree_path(path: Path, root: Path) -> str:
    suffix = "/" if path.is_dir() else ""
    return path.relative_to(root).as_posix() + suffix


def _first_keyword_line(lines: list[str], keyword: str) -> int | None:
    for index, line in enumerate(lines):
        if keyword in line:
            return index
    return None


def _similar_workspace_paths(path_arg: str, workspace_root: Path) -> list[str]:
    root = workspace_root.resolve()
    candidates: list[tuple[float, str]] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(part in NOISE_DIRECTORIES for part in relative_parts):
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        score = max(
            SequenceMatcher(None, path_arg, relative).ratio(),
            SequenceMatcher(None, Path(path_arg).name, path.name).ratio(),
        )
        if score >= 0.45:
            candidates.append((score, relative))
    return [relative for _, relative in sorted(candidates, key=lambda item: (-item[0], item[1]))[:5]]


def _suggest_file_list_parent(path: Path, workspace_root: Path) -> dict[str, object] | None:
    parent = path.parent
    while parent != parent.parent and not parent.exists():
        parent = parent.parent
    if not parent.exists() or not parent.is_dir():
        return None
    return {
        "name": "file_list",
        "args": {
            "path": _display_path(parent, workspace_root),
            "max_depth": 2,
        },
    }


def _search_root_access_error(root: Path, path_arg: str) -> dict[str, Any] | None:
    """搜索根不可读时立即失败，避免把根级权限错误伪装成“无匹配”。"""
    try:
        if root.is_file():
            with root.open("rb") as stream:
                stream.read(1)
            return None
        if root.is_dir():
            next(root.iterdir(), None)
            return None
    except OSError as error:
        return tool_error("search_failed", f"search path is not readable: {path_arg}: {error}")
    return tool_error("tool_argument_invalid", f"path must be a directory or file: {path_arg}; {SEARCH_PATH_GUIDANCE}")


def _grep_with_ripgrep(
    *,
    rg: str,
    pattern: str,
    root: Path,
    workspace_root: Path,
    include: str | None,
    case_sensitive: bool,
    max_matches: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    # JSON 事件流避免 Windows 盘符冒号破坏 path:line:column 解析。
    command = [rg, "--json", "--no-require-git"]
    if not case_sensitive:
        command.append("-i")
    if not root.is_file() and include is not None:
        command.extend(["--glob", include])
        # 用户显式 include 时追加固定排除；默认不传 glob，完整保留 rg 的 ignore 语义。
        for directory in sorted(GREP_NOISE_DIRECTORIES):
            command.extend(["--glob", f"!**/{directory}/**"])
    command.extend(["--", pattern, str(root)])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        output = _subprocess_text(error.stdout or error.output)
        try:
            matches, truncated = _parse_rg_matches(
                output,
                workspace_root,
                max_matches=max_matches,
                allow_incomplete_line=True,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as parse_error:
            return tool_error("search_failed", f"invalid partial ripgrep output: {parse_error}")
        warning = {
            "type": "timeout",
            "message": f"Search timed out after {timeout_seconds} seconds; returned available matches.",
        }
        return _grep_success(
            pattern=pattern,
            matches=matches,
            truncated=truncated,
            partial=True,
            warnings=[warning],
            skipped_paths=[],
            search_backend="ripgrep",
            guidance=GREP_NARROW_GUIDANCE,
        )

    try:
        matches, truncated = _parse_rg_matches(completed.stdout, workspace_root, max_matches=max_matches)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return tool_error("search_failed", f"invalid ripgrep output: {error}")

    if completed.returncode in (0, 1):
        return _grep_success(
            pattern=pattern,
            matches=matches,
            truncated=truncated,
            partial=False,
            warnings=[],
            skipped_paths=[],
            search_backend="ripgrep",
        )

    skipped_paths = _ripgrep_permission_paths(completed.stderr, workspace_root)
    if skipped_paths is None:
        return tool_error("search_failed", completed.stderr.strip() or "ripgrep failed")
    warning = {
        "type": "permission_denied",
        "message": f"Skipped {len(skipped_paths)} inaccessible path(s) while keeping available matches.",
    }
    return _grep_success(
        pattern=pattern,
        matches=matches,
        truncated=truncated,
        partial=True,
        warnings=[warning],
        skipped_paths=skipped_paths,
        search_backend="ripgrep",
        guidance=GREP_NARROW_GUIDANCE,
    )


def _grep_with_python(
    *,
    pattern: str,
    root: Path,
    workspace_root: Path,
    include: str | None,
    case_sensitive: bool,
    max_matches: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as error:
        return tool_error("tool_argument_invalid", f"invalid regex pattern: {error}")

    deadline = time.monotonic() + timeout_seconds
    skipped_by_type: dict[str, set[str]] = {}
    try:
        paths = _python_search_paths(root, workspace_root, include, deadline, skipped_by_type)
    except subprocess.TimeoutExpired:
        return _grep_success(
            pattern=pattern,
            matches=[],
            truncated=False,
            partial=True,
            warnings=[
                {
                    "type": "timeout",
                    "message": f"Search timed out after {timeout_seconds} seconds before file scanning completed.",
                },
            ],
            skipped_paths=[],
            search_backend="python",
            guidance=GREP_NARROW_GUIDANCE,
        )
    except OSError as error:
        return tool_error("search_failed", f"failed to enumerate search files: {error}")

    matches: list[dict[str, Any]] = []
    truncated = False
    timed_out = False
    for path in paths:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        display_path = _display_path(path, workspace_root)
        try:
            with path.open("rb") as stream:
                if b"\0" in stream.read(8192):
                    continue
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    match = compiled.search(line)
                    if match is None:
                        continue
                    if len(matches) >= max_matches:
                        truncated = True
                        break
                    matches.append(
                        {
                            "path": display_path,
                            "line": line_number,
                            "column": match.start() + 1,
                            "text": line.rstrip("\r\n"),
                        },
                    )
            if truncated:
                break
        except UnicodeDecodeError:
            continue
        except OSError as error:
            # 文件级失败只跳过该文件，并通过 partial/warnings 明确暴露。
            warning_type = "permission_denied" if isinstance(error, PermissionError) else "unreadable_file"
            skipped_by_type.setdefault(warning_type, set()).add(display_path)

    warnings = _python_skip_warnings(skipped_by_type)
    if timed_out:
        warnings.append(
            {
                "type": "timeout",
                "message": f"Search timed out after {timeout_seconds} seconds; returned available matches.",
            },
        )
    partial = timed_out or any(kind in skipped_by_type for kind in ("permission_denied", "unreadable_file"))
    skipped_paths = sorted({path for paths in skipped_by_type.values() for path in paths})
    return _grep_success(
        pattern=pattern,
        matches=matches,
        truncated=truncated,
        partial=partial,
        warnings=warnings,
        skipped_paths=skipped_paths,
        search_backend="python",
        guidance=GREP_NARROW_GUIDANCE if partial else None,
    )


def _python_search_paths(
    root: Path,
    workspace_root: Path,
    include: str | None,
    deadline: float,
    skipped_by_type: dict[str, set[str]],
) -> list[Path]:
    if root.is_file():
        return [root]

    git_paths = _git_search_paths(root, include, deadline)
    if git_paths is not None:
        return git_paths

    paths: list[Path] = []
    gitignore_rules = _gitignore_rules(root)

    def onerror(error: OSError) -> None:
        skipped = Path(error.filename) if error.filename else root
        warning_type = "permission_denied" if isinstance(error, PermissionError) else "unreadable_file"
        skipped_by_type.setdefault(warning_type, set()).add(_display_path(skipped, workspace_root))

    for current, directories, files in os.walk(root, topdown=True, onerror=onerror):
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("python grep traversal", max(0.0, deadline - time.monotonic()))
        current_path = Path(current)
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in GREP_NOISE_DIRECTORIES and (include is not None or not directory.startswith("."))
        )
        for filename in sorted(files):
            if include is None and filename.startswith("."):
                continue
            path = current_path / filename
            relative = path.relative_to(root)
            if _is_gitignored(relative, gitignore_rules):
                continue
            if include is None or relative.match(include):
                paths.append(path)
    return paths


def _gitignore_rules(root: Path) -> list[tuple[Path, str, bool, bool]]:
    """读取 Python 后备搜索所需的最小 .gitignore 规则集。"""
    rules: list[tuple[Path, str, bool, bool]] = []
    for current, directories, files in os.walk(root, topdown=True):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in GREP_NOISE_DIRECTORIES and not directory.startswith(".")
        )
        if ".gitignore" not in files:
            continue
        ignore_file = Path(current) / ".gitignore"
        try:
            lines = ignore_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        base = ignore_file.parent.relative_to(root)
        for raw_pattern in lines:
            pattern = raw_pattern.strip()
            if not pattern or pattern.startswith("#"):
                continue
            negated = pattern.startswith("!")
            if negated:
                pattern = pattern[1:]
            directory_only = pattern.endswith("/")
            pattern = pattern.rstrip("/")
            if pattern:
                rules.append((base, pattern, negated, directory_only))
    return rules


def _is_gitignored(path: Path, rules: list[tuple[Path, str, bool, bool]]) -> bool:
    ignored = False
    for base, pattern, negated, directory_only in rules:
        try:
            relative = path.relative_to(base)
        except ValueError:
            continue
        if _gitignore_rule_matches(relative, pattern, directory_only):
            ignored = not negated
    return ignored


def _gitignore_rule_matches(path: Path, pattern: str, directory_only: bool) -> bool:
    anchored = pattern.startswith("/")
    pattern = pattern.lstrip("/")
    candidates = [path]
    if directory_only:
        candidates = [Path(*path.parts[:index]) for index in range(1, len(path.parts))]
    for candidate in candidates:
        relative = candidate.as_posix()
        if anchored or "/" in pattern:
            if fnmatchcase(relative, pattern):
                return True
        elif any(fnmatchcase(part, pattern) for part in candidate.parts):
            return True
    return False


def _git_search_paths(root: Path, include: str | None, deadline: float) -> list[Path] | None:
    git = shutil.which("git")
    if git is None or root.name in GREP_NOISE_DIRECTORIES:
        return None
    timeout = max(0.001, deadline - time.monotonic())
    discovery = subprocess.run(
        [git, "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if discovery.returncode != 0:
        return None

    repository_root = Path(discovery.stdout.strip()).resolve()
    relative_root = root.resolve().relative_to(repository_root)
    pathspec = relative_root.as_posix() if relative_root.parts else "."
    timeout = max(0.001, deadline - time.monotonic())
    listed = subprocess.run(
        [
            git,
            "-C",
            str(repository_root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            pathspec,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if listed.returncode != 0:
        raise OSError(listed.stderr.strip() or "git ls-files failed")

    paths = []
    for raw_path in listed.stdout.split("\0"):
        if not raw_path:
            continue
        path = repository_root / raw_path
        relative = path.relative_to(root)
        if any(part in GREP_NOISE_DIRECTORIES for part in relative.parts[:-1]):
            continue
        if include is None and any(part.startswith(".") for part in relative.parts):
            continue
        if include is not None and not relative.match(include):
            continue
        if path.is_file():
            paths.append(path)
    return sorted(paths, key=lambda path: path.as_posix().lower())


def _parse_rg_matches(
    output: str,
    workspace_root: Path,
    *,
    max_matches: int,
    allow_incomplete_line: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """解析 ripgrep JSON 事件流；超时可能留下一个不完整的末行。"""
    root = workspace_root.resolve()
    matches: list[dict[str, Any]] = []
    truncated = False
    lines = output.splitlines()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if allow_incomplete_line and index == len(lines) - 1:
                break
            raise
        if event.get("type") != "match":
            continue
        if len(matches) >= max_matches:
            truncated = True
            continue
        data = event["data"]
        submatches = data.get("submatches") or [{"start": 0}]
        matches.append(
            {
                "path": _display_path(Path(data["path"]["text"]), workspace_root),
                "line": data["line_number"],
                "column": submatches[0]["start"] + 1,
                "text": data["lines"]["text"].rstrip("\r\n"),
            },
        )
    return matches, truncated


def _ripgrep_permission_paths(stderr: str, workspace_root: Path) -> list[str] | None:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines or any(not any(marker in line.lower() for marker in PERMISSION_ERROR_MARKERS) for line in lines):
        return None

    paths = []
    for line in lines:
        match = re.match(
            r"^rg:\s+(.*?):\s+.*(?:Access(?: is)? denied|Permission denied|Operation not permitted|拒绝访问|os error (?:5|13))",
            line,
            re.IGNORECASE,
        )
        raw_path = match.group(1) if match else line
        paths.append(_display_path(Path(raw_path), workspace_root))
    return sorted(set(paths))


def _python_skip_warnings(skipped_by_type: dict[str, set[str]]) -> list[dict[str, Any]]:
    labels = {
        "permission_denied": "inaccessible",
        "unreadable_file": "unreadable",
    }
    return [
        {
            "type": warning_type,
            "message": f"Skipped {len(paths)} {labels[warning_type]} path(s).",
        }
        for warning_type, paths in sorted(skipped_by_type.items())
        if paths
    ]


def _grep_success(
    *,
    pattern: str,
    matches: list[dict[str, Any]],
    truncated: bool,
    partial: bool,
    warnings: list[dict[str, Any]],
    skipped_paths: list[str],
    search_backend: str,
    guidance: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success",
        "pattern": pattern,
        "matches": matches,
        "match_count": len(matches),
        "truncated": truncated,
        "partial": partial,
        "warnings": warnings,
        "skipped_paths": skipped_paths,
        "search_backend": search_backend,
    }
    if guidance is not None or truncated:
        result["guidance"] = guidance or GREP_TRUNCATED_GUIDANCE
    return result


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
