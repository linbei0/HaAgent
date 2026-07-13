"""
haagent/tools/file_tools.py - 文件类本地工具

实现文件发现、读取、写入和补丁类工具，并限制路径在 workspace 内。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from difflib import SequenceMatcher, unified_diff
from pathlib import Path
from typing import Any

from haagent.memory.path_policy import (
    MEMORY_STORE_PATH_ERROR,
    MEMORY_STORE_PATH_MESSAGE,
    is_workspace_memory_store_path,
)
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, resolve_path_for_access
from haagent.runtime.execution.human_interaction import HumanInteractionHandler, HumanInteractionRequest
from haagent.tools.base import tool_error


PATH_GUIDANCE = "path is relative to workspace_root"
ROOT_GUIDANCE = 'root is relative to workspace_root and may be a directory or file; use "." or omit root'
NOISE_DIRECTORIES = {
    ".git",
    ".runs",
    ".smoke-runs",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}
FILE_READ_MODEL_VISIBLE_CHAR_BUDGET = 12000


def file_list(args: dict[str, Any], workspace_root: Path, path_policy: PathPolicy | None = None) -> dict[str, Any]:
    path_arg = args.get("path", ".")
    if not isinstance(path_arg, str):
        return tool_error("tool_argument_invalid", "path must be a string")
    policy = path_policy or default_path_policy(workspace_root)
    root = resolve_path_for_access(path_arg, policy, "read")
    if isinstance(root, str):
        return tool_error("path_policy_denied", root)
    if not root.exists():
        result = tool_error("tool_argument_invalid", f"path does not exist: {path_arg}; {PATH_GUIDANCE}")
        if suggested_tool := _suggest_file_list_parent(root, workspace_root):
            result["suggested_tool"] = suggested_tool
        return result
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


def grep(args: dict[str, Any], workspace_root: Path, path_policy: PathPolicy | None = None) -> dict[str, Any]:
    """优先使用 ripgrep 搜索文本；rg 不可用时退回 Python 遍历。"""
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return tool_error("tool_argument_invalid", "pattern must be a non-empty string")

    root_arg = args.get("root", ".")
    if not isinstance(root_arg, str):
        return tool_error("tool_argument_invalid", "root must be a string")
    file_glob = args.get("file_glob", "**/*")
    if not isinstance(file_glob, str) or not file_glob:
        return tool_error("tool_argument_invalid", "file_glob must be a non-empty string")
    case_sensitive = args.get("case_sensitive", True)
    if not isinstance(case_sensitive, bool):
        return tool_error("tool_argument_invalid", "case_sensitive must be a boolean")
    max_matches = args.get("max_matches", 200)
    if not isinstance(max_matches, int) or max_matches <= 0:
        return tool_error("tool_argument_invalid", "max_matches must be a positive integer")

    policy = path_policy or default_path_policy(workspace_root)
    root = resolve_path_for_access(root_arg, policy, "read")
    if isinstance(root, str):
        return tool_error("path_policy_denied", root)
    if not root.exists():
        return tool_error("tool_argument_invalid", f"root does not exist: {root_arg}; {ROOT_GUIDANCE}")

    rg = shutil.which("rg")
    if rg:
        # 使用 JSON 输出避免 Windows 盘符冒号破坏 path:line:column 解析。
        command = [rg, "--json"]
        if not case_sensitive:
            command.append("-i")
        if not root.is_file():
            command.extend(["--glob", file_glob])
        command.extend(["--", pattern, str(root)])
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode not in (0, 1):
            return tool_error("search_failed", completed.stderr.strip() or "ripgrep failed")
        matches = _parse_rg_json(completed.stdout, workspace_root, max_matches=max_matches)
        return {
            "status": "success",
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
            "truncated": _rg_match_count_exceeds(completed.stdout, max_matches),
        }

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        compiled = re.compile(pattern, flags)
    except re.error as error:
        return tool_error("tool_argument_invalid", f"invalid regex pattern: {error}")
    matches = []
    paths = [root] if root.is_file() else root.rglob("*")
    for path in paths:
        if len(matches) >= max_matches:
            break
        if not path.is_file():
            continue
        if not root.is_file() and not path.relative_to(root).match(file_glob):
            continue
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                match = compiled.search(line)
                if match:
                    matches.append(
                        {
                            "path": _display_path(path, workspace_root),
                            "line": line_number,
                            "column": match.start() + 1,
                            "text": line,
                        },
                    )
                    if len(matches) >= max_matches:
                        break
        except UnicodeDecodeError:
            continue
    return {
        "status": "success",
        "pattern": pattern,
        "matches": matches,
        "match_count": len(matches),
        "truncated": len(matches) >= max_matches,
    }


def file_read(args: dict[str, Any], workspace_root: Path, path_policy: PathPolicy | None = None) -> dict[str, Any]:
    path_arg = args.get("path")
    if not isinstance(path_arg, str):
        return tool_error("tool_argument_invalid", "path must be a string")
    policy = path_policy or default_path_policy(workspace_root)
    path = resolve_path_for_access(path_arg, policy, "read")
    if isinstance(path, str):
        return tool_error("path_policy_denied", path)
    if not path.exists():
        result = tool_error("tool_argument_invalid", f"path does not exist: {path_arg}; {PATH_GUIDANCE}")
        result["suggestions"] = _similar_workspace_paths(path_arg, workspace_root)
        return result
    if not path.is_file():
        result = tool_error("tool_argument_invalid", f"path must be a file: {path_arg}; {PATH_GUIDANCE}")
        if path.is_dir():
            result["suggested_tool"] = {"name": "file_list", "args": {"path": _display_path(path, workspace_root), "max_depth": 1}}
        return result

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
    interaction_handler: HumanInteractionHandler | None = None,
) -> dict[str, Any]:
    path_arg = args.get("path")
    content = args.get("content")
    mode = args.get("mode")
    if not all(isinstance(value, str) for value in (path_arg, content, mode)):
        return tool_error("tool_argument_invalid", "path, content, and mode must be strings")
    if mode not in {"create", "overwrite", "append"}:
        return tool_error("tool_argument_invalid", "mode must be create, overwrite, or append")

    policy = path_policy or default_path_policy(workspace_root)
    path = resolve_path_for_access(path_arg, policy, "full")
    if isinstance(path, str):
        return tool_error("path_policy_denied", path)
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
        interaction_handler,
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
    interaction_handler: HumanInteractionHandler | None = None,
) -> dict[str, Any]:
    """仅允许工作区内文件，并要求 old_text 唯一匹配后再写回。"""
    path_arg = args.get("path")
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    if not all(isinstance(value, str) for value in (path_arg, old_text, new_text)):
        return tool_error("tool_argument_invalid", "path, old_text, and new_text must be strings")

    policy = path_policy or default_path_policy(workspace_root)
    path = resolve_path_for_access(path_arg, policy, "full")
    if isinstance(path, str):
        return tool_error("path_policy_denied", path)
    if is_workspace_memory_store_path(path, workspace_root):
        return tool_error(MEMORY_STORE_PATH_ERROR, MEMORY_STORE_PATH_MESSAGE)
    if not path.exists():
        return tool_error("tool_argument_invalid", f"path does not exist: {path_arg}; {PATH_GUIDANCE}")
    if not path.is_file():
        return tool_error("tool_argument_invalid", f"path must be a file: {path_arg}; {PATH_GUIDANCE}")

    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count == 0:
        return tool_error("patch_text_not_found", "old_text was not found")
    if count > 1:
        return tool_error("patch_text_not_unique", "old_text must match exactly once")

    updated_text = text.replace(old_text, new_text, 1)
    change = _file_change_summary(
        path=path,
        old_text=text,
        new_text=updated_text,
        change_type="modified",
        replacements=1,
    )
    if approval_error := _request_edit_approval(
        interaction_handler,
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
    interaction_handler: HumanInteractionHandler | None = None,
) -> dict[str, Any]:
    """原子校验多个唯一文本替换；全部可应用后才写文件。"""
    replacements = args.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        return tool_error("tool_argument_invalid", "replacements must be a non-empty list")

    staged_texts: dict[Path, str] = {}
    original_texts: dict[Path, str] = {}
    summaries: list[dict[str, object]] = []
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, dict):
            return _patch_set_error(
                "tool_argument_invalid",
                "each replacement must be an object",
                summaries,
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
                summaries,
                replacements,
                index,
                path_arg,
            )

        policy = path_policy or default_path_policy(workspace_root)
        path = resolve_path_for_access(path_arg, policy, "full")
        if isinstance(path, str):
            return _patch_set_error(
                "path_policy_denied",
                path,
                summaries,
                replacements,
                index,
                path_arg,
            )
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
        interaction_handler,
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
    result = tool_error(error_type, message)
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


def _parse_rg_json(output: str, workspace_root: Path, *, max_matches: int | None = None) -> list[dict[str, Any]]:
    """解析 ripgrep JSON 事件流，只保留 match 事件。"""
    root = workspace_root.resolve()
    matches = []
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        if max_matches is not None and len(matches) >= max_matches:
            break
        data = event["data"]
        submatches = data.get("submatches") or [{"start": 0}]
        matches.append(
            {
                "path": Path(data["path"]["text"]).resolve().relative_to(root).as_posix(),
                "line": data["line_number"],
                "column": submatches[0]["start"] + 1,
                "text": data["lines"]["text"].rstrip("\r\n"),
            },
        )
    return matches


def _rg_match_count_exceeds(output: str, max_matches: int) -> bool:
    count = 0
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        count += 1
        if count > max_matches:
            return True
    return False
