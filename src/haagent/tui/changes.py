"""
haagent/tui/changes.py - 文件变更摘要模型

从结构化工具事件提取本轮 changed files，不依赖 Git 状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from haagent.tui.copy import EMPTY_LABELS
from haagent.tui.theme import status_semantic
from haagent.tui.utils import safe_summary


@dataclass(frozen=True)
class ChangedFileSummary:
    path: str
    change_type: str
    summary: str

    def line_text(self) -> str:
        semantic = status_semantic("success")
        label = {"added": "新增", "modified": "修改"}.get(self.change_type, self.change_type)
        return f"  {semantic.symbol} {label} {self.path}  {self.summary}".rstrip()


def changed_files_from_tool_event(
    tool_name: str,
    *,
    args_summary: dict[str, object],
    result_summary: dict[str, object],
    workspace_root: Path,
) -> list[ChangedFileSummary]:
    structured = result_summary.get("changed_files")
    if isinstance(structured, list):
        items = []
        for item in structured:
            if not isinstance(item, dict):
                continue
            path = _display_path(str(item.get("path") or "unknown"), workspace_root)
            additions = item.get("additions")
            deletions = item.get("deletions")
            if additions is not None and deletions is not None:
                summary = f"+{additions} -{deletions}"
            elif item.get("bytes_written") is not None:
                summary = f"{item['bytes_written']} bytes"
            elif item.get("replacements") is not None:
                summary = f"{item['replacements']} replacements"
            else:
                summary = "changed"
            items.append(
                ChangedFileSummary(
                    path=path,
                    change_type=str(item.get("change_type") or "modified"),
                    summary=summary,
                ),
            )
        if items:
            return items

    if tool_name == "file_write":
        path = _display_path(str(result_summary.get("path") or args_summary.get("path") or "unknown"), workspace_root)
        created = bool(result_summary.get("created"))
        mode = str(result_summary.get("mode") or args_summary.get("mode") or "")
        change_type = "added" if created or mode == "create" else "modified"
        bytes_written = result_summary.get("bytes_written")
        detail = f"{bytes_written} bytes" if bytes_written is not None else f"mode={mode or 'unknown'}"
        return [ChangedFileSummary(path=path, change_type=change_type, summary=detail)]

    if tool_name == "apply_patch":
        path = _display_path(str(result_summary.get("path") or args_summary.get("path") or "unknown"), workspace_root)
        replacements = result_summary.get("replacements")
        detail = f"{replacements} replacements" if replacements is not None else "text replacement"
        return [ChangedFileSummary(path=path, change_type="modified", summary=detail)]

    if tool_name == "apply_patch_set":
        paths = result_summary.get("paths") or args_summary.get("paths")
        if not isinstance(paths, list):
            paths = []
        count = result_summary.get("replacement_count") or args_summary.get("replacement_count") or len(paths)
        return [
            ChangedFileSummary(
                path=_display_path(str(path), workspace_root),
                change_type="modified",
                summary=f"{count} replacements in set",
            )
            for path in paths
        ]

    return []


def merge_changed_files(existing: list[ChangedFileSummary], new_items: list[ChangedFileSummary]) -> list[ChangedFileSummary]:
    merged = {item.path: item for item in existing}
    for item in new_items:
        merged[item.path] = item
    return list(merged.values())


def path_stays_in_workspace(path: str, workspace_root: Path) -> bool:
    root = workspace_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def render_changed_files(items: list[ChangedFileSummary], *, limit: int = 5) -> str:
    if not items:
        return f"  {EMPTY_LABELS['none']}"
    visible = items[-limit:]
    return "\n".join(item.line_text() for item in visible)


def _display_path(path: str, workspace_root: Path) -> str:
    root = workspace_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        return safe_summary(path.replace("\\", "/"), 160)
    try:
        resolved = candidate.resolve()
        if resolved == root:
            return "."
        if root in resolved.parents:
            return resolved.relative_to(root).as_posix()
    except OSError:
        pass
    return safe_summary(path, 160)
