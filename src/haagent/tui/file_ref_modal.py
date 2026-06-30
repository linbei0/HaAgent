"""
haagent/tui/file_ref_modal.py - @file 引用选择面板

展示 workspace 内文件匹配结果，作为输入框上方的轻量候选列表。
"""

from __future__ import annotations

from pathlib import Path

from textual import events, work
from textual.widgets import Static

from haagent.tui.copy import EMPTY_LABELS, MODAL_TITLES
from haagent.tui.file_refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, path_reference_token


VISIBLE_MATCH_COUNT = 4


class FileReferenceOverlay(Static):
    def __init__(self, workspace_root: Path, query: str, index: FileReferenceIndex | None = None) -> None:
        super().__init__("", id="file-ref-dialog")
        self.workspace_root = workspace_root
        self.filter_text = query
        self.selected_index = 0
        self._match_scroll_offset = 0
        self.index = index
        self.matches: list[FileReferenceMatch] = []
        self.loading = index is None
        self._mounted = False

    def on_mount(self) -> None:
        self._mounted = True
        if self.index is None:
            self.update(self._body())
            self._build_index()
        else:
            self._reload()

    def on_unmount(self) -> None:
        self._mounted = False

    def handle_key(self, event: events.Key) -> str | None:
        key = event.key
        if key == "escape":
            event.stop()
            return ""
        if key == "up":
            event.stop()
            self._move(-1)
            return None
        if key == "down":
            event.stop()
            self._move(1)
            return None
        if key == "enter":
            event.stop()
            return self.selected_token()
        return None

    def selected_token(self) -> str | None:
        selected = self._selected()
        if selected is None:
            return None
        return path_reference_token(self.workspace_root, selected.path)

    def update_query(self, query: str) -> None:
        self.filter_text = query
        self._reload()

    def _selected(self) -> FileReferenceMatch | None:
        if not self.matches:
            return None
        return self.matches[min(self.selected_index, len(self.matches) - 1)]

    def _move(self, delta: int) -> None:
        if self.matches:
            self.selected_index = min(max(self.selected_index + delta, 0), len(self.matches) - 1)
            self._ensure_selection_visible()
        self._refresh()

    def _reload(self) -> None:
        self.matches = self.index.matches(self.filter_text) if self.index is not None else []
        self.selected_index = 0
        self._match_scroll_offset = 0
        self._refresh()

    def _refresh(self) -> None:
        if self.is_mounted:
            self.update(self._body())

    def _ensure_selection_visible(self) -> None:
        if self.selected_index < self._match_scroll_offset:
            self._match_scroll_offset = self.selected_index
        elif self.selected_index >= self._match_scroll_offset + VISIBLE_MATCH_COUNT:
            self._match_scroll_offset = self.selected_index - VISIBLE_MATCH_COUNT + 1

    @work(thread=True, exclusive=True)
    def _build_index(self) -> None:
        index = build_file_reference_index(self.workspace_root)
        self.app.call_from_thread(self._handle_index_ready, index)

    def _handle_index_ready(self, index: FileReferenceIndex) -> None:
        if not self._mounted:
            return
        self.index = index
        self.loading = False
        self._reload()

    def _body(self) -> str:
        lines = [MODAL_TITLES["file_refs"], f"搜索: {self.filter_text or '-'}", ""]
        if self.loading:
            lines.append("正在搜索文件...")
            lines.extend(["", "↑/↓ 移动  Enter 插入引用  Esc 关闭"])
            return "\n".join(lines)
        if not self.matches:
            lines.append(EMPTY_LABELS["no_matching_files"])
        visible_matches = self.matches[self._match_scroll_offset : self._match_scroll_offset + VISIBLE_MATCH_COUNT]
        for offset, match in enumerate(visible_matches):
            index = self._match_scroll_offset + offset
            marker = ">" if index == self.selected_index else " "
            lines.append(f"{marker} {match.display_path}")
        lines.extend(["", "↑/↓ 移动  Enter 插入引用  Esc 关闭"])
        return "\n".join(lines)
