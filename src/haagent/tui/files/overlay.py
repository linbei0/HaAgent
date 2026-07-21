"""
haagent/tui/files/overlay.py - @file 引用选择面板

展示 workspace 内文件匹配结果，作为输入框上方的轻量候选列表。
"""

from __future__ import annotations

from pathlib import Path

from textual import events, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from haagent.tui.design.copy import EMPTY_LABELS, MODAL_TITLES
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, path_reference_token


VISIBLE_MATCH_COUNT = 4


class FileReferenceOverlay(Vertical):
    class Selected(Message):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    def __init__(self, workspace_root: Path, query: str, index: FileReferenceIndex | None = None) -> None:
        super().__init__(id="file-ref-dialog")
        self.workspace_root = workspace_root
        self.filter_text = query
        self.selected_index = 0
        self._match_scroll_offset = 0
        self.index = index
        self.matches: list[FileReferenceMatch] = []
        self.loading = index is None
        self._mounted = False

    def compose(self) -> ComposeResult:
        yield Static("", id="file-ref-summary")
        yield OptionList(id="file-ref-list")

    def on_mount(self) -> None:
        self._mounted = True
        if self.index is None:
            self._refresh()
            self._build_index()
        else:
            self._reload()
        try:
            self.query_one(OptionList).focus()
        except NoMatches:
            return

    def on_unmount(self) -> None:
        self._mounted = False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        token = self.selected_token()
        if token is not None:
            event.stop()
            self.post_message(self.Selected(token))

    def handle_navigation_key(self, event: events.Key) -> str | None:
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
        if self.is_mounted:
            index = self.query_one(OptionList).highlighted
            if index is not None:
                self.selected_index = min(max(index, 0), len(self.matches) - 1)
        return self.matches[min(self.selected_index, len(self.matches) - 1)]

    def _move(self, delta: int) -> None:
        # 方向键只改索引/高亮与摘要；禁止每次 set_options 全量重建。
        if self.matches:
            self.selected_index = min(max(self.selected_index + delta, 0), len(self.matches) - 1)
            self._ensure_selection_visible()
        self._refresh_header_and_highlight()

    def _reload(self) -> None:
        self.matches = self.index.matches(self.filter_text) if self.index is not None else []
        self.selected_index = 0
        self._match_scroll_offset = 0
        self._refresh()

    def _refresh(self) -> None:
        try:
            summary = self.query_one("#file-ref-summary", Static)
            option_list = self.query_one(OptionList)
        except NoMatches:
            return
        summary.update(self._body())
        if self.loading:
            option_list.set_options([Option("正在搜索文件...", id="loading", disabled=True)])
            option_list.highlighted = None
            return
        options = [Option(match.display_path, id=match.display_path) for match in self.matches]
        if not options:
            options = [Option(EMPTY_LABELS["no_matching_files"], id="empty", disabled=True)]
        option_list.set_options(options)
        option_list.highlighted = self.selected_index if self.matches else None

    def _refresh_header_and_highlight(self) -> None:
        try:
            summary = self.query_one("#file-ref-summary", Static)
            option_list = self.query_one(OptionList)
        except NoMatches:
            return
        summary.update(self._body())
        option_list.highlighted = self.selected_index if self.matches else None

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
