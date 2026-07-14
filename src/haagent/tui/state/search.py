"""
haagent/tui/state/search.py - 当前对话搜索状态

维护搜索关键字、匹配位置和导航结果，避免搜索行为污染对话流。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    query: str
    count: int
    current: int
    current_line: int | None

    @property
    def status_text(self) -> str:
        if not self.query:
            return "输入关键词搜索当前对话"
        if self.count == 0:
            return f"无匹配：{self.query}"
        return f"{self.current + 1}/{self.count}：{self.query}"


class ConversationSearchState:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.query = ""
        self._matches: list[int] = []
        self._current = 0

    def update_query(self, query: str) -> SearchResult:
        self.query = query
        needle = query.casefold()
        self._matches = []
        if needle:
            for index, line in enumerate(self._lines):
                count = line.casefold().count(needle)
                self._matches.extend([index] * count)
        self._current = 0
        return self.result()

    def next_match(self) -> SearchResult:
        if self._matches:
            self._current = (self._current + 1) % len(self._matches)
        return self.result()

    def previous_match(self) -> SearchResult:
        if self._matches:
            self._current = (self._current - 1) % len(self._matches)
        return self.result()

    def result(self) -> SearchResult:
        current_line = self._matches[self._current] if self._matches else None
        return SearchResult(
            query=self.query,
            count=len(self._matches),
            current=self._current,
            current_line=current_line,
        )
