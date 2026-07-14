"""
haagent/tui/overlays/search.py - 当前对话搜索 modal

提供不进入 conversation、不发送给模型的当前对话搜索界面。
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.tui.design.copy import MODAL_TITLES
from haagent.tui.state.search import ConversationSearchState


class SearchOverlay(ModalScreen[None]):
    def __init__(self, lines: list[str]) -> None:
        super().__init__()
        self.search_state = ConversationSearchState(lines)

    def compose(self) -> ComposeResult:
        yield Static(self._body(), id="search-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if key == "backspace":
            event.stop()
            self.search_state.update_query(self.search_state.query[:-1])
            self._refresh()
            return
        if key == "n":
            event.stop()
            if self.search_state.query:
                self.search_state.next_match()
            else:
                self.search_state.update_query("n")
            self._refresh()
            return
        if key in {"N", "shift+n", "upper_n"}:
            event.stop()
            self.search_state.previous_match()
            self._refresh()
            return
        if event.character and event.character.isprintable():
            event.stop()
            self.search_state.update_query(self.search_state.query + event.character)
            self._refresh()

    def _refresh(self) -> None:
        self.query_one("#search-dialog", Static).update(self._body())

    def _body(self) -> str:
        result = self.search_state.result()
        return "\n".join(
            [
                MODAL_TITLES["search"],
                "范围: conversation",
                f"关键词: {result.query or '-'}",
                result.status_text,
                "",
                "输入关键词  n 下一个  Shift+N 上一个  Esc 关闭",
            ],
        )
