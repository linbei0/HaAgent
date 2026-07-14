"""
haagent/tui/overlays/sessions.py - session overlay 状态与界面

提供可搜索、可键盘操作的会话选择 modal，并把实际 session 操作交还 App 调用 service。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.app.assistant_types import AssistantSessionSummary
from haagent.tui.design.copy import EMPTY_LABELS, MODAL_TITLES
from haagent.tui.design.utils import safe_summary

SessionOverlayAction = Literal["resume", "continue_latest", "new"]


@dataclass(frozen=True)
class SessionOverlayResult:
    action: SessionOverlayAction
    session: AssistantSessionSummary | None = None


@dataclass(frozen=True)
class SessionOverlayState:
    sessions: list[AssistantSessionSummary]
    query: str = ""
    selected_index: int = 0

    @property
    def visible_sessions(self) -> list[AssistantSessionSummary]:
        needle = self.query.casefold()
        if not needle:
            return self.sessions
        return [
            session
            for session in self.sessions
            if needle in session.session_id.casefold() or needle in session.first_request.casefold()
        ]

    @property
    def selected_session(self) -> AssistantSessionSummary | None:
        visible = self.visible_sessions
        if not visible:
            return None
        index = min(max(self.selected_index, 0), len(visible) - 1)
        return visible[index]

    def with_query(self, query: str) -> SessionOverlayState:
        return replace(self, query=query, selected_index=0)

    def move(self, delta: int) -> SessionOverlayState:
        visible = self.visible_sessions
        if not visible:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        return replace(self, selected_index=next_index)

    def render(self) -> str:
        lines = [
            MODAL_TITLES["sessions"],
            f"搜索: {self.query or '-'}",
            "",
        ]
        visible = self.visible_sessions
        if not visible:
            lines.append(EMPTY_LABELS["no_matching_sessions"])
        for index, session in enumerate(visible):
            marker = ">" if index == min(self.selected_index, len(visible) - 1) else " "
            request = safe_summary(session.first_request or "-", 42)
            lines.append(f"{marker} {session.session_id}  turns:{session.turn_count}  {request}")
        lines.extend(["", "输入过滤  ↑/↓ 移动  Enter 恢复  l 继续最新  n 新建  Esc 关闭"])
        return "\n".join(lines)


class SessionOverlay(ModalScreen[SessionOverlayResult | None]):
    def __init__(self, sessions: list[AssistantSessionSummary]) -> None:
        super().__init__()
        self.state = SessionOverlayState(sessions=sessions)

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="sessions-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in {"escape"}:
            event.stop()
            self.dismiss(None)
            return
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if key == "backspace":
            event.stop()
            self._set_state(self.state.with_query(self.state.query[:-1]))
            return
        if key == "enter":
            event.stop()
            selected = self.state.selected_session
            if selected is not None:
                self.dismiss(SessionOverlayResult(action="resume", session=selected))
            return
        if key == "l":
            event.stop()
            self.dismiss(SessionOverlayResult(action="continue_latest"))
            return
        if key == "n":
            event.stop()
            self.dismiss(SessionOverlayResult(action="new"))
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.with_query(self.state.query + event.character))

    def _set_state(self, state: SessionOverlayState) -> None:
        self.state = state
        self.query_one("#sessions-dialog", Static).update(state.render())
