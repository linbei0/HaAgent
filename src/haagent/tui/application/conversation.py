"""
src/haagent/tui/application/conversation.py - TUI 会话时间线协调器

集中承载对话时间线的写入、assistant streaming 批处理、滚动粘底、纯文本镜像
（供搜索使用）和工具活动/详情操作，让主 App 保持薄协调层。
"""

from __future__ import annotations

from typing import Any

from haagent.tui.design.copy import BLOCK_TITLES
from haagent.tui.design.utils import safe_summary
from haagent.tui.widgets import ConversationTimeline, ToolActivity, ToolStatus


class ConversationController:
    """把对话时间线的写入、streaming、滚动和纯文本镜像集中到一个对象。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        # 纯文本镜像：仅供会话内搜索使用，不作为渲染来源。
        self.lines: list[str] = []
        self.placeholder_rendered = False
        self.stick_to_bottom = True
        self.streaming_turn: int | None = None
        self.streaming_text = ""

    # ── 基础写入 ─────────────────────────────────────────────────────────
    def append_block(self, title: str, body: str, *, turn_index: int) -> None:
        display_title = BLOCK_TITLES.get(title, title)
        self.lines.append(f"{display_title}\n  {body}")
        if title == "You":
            self._timeline().add_user(body, turn_index=turn_index)
        elif title == "Assistant":
            self._timeline().add_assistant_message(body, turn_index=turn_index)
        elif title == "Failure":
            self._timeline().add_failure(body, turn_index=turn_index)
        else:
            self._timeline().add_system(display_title, body, turn_index=turn_index)

    def append_line(self, line: str, *, turn_index: int) -> None:
        self.lines.append(line)
        self._timeline().add_system("系统", line, turn_index=turn_index)

    # ── assistant streaming ──────────────────────────────────────────────
    def start_assistant(self, turn_index: int) -> None:
        self._timeline().start_assistant_response(turn_index=turn_index)

    def merge_assistant_delta(self, turn_index: int, delta: str) -> None:
        if not delta:
            return
        self._timeline().update_assistant_delta(turn_index, delta)
        if self.streaming_turn != turn_index:
            self.streaming_turn = turn_index
            self.streaming_text = delta
            return
        self.streaming_text += delta

    def finalize_assistant_message(self, turn_index: int, content: str) -> None:
        if self.streaming_turn == turn_index:
            self._timeline().finalize_assistant(turn_index, content or self.streaming_text)
        else:
            if self.streaming_turn is not None:
                self._timeline().finalize_assistant(self.streaming_turn, self.streaming_text)
            self._timeline().finalize_assistant(turn_index, content)
        self.streaming_turn = None
        self.streaming_text = ""

    def finalize_streaming_if_needed(self) -> None:
        if self.streaming_turn is None:
            return
        self._timeline().finalize_assistant(self.streaming_turn, self.streaming_text)
        self.streaming_turn = None
        self.streaming_text = ""

    def reset_streaming_state(self) -> None:
        self.lines = []
        self.placeholder_rendered = False
        self.streaming_turn = None
        self.streaming_text = ""

    # ── 工具活动 ─────────────────────────────────────────────────────────
    def record_tool_activity(self, turn_index: int, tool_name: str, status: str, summary: str) -> None:
        self._timeline().add_tool_activity(
            ToolActivity(
                turn_index=turn_index,
                tool_name=tool_name,
                status=_tool_status(status),
                summary=safe_summary(summary, 96),
            )
        )

    def record_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        self._timeline().add_tool_diagnostic(turn_index, tool_name, safe_summary(message, 120))

    def set_tool_details(self, enabled: bool) -> None:
        self._timeline().set_tool_details(enabled)

    # ── 刷新与滚动 ───────────────────────────────────────────────────────
    def refresh(self) -> None:
        """把 timeline 的粘底状态与 placeholder 同步到最新内容。"""
        conversation = self._timeline()
        if not conversation.plain_text:
            if not self.placeholder_rendered:
                conversation.show_placeholder()
                self.placeholder_rendered = True
            return
        self.placeholder_rendered = False
        if self._should_stick(conversation):
            self.stick_to_bottom = True
            conversation.set_stick_to_bottom(True)
            self._app.call_after_refresh(self._scroll_to_end_if_sticky)
        else:
            self.stick_to_bottom = False
            conversation.set_stick_to_bottom(False)

    def clear_placeholder_state(self) -> None:
        self.placeholder_rendered = False

    def stick_and_scroll_to_end(self) -> None:
        self.stick_to_bottom = True
        self._timeline().set_stick_to_bottom(True)
        self._scroll_to_end()

    def page_up(self) -> None:
        self.stick_to_bottom = False
        conversation = self._timeline()
        conversation.set_stick_to_bottom(False)
        conversation.scroll_page_up(animate=False, force=True)

    def page_down(self) -> None:
        conversation = self._timeline()
        conversation.scroll_page_down(animate=False, force=True)
        self._app.call_after_refresh(self.sync_stickiness)

    def sync_stickiness(self) -> None:
        conversation = self._timeline()
        self.stick_to_bottom = conversation.scroll_y >= conversation.max_scroll_y - 1
        conversation.set_stick_to_bottom(self.stick_to_bottom)

    def _scroll_to_end(self) -> None:
        conversation = self._timeline()
        conversation.set_stick_to_bottom(True)
        conversation.scroll_to(y=conversation.max_scroll_y, animate=False, immediate=True, force=True)

    def _scroll_to_end_if_sticky(self) -> None:
        conversation = self._timeline()
        if self._should_stick(conversation):
            self._scroll_to_end()

    def _should_stick(self, conversation: ConversationTimeline) -> bool:
        if not self.stick_to_bottom:
            return False
        return conversation.scroll_y >= conversation.max_scroll_y - 1

    def _timeline(self) -> ConversationTimeline:
        return self._app.query_one("#conversation", ConversationTimeline)


def _tool_status(status: str) -> ToolStatus:
    if status in {"started", "running", "pending"}:
        return "running"
    if status in {"approval", "approval_requested"}:
        return "approval"
    if status in {"finished", "done", "completed"}:
        return "done"
    if status in {"failed", "cancelled"}:
        return "failed"
    return "running"
