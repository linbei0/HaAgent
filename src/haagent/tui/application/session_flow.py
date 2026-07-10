"""
src/haagent/tui/application/session_flow.py - TUI 会话生命周期流程

从主 App 迁出新建/恢复/继续会话、session overlay 结果处理、历史回放和清空对话逻辑。
会话状态全部经 AssistantService 管理，前端只负责触发和展示。
"""

from __future__ import annotations

from typing import Any

from haagent.tui.overlays.sessions import SessionOverlay, SessionOverlayResult
from haagent.tui.widgets import ConversationTimeline


class SessionFlow:
    """封装会话新建、恢复、继续和历史回放的交互流程。"""

    def __init__(self, app: Any) -> None:
        self._app = app

    def restore_initial_session(self) -> None:
        """启动时按 --resume / --continue 参数恢复会话。"""
        initial_resume = getattr(self._app.service.sessions, "initial_resume", None)
        if initial_resume is not None:
            try:
                status = self._app.service.sessions.resume(initial_resume)
            except Exception as error:
                self._app._conversation.append_block("Session warning", f"恢复会话失败：{error}")
            else:
                self.show_session_history(status, prefix="已恢复 session")
            return
        if not bool(getattr(self._app.service.sessions, "initial_continue", False)):
            return
        try:
            status = self._app.service.sessions.continue_latest()
        except Exception as error:
            self._app._conversation.append_block("Session warning", f"继续最新 session 失败：{error}")
        else:
            self.show_session_history(status, prefix="已继续最新 session")

    def open_sessions(self) -> None:
        if self._app._prompt_has_pending_text():
            self._app.query_one("#prompt-input").insert("s")
            return
        self._app.push_screen(
            SessionOverlay(self._app.service.sessions.list()),
            self.handle_session_overlay_result,
        )

    def new_session(self) -> None:
        try:
            self._app.service.sessions.create()
        except Exception as error:
            self._app._conversation.append_block("Session warning", f"新建会话失败：{error}")
        else:
            self._app._reset_image_input_state()
            self.clear_conversation_for_new_session()
        self._app._refresh()

    def resume_latest(self) -> None:
        try:
            status = self._app.service.sessions.continue_latest()
        except Exception as error:
            self._app._conversation.append_block("Session warning", f"继续最新会话失败：{error}")
        else:
            self._app._reset_image_input_state()
            self._app._conversation.append_line(f"已恢复会话：{status.session_id}")
        self._app._refresh()

    def handle_session_overlay_result(self, result: SessionOverlayResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        try:
            if result.action == "resume" and result.session is not None:
                status = self._app.service.sessions.resume(result.session.session_path)
            elif result.action == "continue_latest":
                status = self._app.service.sessions.continue_latest()
            else:
                status = self._app.service.sessions.create()
        except Exception as error:
            self._app._conversation.append_block("Session warning", f"会话操作失败：{error}")
        else:
            self._app._reset_image_input_state()
            if result.action == "new":
                self.clear_conversation_for_new_session()
            else:
                self.show_session_history(status, prefix="当前会话")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def show_session_history(self, status: object, *, prefix: str) -> None:
        conversation = self._timeline()
        conversation.clear_timeline()
        try:
            history = list(self._app.service.sessions.history())
        except Exception as error:
            conversation.add_system("会话", f"{prefix}历史读取失败：{error}")
            history = []
        for turn in history:
            assistant_text = session_turn_assistant_text(turn)
            conversation.add_user(turn.request, turn_index=turn.turn_index)
            conversation.add_assistant_message(assistant_text, turn_index=turn.turn_index)
            if turn.status != "completed":
                conversation.add_system("状态", f"状态：{turn.status}", turn_index=turn.turn_index)
        self._app._conversation.reset_streaming_state()

    def clear_conversation_for_new_session(self) -> None:
        self._app._conversation.reset_streaming_state()
        self._app._active_turn_index = None
        self._timeline().clear_timeline()

    def _timeline(self) -> ConversationTimeline:
        return self._app.query_one("#conversation", ConversationTimeline)


def session_turn_assistant_text(turn: object) -> str:
    """从会话历史 turn 提取用于展示的 assistant 文本。"""
    for field_name in ("assistant_display_text", "assistant_final_response", "final_response", "response", "content"):
        value = getattr(turn, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    summary = str(getattr(turn, "summary", "")).strip()
    extracted = _summary_field(summary, "assistant_final_response") or _summary_field(summary, "final_response")
    return extracted or summary


def _summary_field(summary: str, field_name: str) -> str:
    for line in summary.splitlines():
        normalized = line.strip()
        if normalized.startswith("- "):
            normalized = normalized[2:].lstrip()
        colon_prefix = f"{field_name}:"
        equals_prefix = f"{field_name}="
        if normalized.startswith(colon_prefix):
            return normalized.removeprefix(colon_prefix).strip()
        if normalized.startswith(equals_prefix):
            return normalized.removeprefix(equals_prefix).strip()
    return ""
