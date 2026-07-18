"""
src/haagent/tui/application/session_flow.py - TUI 会话生命周期流程

从主 App 迁出新建/恢复/继续会话、session overlay 结果处理、历史回放和清空对话逻辑。
会话状态全部经 AssistantService 管理，前端只负责触发和展示。
磁盘 I/O 与 session 重建在 worker 线程执行，避免阻塞 Textual 事件循环。
"""

from __future__ import annotations

from typing import Any

from haagent.app.assistant_types import (
    AssistantSessionStatus,
    AssistantSessionSummary,
    AssistantSessionTurn,
)
from haagent.tui.overlays.sessions import SessionOverlay, SessionOverlayResult
from haagent.tui.widgets import ConversationTimeline


class SessionFlow:
    """封装会话新建、恢复、继续和历史回放的交互流程。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self._busy = False

    def restore_initial_session(self) -> None:
        """启动时按 --resume / --continue 参数恢复会话。"""
        initial_resume = self._app.service.sessions.initial_resume
        if initial_resume is not None:
            try:
                status = self._app.service.sessions.resume(initial_resume)
            except Exception as error:
                self._app._conversation.append_block("Session warning", f"恢复会话失败：{error}")
            else:
                self.show_session_history(status, prefix="已恢复 session")
            return
        if not self._app.service.sessions.initial_continue:
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
        if self._busy:
            return
        # 列表扫描可能触碰大量 session 目录，放到 worker 避免卡 UI。
        self._app._load_session_list_worker()

    def open_sessions_with_list(self, sessions: list[AssistantSessionSummary]) -> None:
        self._app.push_screen(
            SessionOverlay(sessions),
            self.handle_session_overlay_result,
        )

    def handle_session_list_error(self, error: Exception) -> None:
        self._app._conversation.append_block("Session warning", f"加载会话列表失败：{error}")
        self._app._refresh()

    def new_session(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._app._run_session_create_worker()

    def resume_latest(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._app._run_session_continue_worker()

    def handle_session_overlay_result(self, result: SessionOverlayResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        if self._busy:
            self._app._defer_prompt_focus()
            return
        self._busy = True
        self._app._run_session_overlay_worker(result)

    def apply_create_success(self) -> None:
        self._busy = False
        self._app._reset_image_input_state()
        self.clear_conversation_for_new_session()
        self._app._refresh()
        self._app._defer_prompt_focus()

    def apply_continue_success(self, status: AssistantSessionStatus) -> None:
        self._busy = False
        self._app._reset_image_input_state()
        self._app.clear_context_usage()
        try:
            history = list(self._app.service.sessions.history())
        except Exception as error:
            self._app._prompt_input().clear_request_history()
            self._app._conversation.append_block("Session warning", f"恢复会话历史读取失败：{error}")
        else:
            self._set_prompt_history(history)
        self._app._conversation.append_line(f"已恢复会话：{status.session_id}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def apply_overlay_success(self, result: SessionOverlayResult, status: AssistantSessionStatus) -> None:
        self._busy = False
        self._app._reset_image_input_state()
        if result.action == "new":
            self.clear_conversation_for_new_session()
        else:
            self.show_session_history(status, prefix="当前会话")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def apply_session_error(self, message: str) -> None:
        self._busy = False
        self._app._conversation.append_block("Session warning", message)
        self._app._refresh()
        self._app._defer_prompt_focus()

    def show_session_history(self, status: AssistantSessionStatus, *, prefix: str) -> None:
        self._app.clear_context_usage()
        conversation = self._timeline()
        try:
            history = list(self._app.service.sessions.history())
        except Exception as error:
            conversation.clear_timeline()
            conversation.add_system("会话", f"{prefix}历史读取失败：{error}")
            history = []
        else:
            # 批量装载只同步一次 DOM，避免 N 轮 2N 次全量渲染。
            conversation.load_session_history(history)
        self._set_prompt_history(history)
        self._app._conversation.reset_streaming_state()

    def clear_conversation_for_new_session(self) -> None:
        self._app.clear_context_usage()
        self._app._conversation.reset_streaming_state()
        self._app._active_turn_index = None
        self._app._prompt_input().clear_request_history()
        self._timeline().clear_timeline()

    def _set_prompt_history(self, history: list[AssistantSessionTurn]) -> None:
        # Session turn 是持久化事实源，输入组件只保存当前会话的可浏览请求文本。
        requests = [turn.request for turn in history if turn.request]
        self._app._prompt_input().set_request_history(requests)

    def _timeline(self) -> ConversationTimeline:
        return self._app.query_one("#conversation", ConversationTimeline)


def session_turn_assistant_text(turn: AssistantSessionTurn) -> str:
    """从会话历史 turn 提取用于展示的 assistant 文本。"""
    if turn.assistant_display_text and turn.assistant_display_text.strip():
        return turn.assistant_display_text.strip()
    summary = turn.summary.strip()
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
