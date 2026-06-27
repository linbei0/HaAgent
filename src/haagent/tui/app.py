"""
haagent/tui/app.py - HaAgent Textual 首版界面

提供显式 `haagent tui` 的最小垂直切片，只通过 AssistantService 驱动会话。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RichLog, Static

from haagent.app.assistant_service import AssistantService, AssistantWorkspaceStatus
from haagent.memory import MemoryCandidate
from haagent.runtime.command import redact_secret_like_text
from haagent.runtime.chat_session import ChatEvent
from haagent.runtime.human_interaction import (
    HumanInteractionRequest,
    HumanInteractionResponse,
)


@dataclass
class _PendingInteraction:
    request: HumanInteractionRequest
    done: threading.Event = field(default_factory=threading.Event)
    response: HumanInteractionResponse | None = None


class HelpModal(ModalScreen[None]):
    CSS = """
    HelpModal {
        align: center middle;
    }

    #help-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: solid $primary;
        background: $surface;
    }

    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #help-body {
        margin-bottom: 1;
    }
    """

    BINDINGS = [("escape", "dismiss_help", "关闭")]

    def __init__(self, context: str) -> None:
        super().__init__()
        self.context = context

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("HaAgent Help", id="help-title")
            yield Static(_help_body(self.context), id="help-body")
            yield Static("[Esc]关闭")

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


class ToolApprovalModal(ModalScreen[bool]):
    CSS = """
    ToolApprovalModal {
        align: center middle;
    }

    #approval-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: solid $warning;
        background: $surface;
    }

    #approval-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #approval-body {
        margin-bottom: 1;
    }

    #approval-buttons {
        align-horizontal: center;
        height: auto;
    }

    #approval-allow,
    #approval-deny {
        margin: 0 2;
    }
    """

    BINDINGS = [
        ("y", "allow", "允许"),
        ("n", "deny", "拒绝"),
        ("escape", "deny", "拒绝"),
        ("?", "help", "帮助"),
    ]

    def __init__(self, request: HumanInteractionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static("Tool Approval", id="approval-title")
            yield Static(Text(_approval_body(self.request)), id="approval-body")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow y", id="approval-allow", variant="success")
                yield Button("Deny n", id="approval-deny", variant="error")

    def on_mount(self) -> None:
        self.query_one("#approval-deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approval-allow")

    def on_key(self, event: events.Key) -> None:
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            self.action_help()

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def action_help(self) -> None:
        self.app.push_screen(HelpModal("approval"))


class PromptInput(Input):
    def on_key(self, event: events.Key) -> None:
        if self.value:
            return
        app = self.app
        if not isinstance(app, HaAgentTuiApp):
            return
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            app.action_help()
            return
        if event.key != "m" or app._pending_interaction is not None:
            return
        if isinstance(app, HaAgentTuiApp):
            event.stop()
            app.action_toggle_memory()


class HaAgentTuiApp(App[None]):
    MIN_WIDTH = 80
    MIN_HEIGHT = 24

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    #main {
        height: 1fr;
    }

    #conversation {
        width: 1fr;
        height: 1fr;
        padding: 1;
        border: solid $primary;
        overflow-y: auto;
    }

    #side-bar {
        width: 36;
        height: 1fr;
        padding: 1;
        border: solid $primary;
        overflow-y: auto;
    }

    #input-panel {
        height: 3;
        padding: 0 1;
    }

    #prompt-input {
        width: 1fr;
    }

    #footer-bar {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    #resize-message {
        height: 1fr;
        content-align: center middle;
        padding: 1 2;
        border: solid $warning;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        ("?", "help", "帮助"),
        Binding("m", "toggle_memory", "记忆", priority=True),
        ("enter", "memory_enter", "详情"),
        ("a", "confirm_memory", "确认记忆"),
        ("y", "confirm_memory", "确认记忆"),
        ("r", "reject_memory", "拒绝记忆"),
        ("escape", "cancel_interaction", "取消"),
        ("pageup", "conversation_page_up", "上翻"),
        ("pagedown", "conversation_page_down", "下翻"),
    ]

    def __init__(self, service: AssistantService) -> None:
        super().__init__()
        self.service = service
        self._state = "idle"
        self._conversation_lines: list[str] = []
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = False
        self._tool_lines: list[str] = []
        self._last_failure: dict[str, str] | None = None
        self._pending_interaction: _PendingInteraction | None = None
        self._default_prompt_placeholder = "输入 prompt，Enter 发送"
        self._memory_mode = False
        self._memory_detail_mode = False
        self._memory_candidates: list[MemoryCandidate] = []
        self._memory_selected = 0
        self._memory_error: str | None = None
        self._memory_notice: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        yield Static("终端尺寸过小\n请调整到至少 80x24 后继续使用 HaAgent TUI。", id="resize-message", classes="hidden")
        with Horizontal(id="main"):
            yield RichLog(id="conversation", wrap=True, auto_scroll=True)
            yield Static("", id="side-bar")
        with Vertical(id="input-panel"):
            yield PromptInput(placeholder=self._default_prompt_placeholder, id="prompt-input")
        yield Static(Text("[Enter]发送 [PgUp/PgDn]滚动 [Tab]焦点 [?]帮助 [Ctrl+Q]退出"), id="footer-bar")

    def on_mount(self) -> None:
        self.query_one("#side-bar", Static).can_focus = True
        self._show_initial_configuration_state()
        self._refresh()
        self._update_responsive_layout()
        self.query_one("#prompt-input", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(width=event.size.width, height=event.size.height)

    def on_key(self, event: events.Key) -> None:
        if self._memory_mode and self._pending_interaction is None:
            handled = self._handle_memory_key(event.key)
            if handled:
                event.stop()
                return
        if self._pending_interaction is not None:
            return
        prompt_input = self.query_one("#prompt-input", Input)
        if not prompt_input.has_focus or prompt_input.value:
            return
        if event.key == "m":
            event.stop()
            self.action_toggle_memory()
        elif event.key == "enter" and self._memory_mode:
            event.stop()
            self.action_memory_enter()
        elif event.key in {"a", "y"} and self._memory_mode:
            event.stop()
            self.action_confirm_memory()
        elif event.key == "r" and self._memory_mode:
            event.stop()
            self.action_reject_memory()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        if self._pending_interaction is not None and self._pending_interaction.request.interaction_type == "user_input":
            self._complete_interaction(HumanInteractionResponse(approved=True, answer=prompt))
            return
        self._append_block("You", prompt)
        self._state = "running"
        self._refresh()
        self._run_prompt(prompt)

    def action_help(self) -> None:
        self.push_screen(HelpModal(self._help_context()))

    def action_quit(self) -> None:
        self.exit(None)

    def action_conversation_page_up(self) -> None:
        self.query_one("#conversation", RichLog).scroll_page_up(animate=False, force=True)

    def action_conversation_page_down(self) -> None:
        self.query_one("#conversation", RichLog).scroll_page_down(animate=False, force=True)

    def action_cancel_interaction(self) -> None:
        if self._memory_mode:
            if self._memory_detail_mode:
                self._memory_detail_mode = False
            else:
                self._memory_mode = False
                self.query_one("#prompt-input", Input).focus()
            self._refresh()
            return
        if self._pending_interaction is None:
            return
        self._complete_interaction(HumanInteractionResponse(approved=False, answer=""))

    def action_toggle_memory(self) -> None:
        self._memory_mode = not self._memory_mode
        self._memory_detail_mode = False
        if self._memory_mode:
            self._load_memory_candidates()
            self.query_one("#prompt-input", Input).value = ""
            self.query_one("#side-bar", Static).focus()
        else:
            self.query_one("#prompt-input", Input).focus()
        self._refresh()

    def action_memory_enter(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        self._memory_detail_mode = not self._memory_detail_mode
        self._refresh()

    def action_memory_up(self) -> None:
        self._move_memory_selection(-1)

    def action_memory_down(self) -> None:
        self._move_memory_selection(1)

    def action_memory_first(self) -> None:
        if self._memory_mode and self._memory_candidates and not self._memory_detail_mode:
            self._memory_selected = 0
            self._refresh()

    def action_memory_last(self) -> None:
        if self._memory_mode and self._memory_candidates and not self._memory_detail_mode:
            self._memory_selected = len(self._memory_candidates) - 1
            self._refresh()

    def action_confirm_memory(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        candidate = self._selected_memory_candidate()
        try:
            self.service.confirm_memory_candidate(candidate.candidate_id)
        except Exception as error:
            self._memory_notice = f"Memory confirm failed: {error}"
            self._append_block("Memory warning", f"Memory confirm failed: {error}")
        else:
            self._memory_notice = f"Memory confirmed: {candidate.candidate_id}"
            self._append_line(f"Memory confirmed: {candidate.candidate_id}")
        self._memory_detail_mode = False
        self._load_memory_candidates()
        self._refresh()

    def action_reject_memory(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        candidate = self._selected_memory_candidate()
        try:
            self.service.reject_memory_candidate(candidate.candidate_id, "rejected from TUI")
        except Exception as error:
            self._memory_notice = f"Memory reject failed: {error}"
            self._append_block("Memory warning", f"Memory reject failed: {error}")
        else:
            self._memory_notice = f"Memory rejected: {candidate.candidate_id}"
            self._append_line(f"Memory rejected: {candidate.candidate_id}")
        self._memory_detail_mode = False
        self._load_memory_candidates()
        self._refresh()

    @work(thread=True, exclusive=True)
    def _run_prompt(self, prompt: str) -> None:
        try:
            result = self.service.run_prompt_events(
                prompt,
                event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                interaction_handler=self._handle_interaction,
            )
        except Exception as error:
            self.call_from_thread(self._handle_prompt_error, error)
            return
        status = str(getattr(result, "status", "completed"))
        self.call_from_thread(self._finish_prompt, status)

    def _handle_chat_event(self, event: ChatEvent) -> None:
        event_type = event.event_type
        payload = event.payload
        if event_type == "assistant_message":
            self._append_block("Assistant", _payload_text(payload, "content", event.message))
        elif event_type == "tool_started":
            line = f"Tool {_payload_text(payload, 'tool_name', 'unknown')} ..."
            self._tool_lines.append(line)
            self._append_line(line)
        elif event_type == "tool_finished":
            line = f"Tool {_payload_text(payload, 'tool_name', 'unknown')} done"
            self._tool_lines.append(line)
            self._append_line(line)
        elif event_type == "tool_failed":
            line = f"Tool {_payload_text(payload, 'tool_name', 'unknown')} failed"
            self._tool_lines.append(line)
            self._append_line(line)
        elif event_type == "approval_requested":
            self._state = "waiting approval"
            tool_name = _payload_text(payload, "tool_name", "unknown")
            line = f"{tool_name} pending approval"
            self._tool_lines.append(line)
            self._append_line(f"Tool {line}")
        elif event_type == "user_input_requested":
            self._state = "waiting input"
            question = _payload_text(payload, "question", event.message)
            self._set_answer_required(question)
            self._append_block("Answer required", question)
        elif event_type == "approval_granted":
            tool_name = _payload_text(payload, "tool_name", "unknown")
            self._state = "running"
            self._tool_lines.append(f"{tool_name} approved")
            self._append_line(f"Approval granted: {tool_name}")
        elif event_type == "approval_denied":
            tool_name = _payload_text(payload, "tool_name", "unknown")
            self._tool_lines.append(f"{tool_name} denied")
            self._append_line(f"Approval denied: {tool_name}")
        elif event_type == "user_input_received":
            tool_name = _payload_text(payload, "tool_name", "request_user_input")
            approved = payload.get("approved")
            self._state = "running"
            if approved is False:
                self._append_line(f"Answer declined: {tool_name}")
            else:
                self._append_line(f"Answer submitted: {tool_name}")
        elif event_type == "memory_candidates_created":
            message = _payload_text(
                payload,
                "message",
                event.message or "发现可记忆候选，已放入候选队列，等待你确认。",
            )
            self._append_block("Memory", message)
            self._memory_notice = message
            self._memory_mode = True
            self._memory_detail_mode = False
            self._load_memory_candidates(silent=True)
        elif event_type == "memory_extraction_warning":
            self._append_block("Memory warning", _payload_text(payload, "message", event.message))
        elif event_type == "failure":
            self._state = "failed"
            reason = _payload_text(payload, "reason", event.message)
            failed_stage = _payload_text(payload, "failed_stage", "unknown")
            category = _payload_text(payload, "failure_category", "unknown")
            episode_path = _payload_text(payload, "episode_path", "unknown")
            self._last_failure = {
                "failed_stage": failed_stage,
                "failure_category": category,
                "reason": reason,
                "episode_path": episode_path,
            }
            self._append_block("Failure", _failure_body(failed_stage, category, reason, episode_path))
        self._refresh()

    def _handle_interaction(self, request: HumanInteractionRequest) -> HumanInteractionResponse:
        pending = _PendingInteraction(request)
        self.call_from_thread(self._begin_interaction, pending)
        pending.done.wait()
        return pending.response or HumanInteractionResponse(approved=False, answer="")

    def _begin_interaction(self, pending: _PendingInteraction) -> None:
        self._pending_interaction = pending
        request = pending.request
        if request.interaction_type == "approval":
            self._state = "waiting approval"
            self.push_screen(ToolApprovalModal(request), self._complete_approval)
        else:
            self._state = "waiting input"
            self._set_answer_required(request.question)
        self._refresh()

    def _complete_approval(self, approved: bool | None) -> None:
        self._complete_interaction(HumanInteractionResponse(approved=bool(approved), answer=""))

    def _complete_interaction(self, response: HumanInteractionResponse) -> None:
        pending = self._pending_interaction
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._pending_interaction = None
        self._restore_prompt_input()
        self._state = "running"
        self._refresh()

    def _set_answer_required(self, question: str) -> None:
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.placeholder = f"回答 Agent 的问题：{_safe_summary(question, 90)}"
        prompt_input.focus()

    def _restore_prompt_input(self) -> None:
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.placeholder = self._default_prompt_placeholder

    def _finish_prompt(self, status: str) -> None:
        if status == "completed" and self._state not in {"waiting approval", "waiting input"}:
            self._state = "idle"
        elif status != "completed":
            self._state = "failed"
        self._refresh()

    def _handle_prompt_error(self, error: Exception) -> None:
        self._state = "failed"
        self._append_block("Failure", str(error))
        self._refresh()

    def _show_initial_configuration_state(self) -> None:
        status = self.service.get_workspace_status()
        if status.profile_error is not None:
            self._append_block("Config", "未找到默认模型配置\n请先运行：uv run haagent setup")
        elif status.credential_store_available is False:
            reason = status.credential_store_error or "unknown"
            self._append_block(
                "Config",
                (
                    f"系统凭据库不可用：{reason}\n"
                    "请运行：uv run haagent setup 重新选择凭据来源。"
                ),
            )
        elif status.api_key_env and not status.api_key_available:
            self._append_block(
                "Config",
                (
                    f"API key 缺失：{status.api_key_env}\n"
                    "HaAgent 不会在 TUI 中输入、保存或显示真实 API key。"
                ),
            )

    def _refresh(self) -> None:
        status = self.service.get_workspace_status()
        self.query_one("#status-bar", Static).update(self._status_line(status))
        self._refresh_conversation()
        self.query_one("#side-bar", Static).update(self._side_bar(status))
        self.query_one("#footer-bar", Static).update(Text(self._footer_text()))

    def _help_context(self) -> str:
        if self._pending_interaction is not None:
            if self._pending_interaction.request.interaction_type == "user_input":
                return "pending_input"
            return "approval"
        if self._memory_mode:
            return "memory_detail" if self._memory_detail_mode else "memory_list"
        return "chat"

    def _refresh_conversation(self) -> None:
        conversation = self.query_one("#conversation", RichLog)
        if self._memory_mode:
            conversation.clear()
            conversation.write(Text(self._memory_panel_text()), scroll_end=True, animate=False)
            self._conversation_placeholder_rendered = False
            self._conversation_rendered_count = 0
            return
        if not self._conversation_lines:
            if not self._conversation_placeholder_rendered:
                conversation.clear()
                conversation.write(Text("Ready. 输入 prompt 后按 Enter 发送；Ctrl+Q 退出。"), scroll_end=True, animate=False)
                self._conversation_placeholder_rendered = True
            return
        if self._conversation_placeholder_rendered or self._conversation_rendered_count > len(self._conversation_lines):
            conversation.clear()
            self._conversation_rendered_count = 0
            self._conversation_placeholder_rendered = False
        for line in self._conversation_lines[self._conversation_rendered_count :]:
            conversation.write(Text(line), scroll_end=True, animate=False)
        self._conversation_rendered_count = len(self._conversation_lines)
        self.call_after_refresh(self._scroll_conversation_to_end)

    def _scroll_conversation_to_end(self) -> None:
        conversation = self.query_one("#conversation", RichLog)
        conversation.scroll_to(y=conversation.max_scroll_y, animate=False, immediate=True, force=True)

    def _status_line(self, status: AssistantWorkspaceStatus) -> str:
        width = max(1, self.size.width or 120)
        profile = status.profile_name or "missing"
        provider = status.provider or "-"
        model = status.model or "-"
        key_state = _compact_key_state(status)
        session = status.current_session_id or "-"
        turn_count = status.current_turn_count if status.current_turn_count is not None else 0
        workspace_limit = 5 if width <= 80 else 16
        profile_limit = 12
        provider_limit = 14
        model_limit = 4 if width <= 80 else 14
        session_limit = 4 if width <= 80 else 12
        line = (
            f"ws:{_workspace_label(status.workspace_root, workspace_limit)} "
            f"profile: {_truncate_end(profile, profile_limit)} "
            f"{_truncate_end(provider, provider_limit)}/{_truncate_end(model, model_limit)} "
            f"key: {key_state} "
            f"sid:{_short_session(session, session_limit)} "
            f"turn:{turn_count} "
            f"state: {self._state}"
        )
        return _truncate_status_line(line, width)

    def _side_bar(self, status: AssistantWorkspaceStatus) -> str:
        if self._memory_mode:
            return self._memory_side_bar()
        profile = status.profile_name or "missing"
        provider = status.provider or "-"
        base_url = status.base_url or "-"
        model = status.model or "-"
        api_key_env = status.api_key_env or "-"
        key_state = _key_state(status)
        keyring_status = _keyring_status(status)
        session = status.current_session_id or "-"
        turn_count = status.current_turn_count if status.current_turn_count is not None else 0
        tool_summary = "\n".join(f"  {line}" for line in self._tool_lines[-5:]) or "  none"
        failure_summary = _format_last_failure(self._last_failure)
        return (
            "Profile\n"
            f"  name: {profile}\n"
            f"  provider: {provider}\n"
            f"  base_url: {base_url}\n"
            f"  model: {model}\n"
            f"  api_key_env: {api_key_env}\n"
            f"  key: {key_state}\n"
            f"  keyring: {keyring_status}\n\n"
            "Session\n"
            f"  id: {session}\n"
            f"  turns: {turn_count}\n"
            f"  state: {self._state}\n\n"
            "Tools This Turn\n"
            f"{tool_summary}\n\n"
            "Last Failure\n"
            f"{failure_summary}"
        )

    def _load_memory_candidates(self, *, silent: bool = False) -> None:
        try:
            self._memory_candidates = self.service.list_memory_candidates(status="pending")
            self._memory_error = None
            if self._memory_selected >= len(self._memory_candidates):
                self._memory_selected = max(0, len(self._memory_candidates) - 1)
        except Exception as error:
            self._memory_candidates = []
            self._memory_error = str(error)
            if not silent:
                self._append_block("Memory warning", f"Memory candidates unavailable: {error}")

    def _selected_memory_candidate(self) -> MemoryCandidate:
        return self._memory_candidates[self._memory_selected]

    def _handle_memory_key(self, key: str) -> bool:
        if key in {"up", "k"}:
            self.action_memory_up()
            return True
        if key in {"down", "j"}:
            self.action_memory_down()
            return True
        if key == "g":
            self.action_memory_first()
            return True
        if key in {"G", "shift+g", "upper_g"}:
            self.action_memory_last()
            return True
        return False

    def _move_memory_selection(self, delta: int) -> None:
        if not self._memory_mode or self._memory_detail_mode or not self._memory_candidates:
            return
        next_index = self._memory_selected + delta
        self._memory_selected = min(max(next_index, 0), len(self._memory_candidates) - 1)
        self._refresh()

    def _memory_side_bar(self) -> str:
        return self._memory_panel_text()

    def _memory_panel_text(self) -> str:
        prefix = []
        if self._memory_notice:
            prefix = ["Memory", f"  {self._memory_notice}", ""]
        if self._memory_error:
            return "\n".join(
                [*prefix, "Memory Candidates", f"  Memory candidates unavailable: {self._memory_error}"],
            )
        if not self._memory_candidates:
            return "\n".join([*prefix, "Memory Candidates", "  no pending candidates"])
        if self._memory_detail_mode:
            return "\n".join([*prefix, _memory_candidate_detail(self._selected_memory_candidate())])
        lines = [*prefix, "Memory Candidates"]
        for index, candidate in enumerate(self._memory_candidates):
            marker = ">" if index == self._memory_selected else " "
            lines.append(
                f"{marker} {candidate.candidate_id} [{candidate.scope}/{candidate.category}] {candidate.title}",
            )
        lines.extend(["", "↑/↓ j/k 移动  g/G 首尾  Enter 详情  a/y 确认  r 拒绝  Esc 返回"])
        return "\n".join(lines)

    def _footer_text(self) -> str:
        if self.size.width < self.MIN_WIDTH or self.size.height < self.MIN_HEIGHT:
            return "[Ctrl+Q]退出"
        if self._pending_interaction is not None:
            if self._pending_interaction.request.interaction_type == "user_input":
                return "[Enter]提交回答 [Esc]取消 [?]帮助 [Ctrl+Q]退出"
            return "[y]允许 [n]拒绝 [Esc]关闭 [?]帮助 [Ctrl+Q]退出"
        if self._memory_mode:
            if self._memory_detail_mode:
                return "[Esc]返回列表 [a/y]确认 [r]拒绝 [?]帮助 [Ctrl+Q]退出"
            return "[↑/↓ j/k]移动 [g/G]首尾 [Enter]详情 [a/y]确认 [r]拒绝 [Esc]返回聊天 [?]帮助 [Ctrl+Q]退出"
        return "[Enter]发送 [PgUp/PgDn]滚动 [m]记忆 [Tab]焦点 [?]帮助 [Ctrl+Q]退出"

    def _append_block(self, title: str, body: str) -> None:
        self._conversation_lines.append(f"{title}\n  {body}")

    def _append_line(self, line: str) -> None:
        self._conversation_lines.append(line)

    def _update_responsive_layout(self, width: int | None = None, height: int | None = None) -> None:
        main = self.query_one("#main", Horizontal)
        input_panel = self.query_one("#input-panel", Vertical)
        footer = self.query_one("#footer-bar", Static)
        resize_message = self.query_one("#resize-message", Static)
        side_bar = self.query_one("#side-bar", Static)
        terminal_width = width if width is not None else self.size.width
        terminal_height = height if height is not None else self.size.height
        too_small = terminal_width < self.MIN_WIDTH or terminal_height < self.MIN_HEIGHT
        resize_message.set_class(not too_small, "hidden")
        main.set_class(too_small, "hidden")
        input_panel.set_class(too_small, "hidden")
        footer.set_class(too_small, "hidden")
        side_bar.set_class(too_small or terminal_width < 120, "hidden")


def _approval_body(request: HumanInteractionRequest) -> str:
    args_summary = _format_args_summary(request.args_summary)
    impact_summary = _impact_summary(request.tool_name, request.args_summary)
    lines = [
        "工具请求需要确认",
        "",
        f"tool      {_safe_summary(request.tool_name, 80)}",
        f"question  {_safe_summary(request.question, 160)}",
    ]
    if request.reason:
        lines.append(f"reason    {_safe_summary(request.reason, 160)}")
    if request.risk_level:
        lines.append(f"risk      {_safe_summary(request.risk_level, 40)}")
    lines.extend(
        [
            f"args      {args_summary}",
            f"impact    {impact_summary}",
            "",
            "高风险内容首版只展示摘要，不展示完整 patch、stdout 或 stderr。",
        ],
    )
    return "\n".join(lines)


def _help_body(context: str) -> str:
    if context == "memory_list":
        return "\n".join(
            [
                "记忆候选列表",
                "",
                "↑/↓ 或 j/k  移动选中项",
                "g/G          跳到首项/末项",
                "Enter        查看当前候选详情",
                "a 或 y       确认当前候选",
                "r            拒绝当前候选",
                "Esc          返回聊天模式",
                "Ctrl+Q       退出 TUI",
            ],
        )
    if context == "memory_detail":
        return "\n".join(
            [
                "记忆候选详情",
                "",
                "Esc          返回列表，并保留当前选中项",
                "a 或 y       确认当前候选",
                "r            拒绝当前候选",
                "?            打开此帮助",
                "Ctrl+Q       退出 TUI",
            ],
        )
    if context == "pending_input":
        return "\n".join(
            [
                "等待补充输入",
                "",
                "Enter        提交回答并继续同一轮任务",
                "Esc          取消回答",
                "?            打开此帮助",
                "Ctrl+Q       退出 TUI",
            ],
        )
    if context == "approval":
        return "\n".join(
            [
                "审批确认",
                "",
                "y            允许当前工具调用",
                "n            拒绝当前工具调用",
                "Esc          拒绝并关闭审批",
                "?            打开此帮助",
                "Ctrl+Q       退出 TUI",
            ],
        )
    return "\n".join(
        [
            "聊天模式",
            "",
            "Enter        发送当前输入",
            "PgUp/PgDn    滚动对话",
            "m            打开记忆候选审查",
            "Tab          切换焦点",
            "?            打开此帮助",
            "Ctrl+Q       退出 TUI",
        ],
    )


def _memory_candidate_detail(candidate: MemoryCandidate) -> str:
    evidence = candidate.evidence
    lines = [
        "Memory Candidate Detail",
        f"candidate_id: {candidate.candidate_id}",
        f"status: {candidate.status}",
        f"scope: {candidate.scope}",
        f"category: {candidate.category}",
        f"title: {candidate.title}",
        f"body: {candidate.body}",
        f"source: {candidate.source}",
        f"created_at: {candidate.created_at}",
        f"tags: {', '.join(candidate.tags) if candidate.tags else 'none'}",
        f"risk_flags: {', '.join(candidate.risk_flags) if candidate.risk_flags else 'none'}",
        "",
        "Evidence",
        f"source_type: {evidence.source_type}",
        f"source_summary: {evidence.source_summary or 'none'}",
        f"basis: {evidence.basis or 'none'}",
        f"category_rationale: {evidence.category_rationale or 'none'}",
        f"episode_path: {evidence.episode_path or 'none'}",
    ]
    return "\n".join(lines)


def _format_args_summary(args_summary: dict[str, object]) -> str:
    if not args_summary:
        return "none"
    pieces = []
    for key, value in args_summary.items():
        if isinstance(value, list):
            safe_items = ", ".join(_safe_summary(str(item), 80) for item in value[:3])
            if len(value) > 3:
                safe_items += ", ..."
            pieces.append(f"{key}=[{safe_items}]")
        else:
            pieces.append(f"{key}={_safe_summary(str(value), 120)}")
    return "; ".join(pieces)


def _impact_summary(tool_name: str, args_summary: dict[str, object]) -> str:
    if tool_name in {"file_write", "apply_patch"}:
        path = _safe_summary(str(args_summary.get("path", "unknown")), 120)
        return f"会修改本地文件；path={path}"
    if tool_name == "apply_patch_set":
        paths = args_summary.get("paths")
        if isinstance(paths, list) and paths:
            return f"会修改本地文件；paths={_safe_summary(', '.join(str(path) for path in paths[:3]), 160)}"
        return "会修改本地文件；paths=unknown"
    if tool_name == "shell":
        command = _safe_summary(str(args_summary.get("command", "unknown")), 160)
        return f"会执行本地命令；是否修改本地文件取决于命令；command={command}"
    if tool_name == "code_run":
        return "会执行本地代码；可能读取或修改 workspace 内文件"
    return "影响范围以工具参数摘要为准"


def _safe_summary(value: str, limit: int) -> str:
    redacted, _ = redact_secret_like_text(value)
    normalized = " ".join(redacted.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def _payload_text(payload: dict[str, object], key: str, default: str) -> str:
    value: Any = payload.get(key)
    if value is None:
        return default
    return str(value)


def _key_state(status: AssistantWorkspaceStatus) -> str:
    if status.api_key_available and status.credential_source_used:
        return f"available via {status.credential_source_used}"
    return "missing"


def _compact_key_state(status: AssistantWorkspaceStatus) -> str:
    return "ok" if status.api_key_available else "missing"


def _keyring_status(status: AssistantWorkspaceStatus) -> str:
    if status.credential_store_available is False:
        reason = status.credential_store_error or "unknown"
        return f"keyring unavailable: {reason}"
    if status.credential_store_available is True:
        return "available"
    return "-"


def _failure_body(failed_stage: str, category: str, reason: str, episode_path: str) -> str:
    lines: list[str] = []
    if category == "Loop Limit Failure":
        lines.append("本轮没有完成：模型连续调用工具但没有给出最终回答。")
    lines.extend(
        [
            f"stage={failed_stage}",
            f"category={category}",
            f"reason={reason}",
            f"episode_path={episode_path}",
        ],
    )
    return "\n".join(lines)


def _format_last_failure(failure: dict[str, str] | None) -> str:
    if failure is None:
        return "  none"
    return (
        f"  category: {failure['failure_category']}\n"
        f"  stage: {failure['failed_stage']}\n"
        f"  reason: {failure['reason']}\n"
        f"  episode: {failure['episode_path']}"
    )


def _workspace_label(path: Path, limit: int) -> str:
    name = path.name or str(path)
    if len(name) <= limit:
        return name
    return _truncate_end(name, limit)


def _short_session(session_id: str, limit: int) -> str:
    if len(session_id) <= limit:
        return session_id
    return _truncate_end(session_id, limit)


def _truncate_end(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _truncate_status_line(value: str, width: int) -> str:
    if width <= 0 or len(value) <= width:
        return value
    return _truncate_end(value, width)


def run_tui(service: AssistantService) -> int:
    HaAgentTuiApp(service).run()
    return 0
