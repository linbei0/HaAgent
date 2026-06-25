"""
haagent/tui/app.py - HaAgent Textual 首版界面

提供显式 `haagent tui` 的最小垂直切片，只通过 AssistantService 驱动会话。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RichLog, Static

from haagent.app.assistant_service import AssistantService, AssistantWorkspaceStatus
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

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class HaAgentTuiApp(App[None]):
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

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "退出"),
        ("q", "quit", "退出"),
        ("?", "help", "帮助"),
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

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            yield RichLog(id="conversation", wrap=True, auto_scroll=True)
            yield Static("", id="side-bar")
        with Vertical(id="input-panel"):
            yield Input(placeholder=self._default_prompt_placeholder, id="prompt-input")
        yield Static(Text("[Enter]发送 [PgUp/PgDn]滚动 [Tab]焦点 [?]帮助 [Ctrl+Q]退出"), id="footer-bar")

    def on_mount(self) -> None:
        self._show_initial_configuration_state()
        self._refresh()
        self._update_responsive_layout()
        self.query_one("#prompt-input", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(width=event.size.width)

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
        self._append_block("Help", "Enter 发送，Tab 切换焦点，Ctrl+Q 退出。")
        self._refresh_conversation()

    def action_quit(self) -> None:
        self.exit(None)

    def action_conversation_page_up(self) -> None:
        self.query_one("#conversation", RichLog).scroll_page_up(animate=False, force=True)

    def action_conversation_page_down(self) -> None:
        self.query_one("#conversation", RichLog).scroll_page_down(animate=False, force=True)

    def action_cancel_interaction(self) -> None:
        if self._pending_interaction is None:
            return
        self._complete_interaction(HumanInteractionResponse(approved=False, answer=""))

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

    def _refresh_conversation(self) -> None:
        conversation = self.query_one("#conversation", RichLog)
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
        profile = status.profile_name or "missing"
        provider = status.provider or "-"
        model = status.model or "-"
        api_key_env = status.api_key_env or "-"
        key_state = _key_state(status)
        session = status.current_session_id or "-"
        turn_count = status.current_turn_count if status.current_turn_count is not None else 0
        return (
            f"workspace: {status.workspace_root}  profile: {profile}  "
            f"{provider}/{model}  key: {key_state}({api_key_env})  session: {session}  "
            f"turn: {turn_count}  state: {self._state}"
        )

    def _side_bar(self, status: AssistantWorkspaceStatus) -> str:
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

    def _append_block(self, title: str, body: str) -> None:
        self._conversation_lines.append(f"{title}\n  {body}")

    def _append_line(self, line: str) -> None:
        self._conversation_lines.append(line)

    def _update_responsive_layout(self, width: int | None = None) -> None:
        side_bar = self.query_one("#side-bar", Static)
        terminal_width = width if width is not None else self.size.width
        side_bar.set_class(terminal_width < 120, "hidden")


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


def run_tui(service: AssistantService) -> int:
    HaAgentTuiApp(service).run()
    return 0
