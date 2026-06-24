"""
haagent/tui/app.py - HaAgent Textual 首版界面

提供显式 `haagent tui` 的最小垂直切片，只通过 AssistantService 驱动会话。
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Static

from haagent.app.assistant_service import AssistantService, AssistantWorkspaceStatus
from haagent.runtime.chat_session import ChatEvent


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
    ]

    def __init__(self, service: AssistantService) -> None:
        super().__init__()
        self.service = service
        self._state = "idle"
        self._conversation_lines: list[str] = []
        self._tool_lines: list[str] = []
        self._last_failure: dict[str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="status-bar")
        with Horizontal(id="main"):
            yield Static("", id="conversation")
            yield Static("", id="side-bar")
        with Vertical(id="input-panel"):
            yield Input(placeholder="输入 prompt，Enter 发送", id="prompt-input")
        yield Static(Text("[Enter]发送 [Tab]焦点 [?]帮助 [Ctrl+Q]退出"), id="footer-bar")

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
        self._append_block("You", prompt)
        self._state = "running"
        self._refresh()
        self._run_prompt(prompt)

    def action_help(self) -> None:
        self._append_block("Help", "Enter 发送，Tab 切换焦点，Ctrl+Q 退出。")
        self._refresh_conversation()

    def action_quit(self) -> None:
        self.exit(None)

    @work(thread=True, exclusive=True)
    def _run_prompt(self, prompt: str) -> None:
        try:
            result = self.service.run_prompt_events(
                prompt,
                event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
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
            self._append_block("Waiting", "工具请求需要审批；首版 TUI 暂不实现 allow/deny 流程。")
        elif event_type == "user_input_requested":
            self._state = "waiting input"
            question = _payload_text(payload, "question", event.message)
            self._append_block("Waiting", f"需要补充信息：{question}")
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
        content = "\n".join(self._conversation_lines) if self._conversation_lines else "Ready. 输入 prompt 后按 Enter 发送；Ctrl+Q 退出。"
        self.query_one("#conversation", Static).update(Text(content))

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
            "Tools\n"
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
