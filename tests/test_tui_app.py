"""
tests/test_tui_app.py - HaAgent TUI 垂直切片测试

验证 TUI adapter 通过 AssistantService 风格接口展示状态、运行 prompt 和接收事件。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from haagent import cli
from haagent.app.assistant_service import AssistantWorkspaceStatus
from haagent.runtime.chat_session import ChatEvent
from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.app import HaAgentTuiApp
from textual.widgets import RichLog


class FakeAssistantService:
    def __init__(
        self,
        *,
        workspace_root: Path,
        profile_name: str | None = "local",
        provider: str | None = "openai-chat",
        model: str | None = "deepseek-chat",
        api_key_env: str | None = "DEEPSEEK_API_KEY",
        api_key_available: bool = True,
        credential_source_configured: str | None = "keyring",
        credential_source_used: str | None = "keyring",
        credential_store_available: bool | None = True,
        credential_store_error: str | None = None,
        profile_error: str | None = None,
        block_until_released: bool = False,
        assistant_content: str | None = None,
        failure_event: ChatEvent | None = None,
        interaction_request: HumanInteractionRequest | None = None,
        extra_events: list[ChatEvent] | None = None,
    ) -> None:
        self.workspace_root = workspace_root
        self.profile_name = profile_name
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.api_key_available = api_key_available
        self.credential_source_configured = credential_source_configured
        self.credential_source_used = credential_source_used
        self.credential_store_available = credential_store_available
        self.credential_store_error = credential_store_error
        self.profile_error = profile_error
        self.block_until_released = block_until_released
        self.assistant_content = assistant_content
        self.failure_event = failure_event
        self.interaction_request = interaction_request
        self.extra_events = list(extra_events or [])
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []
        self.interaction_responses: list[HumanInteractionResponse] = []

    def get_workspace_status(self) -> AssistantWorkspaceStatus:
        return AssistantWorkspaceStatus(
            workspace_root=self.workspace_root,
            runs_root=self.workspace_root / ".runs",
            profile_name=self.profile_name,
            provider=self.provider,
            base_url="https://api.deepseek.com",
            model=self.model,
            api_key_env=self.api_key_env,
            api_key_available=self.api_key_available,
            credential_source_configured=self.credential_source_configured,
            credential_source_used=self.credential_source_used,
            credential_store_available=self.credential_store_available,
            credential_store_error=self.credential_store_error,
            profile_error=self.profile_error,
            current_session_id="session-test",
            current_turn_count=len(self.prompts),
        )

    def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
        self.prompts.append(prompt)
        self.started.set()
        if self.block_until_released:
            self.release.wait(timeout=2)
        if event_sink is not None:
            if self.failure_event is not None:
                event_sink(self.failure_event)
                return SimpleNamespace(status="failed")
            for extra_event in self.extra_events:
                event_sink(extra_event)
            if self.interaction_request is not None:
                request = self.interaction_request
                event_sink(_interaction_requested_event(request, len(self.prompts)))
                response = (
                    interaction_handler(request)
                    if interaction_handler is not None
                    else HumanInteractionResponse(approved=False)
                )
                self.interaction_responses.append(response)
                event_sink(_interaction_response_event(request, response, len(self.prompts)))
                if not response.approved:
                    event_sink(
                        ChatEvent(
                            event_type="tool_failed",
                            session_id="session-test",
                            turn_index=len(self.prompts),
                            message="tool failed",
                            payload={
                                "tool_name": request.tool_name,
                                "error_type": (
                                    "approval_denied"
                                    if request.interaction_type == "approval"
                                    else "user_input_unavailable"
                                ),
                                "message": "interaction declined",
                            },
                        ),
                    )
                    return SimpleNamespace(status="failed")
            event_sink(
                ChatEvent(
                    event_type="assistant_message",
                    session_id="session-test",
                    turn_index=len(self.prompts),
                    message="assistant message",
                    payload={"content": self.assistant_content or f"assistant: {prompt}"},
                ),
            )
        return SimpleNamespace(status="completed")


def _text(app: HaAgentTuiApp, selector: str) -> str:
    widget = app.query_one(selector)
    if isinstance(widget, RichLog):
        return "\n".join("".join(segment.text for segment in line) for line in widget.lines)
    return str(widget.content)


def _all_text(app: HaAgentTuiApp) -> str:
    widgets = list(app.query("*"))
    if app.screen is not None:
        widgets.extend(app.screen.query("*"))
    return "\n".join(str(widget.render()) for widget in widgets)


def _interaction_requested_event(request: HumanInteractionRequest, turn_index: int) -> ChatEvent:
    event_type = "approval_requested" if request.interaction_type == "approval" else "user_input_requested"
    return ChatEvent(
        event_type=event_type,
        session_id="session-test",
        turn_index=turn_index,
        message="interaction requested",
        payload={
            "tool_name": request.tool_name,
            "question": request.question,
            "reason": request.reason,
            "risk_level": request.risk_level,
            "args_summary": request.args_summary,
            "approved": None,
        },
    )


def _interaction_response_event(
    request: HumanInteractionRequest,
    response: HumanInteractionResponse,
    turn_index: int,
) -> ChatEvent:
    if request.interaction_type == "approval":
        event_type = "approval_granted" if response.approved else "approval_denied"
        payload = {
            "tool_name": request.tool_name,
            "question": request.question,
            "approved": response.approved,
            "args_summary": request.args_summary,
        }
    else:
        event_type = "user_input_received"
        payload = {
            "tool_name": request.tool_name,
            "question": request.question,
            "answer_chars": len(response.answer),
            "approved": response.approved,
        }
    return ChatEvent(
        event_type=event_type,
        session_id="session-test",
        turn_index=turn_index,
        message="interaction response",
        payload=payload,
    )


def _approval_request(args_summary: dict[str, object] | None = None) -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="Approve high risk tool shell?",
        reason="shell can modify local files",
        risk_level="high",
        args_summary=args_summary or {"command": "uv run pytest -q", "cwd": ".", "timeout_seconds": 30},
    )


def _user_input_request() -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        question="Which file should I inspect?",
        reason="Need target file",
        risk_level="low",
        args_summary={"question": "Which file should I inspect?", "reason": "Need target file"},
    )


def test_tui_parser_accepts_explicit_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["tui", "--workspace-root", "workspace", "--runs-root", "runs"])

    assert args.command == "tui"
    assert args.workspace_root == Path("workspace")
    assert args.runs_root == Path("runs")


def test_tui_app_starts_and_shows_status(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            side = _text(app, "#side-bar")
            assert str(tmp_path) in status
            assert "profile: local" in status
            assert "openai-chat/deepseek-chat" in status
            assert "key: available via keyring" in status
            assert "DEEPSEEK_API_KEY" in status
            assert "session-test" in status
            assert "Profile" in side
            assert "base_url: https://api.deepseek.com" in side
            assert "Ctrl+Q 退出" in _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "[Ctrl+Q]退出" in footer
            assert "[q]退出" not in footer
            assert "[Enter]发送" in str(app.query_one("#footer-bar").render())
            assert "[Tab]焦点" in str(app.query_one("#footer-bar").render())

    asyncio.run(run())


def test_tui_profile_missing_shows_setup_message(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            profile_name=None,
            provider=None,
            model=None,
            api_key_env=None,
            api_key_available=False,
            profile_error="未找到默认模型配置，请先运行 haagent setup",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = _text(app, "#conversation")
            assert "未找到默认模型配置" in conversation
            assert "uv run haagent setup" in conversation

    asyncio.run(run())


def test_tui_api_key_missing_shows_env_name(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            api_key_available=False,
            credential_source_used=None,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "key: missing" in status
            assert "DEEPSEEK_API_KEY" in status
            assert "DEEPSEEK_API_KEY" in conversation

    asyncio.run(run())


def test_tui_api_key_available_via_env(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            credential_source_used="env",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            side = _text(app, "#side-bar")
            assert "key: available via env" in status
            assert "key: available via env" in side

    asyncio.run(run())


def test_tui_keyring_unavailable_shows_reason(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            api_key_available=False,
            credential_source_used=None,
            credential_store_available=False,
            credential_store_error="backend unavailable",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = _text(app, "#conversation")
            side = _text(app, "#side-bar")
            assert "系统凭据库不可用：backend unavailable" in conversation
            assert "uv run haagent setup" in conversation
            assert "keyring unavailable: backend unavailable" in side

    asyncio.run(run())


def test_tui_ctrl_q_exits_even_when_prompt_input_is_focused(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.query_one("#prompt-input").has_focus
            await pilot.press("ctrl+q")
            await pilot.pause(0.1)
            assert not app.is_running

    asyncio.run(run())


def test_tui_submit_prompt_calls_service_and_renders_assistant_event(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Summarize this folder"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts == ["Summarize this folder"]
            conversation = _text(app, "#conversation")
            assert "You" in conversation
            assert "Summarize this folder" in conversation
            assert "assistant: Summarize this folder" in conversation

    asyncio.run(run())


def test_tui_approval_requested_opens_modal_with_deny_focused(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_approval_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Run checks"
            await pilot.press("enter")
            await pilot.pause(0.2)
            modal_text = _all_text(app)
            assert "Tool Approval" in modal_text
            assert "shell" in modal_text
            assert "Approve high risk tool shell?" in modal_text
            assert "uv run pytest -q" in modal_text
            assert "会执行本地命令" in modal_text
            assert app.screen.query_one("#approval-deny").has_focus
            assert "state: waiting approval" in _text(app, "#status-bar")
            assert "shell pending approval" in _text(app, "#side-bar")
            assert "Tool shell pending approval" in _text(app, "#conversation")
            await pilot.press("n")
            await pilot.pause(0.1)

    asyncio.run(run())


def test_tui_approval_allow_returns_approved_true_to_same_prompt(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_approval_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Run checks"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("y")
            await pilot.pause(0.2)
            assert service.prompts == ["Run checks"]
            assert service.interaction_responses == [HumanInteractionResponse(approved=True, answer="")]
            assert "Approval granted: shell" in _text(app, "#conversation")
            assert "shell approved" in _text(app, "#side-bar")
            assert "assistant: Run checks" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_approval_deny_returns_approved_false_to_same_prompt(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_approval_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Run checks"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("n")
            await pilot.pause(0.2)
            assert service.prompts == ["Run checks"]
            assert service.interaction_responses == [HumanInteractionResponse(approved=False, answer="")]
            assert "Approval denied: shell" in _text(app, "#conversation")
            assert "shell denied" in _text(app, "#side-bar")
            assert "state: failed" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_user_input_requested_enters_answer_required_state(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_user_input_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert "state: waiting input" in _text(app, "#status-bar")
            assert "Answer required" in _text(app, "#conversation")
            assert "Which file should I inspect?" in _text(app, "#conversation")
            assert "回答 Agent 的问题" in input_widget.placeholder
            assert input_widget.has_focus
            await pilot.press("escape")
            await pilot.pause(0.1)

    asyncio.run(run())


def test_tui_user_input_answer_continues_same_run_prompt_events(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_user_input_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            input_widget.value = "README.md"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts == ["Inspect"]
            assert service.interaction_responses == [
                HumanInteractionResponse(approved=True, answer="README.md"),
            ]
            conversation = _text(app, "#conversation")
            assert "Answer submitted: request_user_input" in conversation
            assert "README.md" not in conversation
            assert "assistant: Inspect" in conversation

    asyncio.run(run())


def test_tui_user_input_cancel_returns_explicit_denial(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_user_input_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert service.interaction_responses == [HumanInteractionResponse(approved=False, answer="")]
            conversation = _text(app, "#conversation")
            assert "Answer declined: request_user_input" in conversation
            assert "Tool request_user_input failed" in conversation
            assert "state: failed" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_interaction_reused_event_does_not_enter_pending_interaction(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                ChatEvent(
                    event_type="interaction_reused",
                    session_id="session-test",
                    turn_index=1,
                    message="interaction reused",
                    payload={
                        "interaction_type": "user_input",
                        "tool_name": "request_user_input",
                        "status": "answered",
                    },
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.interaction_responses == []
            assert "Answer required" not in _text(app, "#conversation")
            assert "pending approval" not in _text(app, "#conversation")
            assert "state: idle" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_approval_summary_redacts_secret_like_text(tmp_path: Path) -> None:
    async def run() -> None:
        secret = "sk-test1234567890abcdef1234567890abcdef"
        request = _approval_request({"command": f"echo {secret}", "cwd": ".", "timeout_seconds": 30})
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=request)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Run secret command"
            await pilot.press("enter")
            await pilot.pause(0.2)
            rendered = _all_text(app)
            await pilot.press("n")
            await pilot.pause(0.1)
            assert secret not in rendered
            assert "[REDACTED_TOKEN]" in rendered

    asyncio.run(run())


def test_tui_conversation_auto_scrolls_to_latest_content(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 12)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            assert conversation.max_scroll_y > 0
            assert conversation.scroll_y == conversation.max_scroll_y
            assert "line 29" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_conversation_wraps_long_messages_for_scroll_height(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 20)) as pilot:
            conversation = app.query_one("#conversation")
            long_reply = (
                "# 我能做什么？ 我是 **HaAgent**，一个运行在当前工作目录下的本地个人 AI 助手。"
                "我可以读取文件、整理内容、编辑文档、分析项目、运行命令，并支持多轮对话。"
            ) * 8
            app._append_block("Assistant", long_reply)
            app._refresh_conversation()
            await pilot.pause()
            longest_line = max(len(line) for line in _text(app, "#conversation").splitlines())
            assert longest_line <= conversation.content_size.width
            assert conversation.virtual_size.height > len(app._conversation_lines)
            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())


def test_tui_renders_full_long_assistant_message_from_event_sink(tmp_path: Path) -> None:
    async def run() -> None:
        long_reply = ("HaAgent 可以读取文件、整理内容、编辑文档、分析项目。" * 40) + "完整结尾"
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content=long_reply)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 20)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍能力"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "[truncated]" not in conversation
            assert "完整结尾" in conversation
            assert app.query_one("#conversation").scroll_y == app.query_one("#conversation").max_scroll_y

    asyncio.run(run())


def test_tui_failure_event_shows_reason_episode_and_sidebar_summary(tmp_path: Path) -> None:
    async def run() -> None:
        episode_path = tmp_path / ".runs" / "episode-failed"
        service = FakeAssistantService(
            workspace_root=tmp_path,
            failure_event=ChatEvent(
                event_type="failure",
                session_id="session-test",
                turn_index=1,
                message="chat turn failed",
                payload={
                    "status": "failed",
                    "failed_stage": "executing",
                    "failure_category": "Loop Limit Failure",
                    "reason": "exceeded max_turns=20",
                    "episode_path": str(episode_path),
                },
            ),
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍一下项目"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            side = _text(app, "#side-bar")
            assert "本轮没有完成：模型连续调用工具但没有给出最终回答。" in conversation
            assert "stage=executing" in conversation
            assert "category=Loop Limit Failure" in conversation
            assert "reason=exceeded max_turns=20" in conversation
            assert str(episode_path) in conversation.replace("\n", "")
            assert "Last Failure" in side
            assert "Loop Limit Failure" in side
            assert str(episode_path) in side

    asyncio.run(run())


def test_tui_running_state_does_not_block_ui(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            assert "state: running" in _text(app, "#status-bar")
            assert app.query_one("#prompt-input") is input_widget
            service.release.set()
            await pilot.pause(0.2)
            assert "state: idle" in _text(app, "#status-bar")

    asyncio.run(run())
