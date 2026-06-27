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
from haagent.memory import CandidateEvidence, MemoryCandidate, MemoryRecord
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
        memory_candidates: list[MemoryCandidate] | None = None,
        memory_error: Exception | None = None,
        current_session_id: str = "session-test",
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
        self.memory_candidates = list(memory_candidates or [])
        self.memory_error = memory_error
        self.current_session_id = current_session_id
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []
        self.interaction_responses: list[HumanInteractionResponse] = []
        self.confirmed_candidate_ids: list[str] = []
        self.rejected_candidate_ids: list[tuple[str, str]] = []

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
            current_session_id=self.current_session_id,
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

    def list_memory_candidates(self, status: str | None = "pending") -> list[MemoryCandidate]:
        if self.memory_error is not None:
            raise self.memory_error
        if status is None:
            return list(self.memory_candidates)
        return [candidate for candidate in self.memory_candidates if candidate.status == status]

    def get_memory_candidate(self, candidate_id: str) -> MemoryCandidate:
        if self.memory_error is not None:
            raise self.memory_error
        for candidate in self.memory_candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise RuntimeError(f"memory candidate not found: {candidate_id}")

    def confirm_memory_candidate(self, candidate_id: str) -> MemoryRecord:
        candidate = self.get_memory_candidate(candidate_id)
        self.confirmed_candidate_ids.append(candidate_id)
        self.memory_candidates = [item for item in self.memory_candidates if item.candidate_id != candidate_id]
        return MemoryRecord(
            memory_id="mem_" + candidate_id,
            scope=candidate.scope,
            category=candidate.category,
            title=candidate.title,
            body=candidate.body,
            evidence=candidate.evidence,
            source_candidate_id=candidate.candidate_id,
            content_hash="hash",
            created_at="2026-06-26T00:00:00+00:00",
            updated_at="2026-06-26T00:00:00+00:00",
            tags=list(candidate.tags),
        )

    def reject_memory_candidate(self, candidate_id: str, reason: str) -> MemoryCandidate:
        candidate = self.get_memory_candidate(candidate_id)
        self.rejected_candidate_ids.append((candidate_id, reason))
        self.memory_candidates = [item for item in self.memory_candidates if item.candidate_id != candidate_id]
        raw = candidate.to_dict()
        raw["status"] = "rejected"
        raw["updated_at"] = "2026-06-26T00:00:00+00:00"
        return MemoryCandidate.from_dict(raw)


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


def _memory_candidate(candidate_id: str = "cand_abc123", title: str = "用户身份与爱好") -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        scope="user",
        category="user_preferences",
        title=title,
        body="用户叫小明，喜欢唱跳rap篮球。",
        evidence=CandidateEvidence(
            source_type="extraction",
            evidence_summary="用户明确要求记住自己的名字和爱好。",
            session_id="session-test",
            turn_index=1,
            episode_path=".runs/episode-test",
            source_summary="用户明确要求记住自己的名字和爱好。",
            basis="用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。",
            category_rationale="这是跨 workspace 可复用的用户偏好和身份信息。",
        ),
        source="extraction",
        status="pending",
        created_at="2026-06-26T00:00:00+00:00",
        updated_at="2026-06-26T00:00:00+00:00",
        tags=["profile"],
        risk_flags=[],
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
            assert "ws:" in status
            assert str(tmp_path) not in status
            assert "profile: local" in status
            assert "openai-chat/deepseek-chat" in status
            assert "key: ok" in status
            assert "DEEPSEEK_API_KEY" not in status
            assert "session-test" in status
            assert "Profile" in side
            assert "base_url: https://api.deepseek.com" in side
            assert "api_key_env: DEEPSEEK_API_KEY" in side
            assert "Ctrl+Q 退出" in _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "[Ctrl+Q]退出" in footer
            assert "[q]退出" not in footer
            assert "[Enter]发送" in str(app.query_one("#footer-bar").render())
            assert "[Tab]焦点" in str(app.query_one("#footer-bar").render())

    asyncio.run(run())


def test_tui_status_bar_is_compact_at_80_and_120_columns(tmp_path: Path) -> None:
    long_workspace = tmp_path / "very" / "long" / "workspace-name-that-should-not-fill-the-status-bar"
    long_model = "provider-model-name-with-many-segments-and-context-window-very-long"
    long_session = "session-20260627-abcdef1234567890abcdef1234567890"

    async def run_80() -> None:
        service = FakeAssistantService(
            workspace_root=long_workspace,
            model=long_model,
            current_session_id=long_session,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            status = _text(app, "#status-bar")
            side = app.query_one("#side-bar")
            assert len(status) <= 80
            assert "ws:" in status
            assert "profile: local" in status
            assert "openai-chat/" in status
            assert "key: ok" in status
            assert "sid:" in status
            assert "turn:" in status
            assert "state: idle" in status
            assert str(long_workspace) not in status
            assert long_model not in status
            assert long_session not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert side.has_class("hidden")

    async def run_120() -> None:
        service = FakeAssistantService(
            workspace_root=long_workspace,
            model=long_model,
            current_session_id=long_session,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            side = app.query_one("#side-bar")
            assert len(status) <= 120
            assert "ws:" in status
            assert "profile: local" in status
            assert "key: ok" in status
            assert "sid:" in status
            assert str(long_workspace) not in status
            assert long_model not in status
            assert long_session not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert not side.has_class("hidden")
            assert "base_url: https://api.deepseek.com" in _text(app, "#side-bar")
            assert "api_key_env: DEEPSEEK_API_KEY" in _text(app, "#side-bar")

    asyncio.run(run_80())
    asyncio.run(run_120())


def test_tui_responsive_minimum_size_and_layout_breakpoints(tmp_path: Path) -> None:
    async def run_too_small() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(70, 20)):
            assert "终端尺寸过小" in _all_text(app)
            assert "请调整到至少 80x24" in _all_text(app)
            assert app.query_one("#main").has_class("hidden")
            assert app.query_one("#input-panel").has_class("hidden")

    async def run_80() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            assert app.query_one("#resize-message").has_class("hidden")
            assert not app.query_one("#main").has_class("hidden")
            assert app.query_one("#side-bar").has_class("hidden")
            assert "[Ctrl+Q]退出" in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert app.query_one("#resize-message").has_class("hidden")
            assert not app.query_one("#main").has_class("hidden")
            assert not app.query_one("#side-bar").has_class("hidden")
            assert "Profile" in _text(app, "#side-bar")

    asyncio.run(run_too_small())
    asyncio.run(run_80())
    asyncio.run(run_120())


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
            assert "DEEPSEEK_API_KEY" not in status
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
            assert "key: ok" in status
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


def test_tui_q_does_not_exit_while_prompt_input_is_focused(tmp_path: Path) -> None:
    async def run_empty_input() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            assert input_widget.has_focus
            await pilot.press("q")
            await pilot.pause(0.1)
            assert app.is_running
            assert input_widget.value == "q"

    async def run_existing_input() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "abc"
            await pilot.press("q")
            await pilot.pause(0.1)
            assert app.is_running
            assert input_widget.value == "abcq"

    asyncio.run(run_empty_input())
    asyncio.run(run_existing_input())


def test_tui_help_uses_modal_without_polluting_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            before = _text(app, "#conversation")
            await pilot.press("?")
            await pilot.pause(0.1)
            after = _text(app, "#conversation")
            rendered = _all_text(app)
            assert after == before
            assert "HaAgent Help" in rendered
            assert "聊天模式" in rendered
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "HaAgent Help" not in _all_text(app)

    asyncio.run(run())


def test_tui_help_modal_is_contextual_for_memory_modes(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert "记忆候选列表" in _all_text(app)
            assert "j/k" in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert "记忆候选详情" in _all_text(app)
            assert "返回列表" in _all_text(app)

    asyncio.run(run())


def test_tui_help_modal_is_contextual_for_pending_input(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_user_input_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            before = _text(app, "#conversation")
            await pilot.press("?")
            await pilot.pause(0.1)
            after_help = _text(app, "#conversation")
            rendered = _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            if app._pending_interaction is not None:
                app._complete_interaction(HumanInteractionResponse(approved=False, answer=""))
                await pilot.pause(0.2)
            assert after_help == before
            assert "等待补充输入" in rendered
            assert "Enter" in rendered

    asyncio.run(run())


def test_tui_help_modal_is_contextual_for_approval_modal(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_approval_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Run checks"
            await pilot.press("enter")
            await pilot.pause(0.2)
            before = _text(app, "#conversation")
            await pilot.press("?")
            await pilot.pause(0.1)
            after_help = _text(app, "#conversation")
            rendered = _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            if "Tool Approval" in _all_text(app):
                await pilot.press("n")
                await pilot.pause(0.2)
            assert after_help == before
            assert "审批确认" in rendered
            assert "y" in rendered
            assert "n" in rendered

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
            deny_has_focus = app.screen.query_one("#approval-deny").has_focus
            status = _text(app, "#status-bar")
            side = _text(app, "#side-bar")
            conversation = _text(app, "#conversation")
            await pilot.press("n")
            await pilot.pause(0.1)
            assert "Tool Approval" in modal_text
            assert "shell" in modal_text
            assert "Approve high risk tool shell?" in modal_text
            assert "uv run pytest -q" in modal_text
            assert "会执行本地命令" in modal_text
            assert deny_has_focus
            assert "state: waiting approval" in status
            assert "shell pending approval" in side
            assert "Tool shell pending approval" in conversation

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
            status = _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            placeholder = input_widget.placeholder
            input_has_focus = input_widget.has_focus
            await pilot.press("escape")
            await pilot.pause(0.1)
            if app._pending_interaction is not None:
                app._complete_interaction(HumanInteractionResponse(approved=False, answer=""))
                await pilot.pause(0.1)
            assert "state: waiting input" in status
            assert "Answer required" in conversation
            assert "Which file should I inspect?" in conversation
            assert "回答 Agent 的问题" in placeholder
            assert input_has_focus

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


def test_tui_memory_candidate_event_shows_notice(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[_memory_candidate()],
            extra_events=[
                ChatEvent(
                    event_type="memory_candidates_created",
                    session_id="session-test",
                    turn_index=1,
                    message="memory candidates created",
                    payload={
                        "count": 1,
                        "message": "发现 1 条可记忆候选，已放入候选队列，等待你确认。",
                    },
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "记住我的爱好"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "发现 1 条可记忆候选，已放入候选队列，等待你确认。" in conversation
            assert "Memory Candidates" in _text(app, "#side-bar")
            assert "cand_abc123" in _text(app, "#side-bar")
            assert "[a/y]确认" in _text(app, "#footer-bar")

    asyncio.run(run())


def test_tui_memory_panel_lists_and_shows_candidate_details(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            side = _text(app, "#side-bar")
            footer = _text(app, "#footer-bar")
            assert "Memory Candidates" in side
            assert "cand_abc123" in side
            assert "用户身份与爱好" in side
            assert "[Enter]详情" in footer

            await pilot.press("enter")
            await pilot.pause(0.1)
            side = _text(app, "#side-bar")
            assert "source_summary: 用户明确要求记住自己的名字和爱好。" in side
            assert "basis: 用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in side
            assert "category_rationale: 这是跨 workspace 可复用的用户偏好和身份信息。" in side

    asyncio.run(run())


def test_tui_memory_navigation_selects_second_candidate_for_confirm_and_reject(tmp_path: Path) -> None:
    async def confirm_run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            side = _text(app, "#side-bar")
            assert "> cand_second" in side
            assert "  cand_first" in side
            await pilot.press("a")
            await pilot.pause(0.1)
            assert service.confirmed_candidate_ids == ["cand_second"]

    async def reject_run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("j")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")
            await pilot.press("r")
            await pilot.pause(0.1)
            assert service.rejected_candidate_ids == [("cand_second", "rejected from TUI")]

    asyncio.run(confirm_run())
    asyncio.run(reject_run())


def test_tui_memory_navigation_supports_home_end_and_keeps_selection_after_detail(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_middle", "中间偏好"),
                _memory_candidate("cand_last", "最后偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("G")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#side-bar")

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "candidate_id: cand_last" in _text(app, "#side-bar")
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#side-bar")

            await pilot.press("g")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#side-bar")
            footer = _text(app, "#footer-bar")
            assert "[↑/↓ j/k]移动" in footer
            assert "[g/G]首尾" in footer

    asyncio.run(run())


def test_tui_memory_navigation_moves_one_candidate_per_keypress(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
                _memory_candidate("cand_third", "第三条偏好"),
                _memory_candidate("cand_fourth", "第四条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")

            await pilot.press("j")
            await pilot.pause(0.1)
            assert "> cand_third" in _text(app, "#side-bar")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")

            await pilot.press("k")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#side-bar")

    asyncio.run(run())


def test_tui_memory_mode_is_readable_without_sidebar(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(90, 30)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "Memory Candidates" in conversation
            assert "cand_abc123" in conversation

            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "Memory Candidate Detail" in conversation
            assert "basis: 用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in conversation

    asyncio.run(run())


def test_tui_memory_confirm_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("a")
            await pilot.pause(0.1)
            assert service.confirmed_candidate_ids == ["cand_abc123"]
            assert "Memory confirmed: cand_abc123" in _text(app, "#conversation")
            assert "no pending candidates" in _text(app, "#side-bar")

    asyncio.run(run())


def test_tui_memory_reject_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            await pilot.press("r")
            await pilot.pause(0.1)
            assert service.rejected_candidate_ids == [("cand_abc123", "rejected from TUI")]
            assert "Memory rejected: cand_abc123" in _text(app, "#conversation")
            assert "no pending candidates" in _text(app, "#side-bar")

    asyncio.run(run())


def test_tui_memory_panel_shows_empty_and_load_errors(tmp_path: Path) -> None:
    async def empty_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            assert "no pending candidates" in _text(app, "#side-bar")

    async def error_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_error=RuntimeError("queue broken"))
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)
            assert "Memory candidates unavailable: queue broken" in _text(app, "#conversation")

    asyncio.run(empty_run())
    asyncio.run(error_run())


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
        async with app.run_test(size=(80, 24)) as pilot:
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
        async with app.run_test(size=(120, 40)) as pilot:
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
        async with app.run_test(size=(120, 40)) as pilot:
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
