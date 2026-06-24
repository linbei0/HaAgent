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
from haagent.tui.app import HaAgentTuiApp


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
        failure_event: ChatEvent | None = None,
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
        self.failure_event = failure_event
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []

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

    def run_prompt_events(self, prompt: str, *, event_sink=None):
        self.prompts.append(prompt)
        self.started.set()
        if self.block_until_released:
            self.release.wait(timeout=2)
        if event_sink is not None:
            if self.failure_event is not None:
                event_sink(self.failure_event)
                return SimpleNamespace(status="failed")
            event_sink(
                ChatEvent(
                    event_type="assistant_message",
                    session_id="session-test",
                    turn_index=len(self.prompts),
                    message="assistant message",
                    payload={"content": f"assistant: {prompt}"},
                ),
            )
        return SimpleNamespace(status="completed")


def _text(app: HaAgentTuiApp, selector: str) -> str:
    return str(app.query_one(selector).content)


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
            assert "Ctrl+Q 退出" in str(app.query_one("#conversation").render())
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
            assert str(episode_path) in conversation
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
