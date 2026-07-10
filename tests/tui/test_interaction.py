"""
tests/tui/test_interaction.py - HaAgent TUI interaction 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from haagent import cli
from haagent.app.assistant_service import (
    AssistantSandboxStatus,
    AssistantSessionStatus,
    AssistantSessionSummary,
    AssistantWorkspaceStatus,
    SandboxDoctorReport,
)
from haagent.memory import CandidateEvidence, MemoryCandidate, MemoryRecord
from haagent.runtime.events import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RuntimeUiEvent,
    RuntimeUiEventMapper,
    TaskProgressEvent,
    ToolActivityEvent,
    UserInputStateEvent,
)
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.flows.path_authorization import find_untrusted_absolute_paths
from haagent.tui.commands import SlashCommandResult, command_registry, parse_slash_command
from haagent.tui.design.failures import failure_from_payload, failure_next_steps
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, fuzzy_file_matches, path_reference_token
from haagent.tui.design.keys import APP_BINDINGS, footer_text, help_body, key_help_lines
from haagent.tui.overlays.models import ModelCatalogLoadingOverlay
from haagent.tui.design.copy import MODAL_TITLES, PANEL_TITLES
from haagent.tui.design.renderers import memory_panel_text, status_line
from haagent.tui.state.search import ConversationSearchState
from haagent.tui.overlays.sessions import SessionOverlayState
from haagent.tui.state import ResponsiveLayout, layout_for_size
from haagent.tui.presentation.progress import ProgressStatusState
from haagent.tui.design.theme import (
    SemanticToken,
    TuiThemeMode,
    no_color_enabled,
    select_theme,
    semantic_tokens,
    status_semantic,
)
from haagent.tui.widgets import ConversationTimeline, ProgressStatusLine, PromptInput
from haagent.tui.typography.wrap import is_textual_line_breaking_installed
from textual.widgets import Markdown, OptionList, RichLog, TextArea
from textual.screen import Screen

from tests.tui.support import (
    FakeAssistantService,
    _all_text,
    _approval_request,
    _assistant_event,
    _connection_record,
    _edit_diff_request,
    _interaction_requested_event,
    _interaction_response_event,
    _memory_candidate,
    _open_memory_panel,
    _runtime_event,
    _session_summary,
    _text,
    _tool_event,
    _user_input_request,
    _wait_for_conversation_bottom,
)

def test_tui_ctrl_q_cancels_running_task_before_exit(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            await pilot.press("ctrl+q")
            await pilot.pause(0.1)

            assert service.cancelled_count == 1
            service.release.set()

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
            if "工具审批" in _all_text(app):
                await pilot.press("n")
                await pilot.pause(0.2)
            assert after_help == before
            assert "审批确认" in rendered
            assert "y" in rendered
            assert "n" in rendered

    asyncio.run(run())

def test_tui_edit_diff_modal_returns_allow_and_deny_responses(tmp_path: Path) -> None:
    async def allow_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path / "allow", interaction_request=_edit_diff_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Write notes"
            await pilot.press("enter")
            await pilot.pause(0.2)
            rendered = _all_text(app)
            assert "文件改动审批" in rendered
            assert "notes.txt" in rendered
            assert "-old" in rendered
            assert "+new" in rendered
            await pilot.press("y")
            await pilot.pause(0.2)
            assert service.interaction_responses[-1].approved is True
            assert service.interaction_responses[-1].answer == "once"

    async def deny_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path / "deny", interaction_request=_edit_diff_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Write notes"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("n")
            await pilot.pause(0.2)
            assert service.interaction_responses[-1].approved is False
            assert service.interaction_responses[-1].answer == "deny"

    asyncio.run(allow_run())
    asyncio.run(deny_run())

def test_tui_pending_input_answer_uses_enter_and_continues_same_turn(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, interaction_request=_user_input_request())
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Inspect"
            await pilot.press("enter")
            await pilot.pause(0.2)
            input_widget.value = "README.md\nand docs"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts == ["Inspect"]
            assert service.interaction_responses == [
                HumanInteractionResponse(approved=True, answer="README.md\nand docs"),
            ]
            assert input_widget.value == ""

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
            conversation = _text(app, "#conversation")
            await pilot.press("n")
            await pilot.pause(0.1)
            assert "工具审批" in modal_text
            assert "shell" in modal_text
            assert "Approve high risk tool shell?" in modal_text
            assert "uv run pytest -q" in modal_text
            assert "会执行本地命令" in modal_text
            assert deny_has_focus
            assert "state: waiting approval" in status
            assert list(app.query("#side-bar")) == []
            assert "需要确认：shell" in conversation
            assert "建议：在弹窗中确认或拒绝" in conversation
            assert "工具 1 项" not in conversation
            assert "1 待确认" not in conversation
            assert "shell" in conversation

    asyncio.run(run())

def test_tui_running_task_can_cancel_and_submit_again(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            await pilot.press("ctrl+x")
            await pilot.pause(0.2)

            assert service.cancelled_count == 1
            assert "state: cancelling" in _text(app, "#status-bar")
            assert "任务正在取消" in _text(app, "#conversation")
            service.release.set()
            await pilot.pause(0.2)
            assert "state: cancelled" in _text(app, "#status-bar")
            assert app._pending_interaction is None
            assert "任务已取消" in _text(app, "#conversation")

            service.started.clear()
            input_widget.value = "Second task"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts[-1] == "Second task"

    asyncio.run(run())

def test_tui_cancel_returns_idle_when_no_active_run_remains(tmp_path: Path) -> None:
    class IdleCancelService(FakeAssistantService):
        def cancel_current_run(self):
            self.cancelled_count += 1
            return SimpleNamespace(status="idle", reason="no_active_run")

    async def run() -> None:
        service = IdleCancelService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            app._state = "running"
            app.action_cancel_current_task()
            await asyncio.sleep(0)

            assert service.cancelled_count == 1
            assert "state: idle" in _text(app, "#status-bar")
            assert "当前没有仍在运行的任务" in _text(app, "#conversation")

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
            assert "审批已允许：shell" not in _text(app, "#conversation")
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
            assert "审批已拒绝：shell" in _text(app, "#conversation")
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
            assert "需要补充" in conversation
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
            assert "回答已提交：request_user_input" not in conversation
            assert "需要补充信息" in conversation
            assert "Which file should I inspect?" in conversation
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
            assert "回答已取消：request_user_input" in conversation
            assert "已处理 1 项 >" in conversation
            assert "工具运行失败：request_user_input" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "工具运行失败：request_user_input" in conversation
            assert "工具 1 项" not in conversation
            assert "request_user_input" in conversation
            assert "失败" in conversation
            assert "state: failed" in _text(app, "#status-bar")

    asyncio.run(run())

def test_tui_interaction_reused_event_does_not_enter_pending_interaction(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _runtime_event(
                    "interaction_reused",
                    1,
                    interaction_type="user_input",
                    tool_name="request_user_input",
                    status="answered",
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
            assert "需要补充" not in _text(app, "#conversation")
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

def test_tui_cancelled_failure_event_does_not_show_none_placeholders(tmp_path: Path) -> None:
    async def run() -> None:
        episode_path = tmp_path / ".runs" / "episode-cancelled"
        service = FakeAssistantService(
            workspace_root=tmp_path,
            failure_event=FailureNoticeEvent(
                session_id="session-test",
                turn_index=1,
                status="cancelled",
                failed_stage="none",
                failure_category="none",
                reason="none",
                episode_path=str(episode_path),
            ),
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "停止当前任务"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "阶段：cancelled" in conversation
            assert "来源：Runtime Failure" in conversation
            assert "错误：user cancelled current run" in conversation
            assert "阶段：none" not in conversation
            assert "来源：none" not in conversation
            assert "错误：none" not in conversation

    asyncio.run(run())

