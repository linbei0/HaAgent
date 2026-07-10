"""
tests/tui/test_tools_details.py - HaAgent TUI tools 集成测试

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

def test_tui_tools_entry_points_are_removed() -> None:
    registry = command_registry()
    chat_footer = footer_text("chat")
    chat_help = help_body("chat")
    binding_actions = {binding.action if hasattr(binding, "action") else binding[1] for binding in APP_BINDINGS}

    assert parse_slash_command("/tools", registry).error == "未知命令：/tools"
    assert "tools" not in {command.name for command in registry.commands()}
    assert "/tools" not in chat_footer
    assert "/tools" not in chat_help
    assert "任务工作台" not in chat_help
    assert "focus_tools" not in binding_actions

def test_tui_timeline_hides_worker_lifecycle_internal_events(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _runtime_event(
                    "worker_started",
                    1,
                    agent_id="explorer-1",
                    task_id="task-1",
                    team_id="team-session-test",
                    subagent_type="explorer",
                    description="Inspect project",
                    status="running",
                ),
                _runtime_event(
                    "worker_completed",
                    1,
                    agent_id="explorer-1",
                    task_id="task-1",
                    team_id="team-session-test",
                    subagent_type="explorer",
                    description="Inspect project",
                    status="completed",
                ),
            ],
            assistant_content="已综合 worker 结果。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "分派检查"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "已综合 worker 结果。" in conversation
            assert "agent:explorer-1" not in conversation
            assert "worker_started" not in conversation
            assert "worker_completed" not in conversation

    asyncio.run(run())

def test_tui_progress_status_line_updates_and_clears(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            status = app.query_one("#progress-status", ProgressStatusLine)

            assert status.display is False

            app.set_progress_status(
                ProgressStatusState(
                    text="正在阅读文件...",
                    severity="info",
                    turn_index=1,
                    source="tool",
                )
            )

            assert status.display is True
            assert "正在阅读文件..." in _text(app, "#progress-status")

            app.clear_progress_status()

            assert status.display is False
            assert _text(app, "#progress-status") == ""

    asyncio.run(run())

def test_plain_greeting_never_shows_task_progress_or_status_line(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            assistant_content="你好！我是 HaAgent。",
            extra_events=[
                TaskProgressEvent(
                    session_id="session-test",
                    turn_index=1,
                    model_turn=1,
                    event_name="task_step_progress",
                    step_id="step-001",
                    title="你好",
                    status="running",
                    summary="model turn started",
                    owner="main",
                    category="model_turn_started",
                )
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "你好"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "你好！我是 HaAgent。" in conversation
            assert "任务进度" not in conversation
            assert "step-001" not in conversation
            assert "model_turn_started" not in conversation
            assert _text(app, "#progress-status") == ""

    asyncio.run(run())

def test_tui_read_only_tool_activity_does_not_persist_when_details_off(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "file_read"),
                _tool_event("tool_finished", 1, "file_read"),
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
                _tool_event("tool_started", 1, "web_fetch"),
            ],
            assistant_content="资料已经核对。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "联网核对"
            await pilot.press("enter")
            await pilot.pause(0.2)

            compact = _text(app, "#conversation")
            assert "资料已经核对。" in compact
            assert "工具 " not in compact
            assert "file_read" not in compact
            assert "web_search" not in compact
            assert "web_fetch" not in compact

    asyncio.run(run())

def test_tui_empty_tool_summary_is_not_rendered(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="资料已经核对。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "联网核对"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assert "工具 0 项" not in _text(app, "#conversation")
            assert not any(widget.display for widget in conversation.query(".timeline-tools"))

    asyncio.run(run())

def test_tui_tool_failure_adds_actionable_notice_and_keeps_answer_readable(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "file_read"),
                _tool_event("tool_finished", 1, "file_read"),
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_failed", 1, "web_search", message="timeout"),
            ],
            assistant_content="我已经整理好结论。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "整理资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "已处理 1 项 >" in conversation
            assert "工具运行失败：web_search" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "工具运行失败：web_search" in conversation
            assert "建议：查看错误摘要后重试或调整命令" in conversation
            assert "详情：点击展开" in conversation
            assert "工具 2 项" not in conversation
            assert "运行中" not in conversation
            assert "web_search" in conversation
            assert "我已经整理好结论。" in conversation

    asyncio.run(run())

def test_tui_tool_failure_notice_omits_long_read_only_summaries(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "file_read", message="reading very long local file"),
                _tool_event("tool_finished", 1, "file_read", message="file_read finished with a long summary"),
                _tool_event("tool_started", 1, "web_search", message="searching the web with a long query"),
                _tool_event("tool_failed", 1, "web_search", message="web_search failed with timeout"),
            ],
            assistant_content="我已经整理好结论。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "整理资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "已处理 1 项 >" in conversation
            assert "工具运行失败：web_search" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "工具运行失败：web_search" in conversation
            assert "工具 2 项" not in conversation
            assert "reading very long local file" not in conversation
            assert "web_search failed with timeout" not in conversation

    asyncio.run(run())

def test_tui_read_only_tool_events_do_not_count_calls_in_timeline(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
                _tool_event("tool_started", 1, "mcp__exa__web_search_exa"),
                _tool_event("tool_finished", 1, "mcp__exa__web_search_exa"),
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
                _tool_event("tool_started", 1, "web_fetch"),
                _tool_event("tool_failed", 1, "web_fetch", message="timed out"),
                _tool_event("tool_started", 1, "web_fetch"),
            ],
            assistant_content="资料已经核对。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "联网查证"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert "已处理 1 项 >" in conversation
            assert "工具运行失败：web_fetch" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "工具运行失败：web_fetch" in conversation
            assert "工具 5 项" not in conversation
            assert "3 成功" not in conversation
            assert "1 运行中" not in conversation
            assert "1 失败" not in conversation
            assert "工具 9 项" not in conversation
            assert "5 运行中" not in conversation

    asyncio.run(run())

def test_tui_tool_summary_updates_pending_confirmation_on_response_events(tmp_path: Path) -> None:
    def interaction_event(event_type: str, turn_index: int, tool_name: str) -> RuntimeUiEvent:
        return _runtime_event(event_type, turn_index, tool_name=tool_name, question="Approve?", approved=None)

    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "code_run"),
                interaction_event("approval_requested", 1, "code_run"),
                interaction_event("approval_denied", 1, "code_run"),
                _tool_event("tool_started", 1, "shell"),
                interaction_event("approval_requested", 1, "shell"),
                interaction_event("approval_granted", 1, "shell"),
                _tool_event("tool_started", 1, "file_write"),
                interaction_event("edit_diff_requested", 1, "file_write"),
                interaction_event("edit_diff_denied", 1, "file_write"),
            ],
            assistant_content="审批状态已处理。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "需要审批的操作"
            await pilot.press("enter")
            await pilot.pause(0.4)

            conversation = _text(app, "#conversation")
            assert "已处理 5 项 >" in conversation
            assert "需要确认：code_run" not in conversation
            assert "审批已拒绝：code_run" not in conversation
            assert "需要确认：shell" not in conversation
            assert "需要确认文件改动" not in conversation
            assert "文件改动已拒绝" not in conversation
            app.query_one("#conversation", ConversationTimeline).toggle_process_group(1)
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "需要确认：code_run" in conversation
            assert "审批已拒绝：code_run" in conversation
            assert "需要确认：shell" in conversation
            assert "需要确认文件改动" in conversation
            assert "文件改动已拒绝" in conversation
            assert "工具 3 项" not in conversation
            assert "1 运行中" not in conversation
            assert "2 失败" not in conversation
            assert "审批已允许：shell" not in conversation

    asyncio.run(run())

def test_tui_details_command_toggles_full_tool_activity(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "file_read"),
                _tool_event("tool_finished", 1, "file_read"),
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
                _runtime_event(
                    "compression_diagnostic",
                    1,
                    subject="web_search",
                    stage="historical_tool_message",
                    original_chars=1854,
                    final_chars=929,
                    decision="collapsed",
                    reason="long_text_result",
                ),
                _runtime_event(
                    "loop_suggestion_added",
                    1,
                    tool_name="file_write",
                    message="File change succeeded. Consider reading back notes.md.",
                ),
                _tool_event("tool_started", 1, "web_fetch"),
            ],
            assistant_content="资料已经核对。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "联网核对"
            await pilot.press("enter")
            await pilot.pause(0.2)
            compact = _text(app, "#conversation")
            assert "工具 1 项" in compact
            assert "web_fetch" not in compact
            assert "file_read" not in compact
            assert "result compacted" not in compact
            assert "File change succeeded" not in compact
            assert "旧工具消息降级" not in compact

            input_widget.value = "/details"
            await pilot.press("enter")
            await pilot.pause(0.1)
            detailed = _text(app, "#conversation")
            assert "工具详情已开启" in detailed
            assert "工具 web_search ok" in detailed
            assert "旧工具消息降级：web_search 1854 chars -> 929 chars" in detailed
            assert "web_fetch" not in detailed
            assert "工具 file_read ok" not in detailed
            assert "File change succeeded" not in detailed

    asyncio.run(run())

def test_tui_tool_details_use_inline_log_widget(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
            ],
            assistant_content="资料已经核对。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "联网核对"
            await pilot.press("enter")
            await pilot.pause(0.2)

            tool_log = app.query_one(".timeline-assistant .timeline-tools")
            assert tool_log.__class__.__name__ == "ToolActivityLog"
            assert getattr(tool_log, "max_lines") == 32
            assert tool_log.styles.overflow_x == "hidden"
            assert tool_log.styles.overflow_y == "hidden"
            assert tool_log.show_horizontal_scrollbar is False
            assert tool_log.show_vertical_scrollbar is False
            assert tool_log.styles.color == app.screen.styles.color
            selection_style = app.screen.get_component_rich_style("screen--selection")
            assert selection_style.color is not None
            assert selection_style.bgcolor is not None
            assert selection_style.color != selection_style.bgcolor

    asyncio.run(run())

