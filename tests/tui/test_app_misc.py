"""
tests/tui/test_app_misc.py - HaAgent TUI misc 集成测试

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

def test_tui_timeline_uses_distinct_message_widgets_for_visual_hierarchy(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
            ],
            assistant_content="已经完成搜索。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "搜索资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assert conversation.has_class("timeline-ready")
            assert list(conversation.query(".timeline-item"))
            assert list(conversation.query(".timeline-user"))
            assert list(conversation.query(".timeline-assistant"))
            assert list(conversation.query(".timeline-tools"))
            assert list(conversation.query(".timeline-body"))

    asyncio.run(run())

