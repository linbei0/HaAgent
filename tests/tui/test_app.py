"""
tests/tui/test_app.py - HaAgent TUI 垂直切片测试

验证 TUI adapter 通过 AssistantService 风格接口展示状态、运行 prompt 和接收事件。
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
    ToolActivityEvent,
    UserInputStateEvent,
)
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.application.app import HaAgentTuiApp, find_untrusted_absolute_paths
from haagent.tui.commands import SlashCommandResult, command_registry, parse_slash_command
from haagent.tui.design.failures import failure_next_steps
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, fuzzy_file_matches, path_reference_token
from haagent.tui.design.keys import APP_BINDINGS, footer_text, help_body, key_help_lines
from haagent.tui.overlays.models import ModelCatalogLoadingOverlay
from haagent.tui.design.copy import MODAL_TITLES, PANEL_TITLES
from haagent.tui.design.renderers import memory_panel_text, status_line
from haagent.tui.state.search import ConversationSearchState
from haagent.tui.overlays.sessions import SessionOverlayState
from haagent.tui.state import ResponsiveLayout, layout_for_size
from haagent.tui.design.theme import (
    SemanticToken,
    TuiThemeMode,
    no_color_enabled,
    select_theme,
    semantic_tokens,
    status_semantic,
)
from haagent.tui.widgets import ConversationTimeline, PromptInput
from haagent.tui.typography.wrap import is_textual_line_breaking_installed
from textual.widgets import Markdown, RichLog, TextArea
from textual.screen import Screen


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
        failure_event: RuntimeUiEvent | None = None,
        interaction_request: HumanInteractionRequest | None = None,
        extra_events: list[RuntimeUiEvent] | None = None,
        memory_candidates: list[MemoryCandidate] | None = None,
        memory_error: Exception | None = None,
        current_session_id: str = "session-test",
        sessions: list[AssistantSessionSummary] | None = None,
        session_histories: dict[str, list[SimpleNamespace]] | None = None,
        enable_web: bool = False,
        external_roots: list[dict[str, str]] | None = None,
        permission_mode: str = "request_approval",
        skills: list[dict[str, object]] | None = None,
        blocked_project_skill_roots: list[str] | None = None,
        marketplace_results: list[SimpleNamespace] | None = None,
        marketplace_warnings: list[str] | None = None,
        mcp_status: dict[str, object] | None = None,
        agents: list[dict[str, object]] | None = None,
        sandbox_status: AssistantSandboxStatus | None = None,
        sandbox_doctor_report: SandboxDoctorReport | None = None,
        image_input_supported: bool | None = None,
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
        self.sessions = list(sessions or [])
        self.session_histories = dict(session_histories or {})
        self.enable_web = enable_web
        self.external_roots = list(external_roots or [])
        self.permission_mode = permission_mode
        self.skills = list(skills or [])
        self.blocked_project_skill_roots = list(blocked_project_skill_roots or [])
        self.marketplace_results = list(marketplace_results or [])
        self.marketplace_warnings = list(marketplace_warnings or [])
        self.agents = list(agents or [])
        self.sandbox_status = sandbox_status or AssistantSandboxStatus(
            backend="local_subprocess",
            degraded=True,
            reason="docker sandbox disabled",
        )
        self.sandbox_doctor_report = sandbox_doctor_report or SandboxDoctorReport(
            backend="local_subprocess",
            ready=False,
            docker_cli="not_checked",
            docker_daemon="not_checked",
            image="not_checked",
            auto_build_image=True,
            reason="docker sandbox disabled",
            next_action="Run `haagent sandbox enable docker` to enable Docker isolation.",
        )
        self.image_input_supported = image_input_supported
        self.mcp_status = dict(
            mcp_status
            or {
                "configured_count": 0,
                "connected_count": 0,
                "failed_count": 0,
                "servers": [],
            }
        )
        self.trusted_skills_count = 0
        self.untrusted_skills_count = 0
        self.read_skill_names: list[str] = []
        self.searched_marketplace_queries: list[tuple[str, tuple[str, ...] | None, int]] = []
        self.installed_marketplace_ids: list[str] = []
        self.compacted_count = 0
        self.next_turn_target_paths: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []
        self.prompt_attachments: list[list[object]] = []
        self.clipboard_attachments: list[object] = []
        self.interaction_responses: list[HumanInteractionResponse] = []
        self.confirmed_candidate_ids: list[str] = []
        self.rejected_candidate_ids: list[tuple[str, str]] = []
        self.created_sessions: list[str] = []
        self.resumed_sessions: list[str] = []
        self.continued_latest_count = 0
        self.cancelled_count = 0
        self.sandbox_enabled_count = 0
        self.sandbox_disabled_count = 0
        self.model_profiles: list[SimpleNamespace] = []
        self.switched_model_profile: str | None = None
        self.default_model_profile: str | None = None
        self.deleted_model_profile: str | None = None
        self.catalog_providers: list[SimpleNamespace] = []
        self.cached_catalog_providers: list[SimpleNamespace] | None = None
        self.configured_model_profile = None
        self.configured_api_key: str | None = None
        self.connection_test_result = SimpleNamespace(
            ok=True,
            profile_name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            message="OK",
        )
        self.tested_model_profile: str | None = None
        self.refreshed_catalog_count = 0
        self.got_catalog_count = 0
        self.catalog_refresh_release: threading.Event | None = None

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
            web_enabled=self.enable_web,
            external_roots=list(self.external_roots),
            permission_mode=self.permission_mode,
            sandbox_status=self.sandbox_status,
            image_input_supported=self.image_input_supported,
        )

    def get_mcp_status(self) -> dict[str, object]:
        return dict(self.mcp_status)

    def list_agents(self) -> list[dict[str, object]]:
        return list(self.agents)

    def get_sandbox_status(self) -> AssistantSandboxStatus:
        return self.sandbox_status

    def get_sandbox_doctor_report(self) -> SandboxDoctorReport:
        return self.sandbox_doctor_report

    def enable_docker_sandbox(self, *, fail_if_unavailable: bool = True) -> AssistantSandboxStatus:
        self.sandbox_enabled_count += 1
        self.sandbox_status = AssistantSandboxStatus(
            backend="docker",
            degraded=False,
            reason="" if fail_if_unavailable else "fallback allowed",
        )
        return self.sandbox_status

    def disable_sandbox(self) -> AssistantSandboxStatus:
        self.sandbox_disabled_count += 1
        self.sandbox_status = AssistantSandboxStatus(
            backend="local_subprocess",
            degraded=True,
            reason="docker sandbox disabled",
        )
        return self.sandbox_status

    def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
        self.prompts.append(prompt)
        self.prompt_attachments.append(list(attachments or []))
        self.started.set()
        if event_sink is not None:
            if self.failure_event is not None:
                event_sink(self.failure_event)
                return SimpleNamespace(status="failed")
            for extra_event in self.extra_events:
                event_sink(extra_event)
        if self.block_until_released:
            self.release.wait(timeout=2)
        if self.cancelled_count:
            return SimpleNamespace(status="cancelled")
        if event_sink is not None:
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
                        ToolActivityEvent(
                            session_id="session-test",
                            turn_index=len(self.prompts),
                            model_turn=None,
                            tool_name=request.tool_name,
                            status="failed",
                            summary="interaction declined",
                            error_type=(
                                "approval_denied"
                                if request.interaction_type == "approval"
                                else "user_input_unavailable"
                            ),
                            error_message="interaction declined",
                        ),
                    )
                    return SimpleNamespace(status="failed")
            event_sink(
                AssistantMessageEvent(
                    session_id="session-test",
                    turn_index=len(self.prompts),
                    model_turn=None,
                    content=self.assistant_content or f"assistant: {prompt}",
                ),
            )
        return SimpleNamespace(status="completed")

    def paste_clipboard_image(self, *, existing=None):
        attachment = SimpleNamespace(
            id=f"img-{len(self.clipboard_attachments) + 1}",
            filename=f"img-{len(self.clipboard_attachments) + 1}.png",
            mime_type="image/png",
            size_bytes=123,
            width=2,
            height=1,
            relative_path=f"attachments/img-{len(self.clipboard_attachments) + 1}.png",
        )
        self.clipboard_attachments.append(attachment)
        return attachment

    def create_session(self) -> AssistantSessionStatus:
        session_id = f"session-new-{len(self.created_sessions) + 1}"
        self.created_sessions.append(session_id)
        self.current_session_id = session_id
        return self._session_status(session_id)

    def resume_session(self, session: str | Path) -> AssistantSessionStatus:
        session_text = str(session)
        self.resumed_sessions.append(session_text)
        match = next((item for item in self.sessions if str(item.session_path) == session_text), None)
        self.current_session_id = match.session_id if match is not None else session_text
        return self._session_status(self.current_session_id)

    def continue_latest_session(self) -> AssistantSessionStatus:
        self.continued_latest_count += 1
        if self.sessions:
            self.current_session_id = self.sessions[0].session_id
        return self._session_status(self.current_session_id)

    def compact_current_session(self):
        self.compacted_count += 1
        return SimpleNamespace(
            applied=True,
            reason="applied",
            original_turn_count=8,
            compacted_turn_count=2,
            preserved_recent_count=6,
            saved_chars=1200,
        )

    def cancel_current_run(self):
        self.cancelled_count += 1
        return SimpleNamespace(status="cancelled", reason="user_cancelled")

    def list_sessions(self) -> list[AssistantSessionSummary]:
        return list(self.sessions)

    def current_session_history(self):
        return list(self.session_histories.get(self.current_session_id, []))

    def list_model_profiles(self):
        return list(self.model_profiles)

    def refresh_model_catalog(self):
        self.refreshed_catalog_count += 1
        if self.catalog_refresh_release is not None:
            self.catalog_refresh_release.wait(timeout=5)
        return SimpleNamespace(providers=list(self.catalog_providers), used_cache=False, error=None)

    def get_model_catalog(self):
        self.got_catalog_count += 1
        if self.catalog_refresh_release is not None:
            self.catalog_refresh_release.wait(timeout=5)
        providers = self.catalog_providers if self.cached_catalog_providers is None else self.cached_catalog_providers
        return SimpleNamespace(providers=list(providers), used_cache=True, error=None)

    def configure_model_profile(self, request):
        self.configured_model_profile = request
        self.configured_api_key = request.api_key
        return request

    def test_model_profile(self, profile_name: str):
        self.tested_model_profile = profile_name
        return self.connection_test_result

    def switch_current_session_model(self, profile_name: str) -> AssistantSessionStatus:
        self.switched_model_profile = profile_name
        return AssistantSessionStatus(
            session_id=self.current_session_id,
            workspace_root=self.workspace_root,
            runs_root=self.workspace_root / ".runs",
            session_path=self.workspace_root / ".runs" / "sessions" / self.current_session_id,
            turn_count=len(self.prompts),
            max_turns=None,
            provider=self.provider or "-",
            model_profile_name=profile_name,
            model=next((item.model for item in self.model_profiles if item.name == profile_name), self.model),
            base_url="https://api.deepseek.com",
            external_roots=list(self.external_roots),
            permission_mode=self.permission_mode,
        )

    def set_default_model_profile(self, profile_name: str) -> None:
        self.default_model_profile = profile_name
        for index, profile in enumerate(self.model_profiles):
            self.model_profiles[index] = SimpleNamespace(
                **{**profile.__dict__, "active": profile.name == profile_name},
            )

    def set_web_enabled(self, enabled: bool) -> AssistantWorkspaceStatus:
        self.enable_web = enabled
        return self.get_workspace_status()

    def delete_model_profile(self, profile_name: str) -> None:
        self.deleted_model_profile = profile_name
        self.model_profiles = [profile for profile in self.model_profiles if profile.name != profile_name]

    def _session_status(self, session_id: str) -> AssistantSessionStatus:
        return AssistantSessionStatus(
            session_id=session_id,
            workspace_root=self.workspace_root,
            runs_root=self.workspace_root / ".runs",
            session_path=self.workspace_root / ".runs" / "sessions" / session_id,
            turn_count=len(self.prompts),
            max_turns=None,
            provider=self.provider or "-",
            web_enabled=self.enable_web,
            external_roots=list(self.external_roots),
            permission_mode=self.permission_mode,
        )

    def set_permission_mode(self, mode: str) -> AssistantSessionStatus:
        self.permission_mode = mode
        return self._session_status(self.current_session_id)

    def set_next_turn_target_paths(self, paths: list[str | Path]) -> AssistantSessionStatus:
        self.next_turn_target_paths = [str(Path(path).resolve()) for path in paths]
        return self._session_status(self.current_session_id)

    def add_external_root(self, path: str | Path, access: str) -> AssistantSessionStatus:
        resolved = str(Path(path).resolve())
        self.external_roots = [root for root in self.external_roots if root["path"] != resolved]
        self.external_roots.append({"path": resolved, "access": access, "source": "user"})
        return self._session_status(self.current_session_id)

    def remove_external_root(self, path: str | Path) -> AssistantSessionStatus:
        resolved = str(Path(path).resolve())
        self.external_roots = [root for root in self.external_roots if root["path"] != resolved]
        return self._session_status(self.current_session_id)

    def set_external_root_access(self, path: str | Path, access: str) -> AssistantSessionStatus:
        self.add_external_root(path, access)
        return self._session_status(self.current_session_id)

    def clear_external_roots(self) -> AssistantSessionStatus:
        self.external_roots = []
        return self._session_status(self.current_session_id)

    def switch_project_root(self, path: str | Path) -> AssistantSessionStatus:
        self.workspace_root = Path(path).resolve()
        self.external_roots = []
        return self._session_status(self.current_session_id)

    def list_skills(self):
        return SimpleNamespace(
            skills=list(self.skills),
            blocked_project_skill_roots=list(self.blocked_project_skill_roots),
        )

    def trust_project_skills(self):
        self.trusted_skills_count += 1
        self.blocked_project_skill_roots = []
        return self.list_skills()

    def untrust_project_skills(self):
        self.untrusted_skills_count += 1
        return self.list_skills()

    def read_skill_for_user(self, name: str):
        self.read_skill_names.append(name)
        match = next((item for item in self.skills if item.get("name") == name or item.get("command_name") == name), None)
        if match is None:
            raise RuntimeError(f"skill not found: {name}")
        return SimpleNamespace(
            name=str(match.get("name")),
            command_name=str(match.get("command_name") or match.get("name")),
            content=f"# {match.get('name')}\nFollow this workflow.",
        )

    def search_skill_marketplace(self, query: str, *, providers=None, limit: int = 10):
        provider_tuple = tuple(providers) if providers is not None else None
        self.searched_marketplace_queries.append((query, provider_tuple, limit))
        return SimpleNamespace(
            status="success" if self.marketplace_results else "error",
            query=query,
            results=list(self.marketplace_results),
            warnings=list(self.marketplace_warnings),
        )

    def install_marketplace_skill(self, result_id: str):
        self.installed_marketplace_ids.append(result_id)
        match = next((item for item in self.marketplace_results if item.result_id == result_id), None)
        if match is None:
            raise RuntimeError(f"unknown marketplace result id: {result_id}")
        if not match.installable:
            raise RuntimeError("only skills_sh results are installable in marketplace v1")
        return SimpleNamespace(
            name=match.name,
            command_name=match.name.lower().replace(" ", "-"),
            skill_dir=self.workspace_root / ".haagent" / "skills" / match.name.lower().replace(" ", "-"),
            skill_file=self.workspace_root / ".haagent" / "skills" / match.name.lower().replace(" ", "-") / "SKILL.md",
            source_url=match.detail_url,
        )

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
    plain_text = getattr(widget, "plain_text", None)
    if isinstance(plain_text, str):
        return plain_text
    if isinstance(widget, TextArea):
        return widget.text
    if isinstance(widget, RichLog):
        return "\n".join("".join(segment.text for segment in line) for line in widget.lines)
    return str(widget.content)


async def _wait_for_conversation_bottom(app: HaAgentTuiApp, pilot, *, attempts: int = 10) -> None:
    conversation = app.query_one("#conversation")
    for _ in range(attempts):
        if conversation.scroll_y == conversation.max_scroll_y:
            return
        await pilot.pause(0.05)


def _all_text(app: HaAgentTuiApp) -> str:
    widgets = list(app.query("*"))
    if app.screen is not None:
        widgets.extend(app.screen.query("*"))
    pieces = []
    for widget in widgets:
        plain_text = getattr(widget, "plain_text", None)
        if isinstance(plain_text, str):
            pieces.append(plain_text)
        elif isinstance(widget, TextArea):
            pieces.append(widget.text)
        elif isinstance(widget, RichLog):
            pieces.append("\n".join("".join(segment.text for segment in line) for line in widget.lines))
        else:
            pieces.append(str(widget.render()))
    return "\n".join(pieces)


async def _open_memory_panel(app: HaAgentTuiApp, pilot) -> None:
    prompt_input = app.query_one("#prompt-input")
    prompt_input.value = "/memory"
    await pilot.press("enter")
    await pilot.pause(0.1)


def _runtime_event(event_type: str, turn_index: int, **payload: object) -> RuntimeUiEvent:
    return RuntimeUiEventMapper.to_ui_event(
        {"event_type": event_type, **payload},
        session_id="session-test",
        turn_index=turn_index,
    )


def _interaction_requested_event(request: HumanInteractionRequest, turn_index: int) -> RuntimeUiEvent:
    if request.interaction_type == "approval":
        return ApprovalStateEvent(
            session_id="session-test",
            turn_index=turn_index,
            model_turn=None,
            tool_name=request.tool_name,
            state="requested",
            question=request.question,
            approved=None,
            args_summary=request.args_summary,
        )
    if request.interaction_type == "edit_diff":
        return ApprovalStateEvent(
            session_id="session-test",
            turn_index=turn_index,
            model_turn=None,
            tool_name=request.tool_name,
            state="requested",
            question=request.question,
            approved=None,
            args_summary=request.args_summary,
            approval_kind="edit_diff",
        )
    return UserInputStateEvent(
        session_id="session-test",
        turn_index=turn_index,
        model_turn=None,
        tool_name=request.tool_name,
        state="requested",
        question=request.question,
        reason=request.reason,
    )


def _tool_event(event_type: str, turn_index: int, tool_name: str, *, message: str | None = None) -> ToolActivityEvent:
    if event_type == "tool_finished":
        status = "finished"
    elif event_type == "tool_failed":
        status = "failed"
    else:
        status = "started"
    return ToolActivityEvent(
        session_id="session-test",
        turn_index=turn_index,
        model_turn=None,
        tool_name=tool_name,
        status=status,
        summary=message or tool_name,
    )


def _assistant_event(event_type: str, turn_index: int, text: str) -> RuntimeUiEvent:
    if event_type == "assistant_delta":
        return AssistantDeltaEvent("session-test", turn_index, None, text)
    return AssistantMessageEvent("session-test", turn_index, None, text)


def _interaction_response_event(
    request: HumanInteractionRequest,
    response: HumanInteractionResponse,
    turn_index: int,
) -> RuntimeUiEvent:
    if request.interaction_type == "approval":
        return ApprovalStateEvent(
            session_id="session-test",
            turn_index=turn_index,
            model_turn=None,
            tool_name=request.tool_name,
            state="granted" if response.approved else "denied",
            question=request.question,
            approved=response.approved,
            args_summary=request.args_summary,
        )
    if request.interaction_type == "edit_diff":
        return ApprovalStateEvent(
            session_id="session-test",
            turn_index=turn_index,
            model_turn=None,
            tool_name=request.tool_name,
            state="granted" if response.approved else "denied",
            question=request.question,
            approved=response.approved,
            args_summary=request.args_summary,
            approval_kind="edit_diff",
        )
    return UserInputStateEvent(
        session_id="session-test",
        turn_index=turn_index,
        model_turn=None,
        tool_name=request.tool_name,
        state="received",
        question=request.question,
        answer_chars=len(response.answer),
        approved=response.approved,
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


def _session_summary(tmp_path: Path, session_id: str, first_request: str, turn_count: int = 1) -> AssistantSessionSummary:
    return AssistantSessionSummary(
        session_id=session_id,
        created_at="2026-06-27T00:00:00+00:00",
        updated_at="2026-06-27T01:00:00+00:00",
        workspace_root=tmp_path,
        turn_count=turn_count,
        first_request=first_request,
        session_path=tmp_path / ".runs" / "sessions" / session_id,
    )


def test_tui_status_line_renderer_truncates_to_terminal_width(tmp_path: Path) -> None:
    status = FakeAssistantService(
        workspace_root=tmp_path / "very-long-workspace-name-for-status-rendering",
        model="very-long-model-name-for-status-rendering",
        current_session_id="session-abcdefghijklmnopqrstuvwxyz",
    ).get_workspace_status()

    line_80 = status_line(status, ui_state="waiting approval", width=80)
    line_120 = status_line(status, ui_state="running", width=120)

    assert len(line_80) <= 80
    assert len(line_120) <= 120
    assert "state: waiting approval" in line_80
    assert "state: running" in line_120


def test_tui_status_renderers_show_explicit_web_state(tmp_path: Path) -> None:
    offline = FakeAssistantService(workspace_root=tmp_path / "offline").get_workspace_status()
    online = FakeAssistantService(workspace_root=tmp_path / "online", enable_web=True).get_workspace_status()

    assert "web:off" in status_line(offline, ui_state="idle", width=120)
    assert "web:on" in status_line(online, ui_state="idle", width=120)


def test_tui_status_renderer_shows_sandbox_state(tmp_path: Path) -> None:
    status = FakeAssistantService(workspace_root=tmp_path / "sandbox").get_workspace_status()

    line = status_line(status, ui_state="idle", width=140)

    assert "sandbox:degraded" in line


def test_tui_keymap_help_and_footer_share_context_definitions() -> None:
    for context in ("chat", "memory_list", "memory_detail", "pending_input", "approval", "too_small"):
        footer = footer_text(context)
        help_text = help_body(context)
        for key, _description in key_help_lines(context, include_footer_only=False):
            assert key in help_text
        for key, _description in key_help_lines(context, footer_only=True):
            assert key in footer
    assert "Ctrl+T" in footer_text("chat")
    assert "切换主题" in help_body("chat")
    assert "End" not in footer_text("chat")
    assert "End" in help_body("chat")
    assert "回到底部" in help_body("chat")


def test_tui_chat_memory_entry_is_only_slash_command() -> None:
    chat_footer = footer_text("chat")
    chat_help = help_body("chat")
    binding_keys = {binding.key if hasattr(binding, "key") else binding[0] for binding in APP_BINDINGS}
    input_binding_keys = {binding.key for binding in PromptInput.BINDINGS}

    assert "/memory" in chat_help
    assert "[m]记忆" not in chat_footer
    assert "m" not in binding_keys
    assert "m" not in input_binding_keys


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


def test_tui_semantic_tokens_cover_required_statuses() -> None:
    assert semantic_tokens() == {
        SemanticToken.DEFAULT,
        SemanticToken.MUTED,
        SemanticToken.EMPHASIS,
        SemanticToken.SUCCESS,
        SemanticToken.WARNING,
        SemanticToken.ERROR,
        SemanticToken.INFO,
        SemanticToken.SELECTION,
        SemanticToken.FOCUS,
        SemanticToken.RUNNING,
        SemanticToken.CANCELLED,
        SemanticToken.PENDING,
        SemanticToken.DANGER,
    }

    expectations = {
        "idle": (SemanticToken.DEFAULT, "-", "空闲"),
        "running": (SemanticToken.RUNNING, "...", "运行中"),
        "waiting approval": (SemanticToken.WARNING, "?", "待审批"),
        "waiting input": (SemanticToken.PENDING, "?", "待补充"),
        "done": (SemanticToken.SUCCESS, "ok", "成功"),
        "failed": (SemanticToken.ERROR, "!", "失败"),
        "cancelled": (SemanticToken.CANCELLED, "x", "已取消"),
        "denied": (SemanticToken.DANGER, "!", "已拒绝"),
    }

    for raw_status, (token, symbol, label) in expectations.items():
        semantic = status_semantic(raw_status)
        assert semantic.token is token
        assert semantic.symbol == symbol
        assert semantic.label == label
        assert semantic.css_class == f"status-{token.value}"


def test_tui_theme_selection_respects_env_and_no_color(monkeypatch) -> None:
    assert not no_color_enabled({})
    assert no_color_enabled({"NO_COLOR": "1"})
    assert no_color_enabled({"NO_COLOR": ""})

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)
    assert select_theme().mode is TuiThemeMode.DARK
    assert select_theme("light").mode is TuiThemeMode.LIGHT
    assert select_theme("monochrome").mode is TuiThemeMode.MONOCHROME

    monkeypatch.setenv("HAAGENT_TUI_THEME", "light")
    assert select_theme().mode is TuiThemeMode.LIGHT
    monkeypatch.setenv("NO_COLOR", "1")
    assert select_theme().mode is TuiThemeMode.MONOCHROME


def test_tui_copy_titles_are_chinese_and_keep_protocol_names() -> None:
    assert PANEL_TITLES["conversation"] == "对话"
    assert PANEL_TITLES["sessions"] == "会话"
    assert PANEL_TITLES["memory"] == "记忆候选"
    assert PANEL_TITLES["search"] == "搜索"
    assert MODAL_TITLES["approval"] == "工具审批"
    assert "workbench" not in PANEL_TITLES
    assert "tools" not in PANEL_TITLES
    assert "tool_details" not in MODAL_TITLES


def test_tui_memory_panel_renderer_marks_selection_and_detail() -> None:
    candidates = [
        _memory_candidate("cand_first", "第一条"),
        _memory_candidate("cand_second", "第二条"),
    ]

    list_text = memory_panel_text(
        candidates=candidates,
        selected_index=1,
        detail_mode=False,
        notice="发现候选",
        error=None,
    )
    detail_text = memory_panel_text(
        candidates=candidates,
        selected_index=1,
        detail_mode=True,
        notice=None,
        error=None,
    )

    assert "  发现候选" in list_text
    assert "  cand_first" in list_text
    assert "> cand_second" in list_text
    assert "candidate_id: cand_second" in detail_text
    assert "candidate_id: cand_first" not in detail_text


def test_tui_responsive_layout_state_is_testable_without_widgets() -> None:
    assert layout_for_size(79, 24) == ResponsiveLayout(too_small=True)
    assert layout_for_size(80, 23) == ResponsiveLayout(too_small=True)
    assert layout_for_size(80, 24) == ResponsiveLayout(too_small=False)
    assert layout_for_size(120, 24) == ResponsiveLayout(too_small=False)


def test_tui_slash_command_registry_parses_known_and_unknown_commands() -> None:
    registry = command_registry()

    result = parse_slash_command("/sessions", registry)
    model = parse_slash_command("/model", registry)
    models = parse_slash_command("/models", registry)
    unknown = parse_slash_command("/wat", registry)
    not_command = parse_slash_command(" /help", registry)

    assert result == SlashCommandResult(command=registry.require("sessions"), argument="")
    assert model.command.action == "open_models"
    assert model.command.name == "model"
    assert models.command.action == "open_models"
    assert models.command.name == "model"
    assert unknown.command is None
    assert unknown.error == "未知命令：/wat"
    assert not_command is None
    assert parse_slash_command("/review 看看改动", registry) is None
    assert parse_slash_command("/debug", registry) is None
    assert parse_slash_command("/verify", registry) is None
    assert {command.name for command in registry.commands()} >= {
        "help",
        "sessions",
        "compact",
        "memory",
        "skills",
        "skill",
        "sandbox",
        "new",
        "resume",
        "model",
        "mcp",
        "agents",
        "web",
        "permissions",
        "review",
        "debug",
        "verify",
    }


def test_tui_prompt_pack_command_suggestion_fills_prompt_input(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.pause(0.1)
            app._command_suggestion_overlay.update_query("rev")

            app.action_accept_command_suggestion()

            input_widget = app.query_one("#prompt-input", PromptInput)
            assert app._prompt_value(input_widget) == "/review "
            assert service.prompts == []
            assert "未知命令" not in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_prompt_pack_command_is_submitted_to_chat_runtime(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            app._set_prompt_value(input_widget, "/review 看看改动")

            app._submit_prompt(input_widget)
            await asyncio.to_thread(service.started.wait, 2)

            assert service.prompts == ["/review 看看改动"]
            assert "未知命令" not in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_sandbox_command_shows_status_doctor_and_updates_settings(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")

            input_widget.value = "/sandbox"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "当前沙箱：local_subprocess" in conversation
            assert "haagent sandbox enable docker" in conversation

            input_widget.value = "/sandbox doctor"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "Docker CLI: not_checked" in conversation
            assert "docker sandbox disabled" in conversation

            input_widget.value = "/sandbox enable docker"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert service.sandbox_enabled_count == 1
            assert "Docker 沙箱已启用" in conversation
            assert "新 session 生效" in conversation
            assert "sandbox:docker" in _text(app, "#status-bar")

            input_widget.value = "/sandbox disable"
            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert service.sandbox_disabled_count == 1
            assert "已恢复 local_subprocess" in conversation
            assert "sandbox:degraded" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_agents_command_lists_current_workers(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        agents=[
            {
                "agent_id": "explorer-1",
                "task_id": "task-1",
                "team_id": "team-session-test",
                "subagent_type": "explorer",
                "description": "Inspect project",
                "status": "running",
            },
            {
                "agent_id": "verification-1",
                "task_id": "task-2",
                "team_id": "team-session-test",
                "subagent_type": "verification",
                "description": "Run tests",
                "status": "completed",
            },
        ],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/agents"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "Workers" in conversation
            assert "explorer-1" in conversation
            assert "running" in conversation
            assert "Inspect project" in conversation
            assert "verification-1" in conversation
            assert "completed" in conversation
            assert service.prompts == []

    asyncio.run(run())


def test_tui_timeline_shows_worker_lifecycle_events(tmp_path: Path) -> None:
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
            assert "agent:explorer-1" in conversation
            assert "已综合 worker 结果。" in conversation

    asyncio.run(run())


def test_tui_slash_command_registry_includes_mcp() -> None:
    registry = command_registry()

    assert registry.get("mcp") is not None
    assert "models" not in {command.name for command in registry.commands()}


def test_tui_installs_unicode_line_breaking_for_assistant_markdown(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        assistant_content="各地举行建国 250 周年庆祝活动，约 400 架无人机参与。",
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(42, 24)) as pilot:
            assert is_textual_line_breaking_installed()
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "看看新闻"
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert "建国 250 周年" in _text(app, "#conversation")
            rendered = app.query_one(Markdown).render()
            rendered_text = getattr(rendered, "plain", str(rendered))
            assert "250\n周年" not in rendered_text
            assert "400\n架" not in rendered_text

    asyncio.run(run())


def test_tui_compact_command_compacts_session_without_running_prompt(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/compact"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.compacted_count == 1
            assert service.prompts == []
            assert "已压缩当前会话" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_skills_command_lists_skills_and_trusts_project_roots(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
        blocked_project_skill_roots=[str(tmp_path / ".haagent" / "skills")],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            before = _text(app, "#conversation")

            input_widget.value = "/skills"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "Skills" in _all_text(app)
            assert "review" in _all_text(app)
            assert _text(app, "#conversation") == before
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == ""
            assert service.prompts == []
            await pilot.press("escape")
            await pilot.pause(0.1)

            input_widget.value = "/skills trust"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.trusted_skills_count == 1
            assert "已信任当前 workspace 的项目 skills" in _text(app, "#conversation")

    asyncio.run(run_test())


def test_tui_skills_search_lists_marketplace_results(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skills_sh-1",
                provider="skills_sh",
                name="analyze-csv",
                source="office",
                summary="Analyze CSV files.",
                detail_url="https://skills.sh/office/analyze-csv",
                installable=True,
                quality={"installs": 1234},
            ),
            SimpleNamespace(
                result_id="skillsmp-2",
                provider="skillsmp",
                name="csv-helper",
                source="data-team",
                summary="Find CSV workflows.",
                detail_url="https://skillsmp.com/skills/csv-helper",
                installable=False,
                quality={"stars": 42},
            ),
        ],
        marketplace_warnings=["skillsmp search failed: HTTP 502"],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills search csv"
            await pilot.press("enter")

            conversation = _text(app, "#conversation")
            assert service.searched_marketplace_queries == [("csv", None, 10)]
            assert "analyze-csv" in conversation
            assert "skills_sh-1" in conversation
            assert "skills_sh" in conversation
            assert "可安装" in conversation
            assert "csv-helper" in conversation
            assert "skillsmp" in conversation
            assert "暂不支持直接安装" in conversation
            assert "skillsmp search failed: HTTP 502" in conversation

    asyncio.run(run_test())


def test_tui_skills_install_installs_cached_marketplace_result(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skills_sh-1",
                provider="skills_sh",
                name="analyze-csv",
                source="office",
                summary="Analyze CSV files.",
                detail_url="https://skills.sh/office/analyze-csv",
                installable=True,
                quality={},
            ),
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills install skills_sh-1"
            await pilot.press("enter")
            await pilot.pause()

            assert service.installed_marketplace_ids == []
            assert "安装远端 skill" in _all_text(app)
            await pilot.press("y")
            await pilot.pause()

            conversation = _text(app, "#conversation")
            assert service.installed_marketplace_ids == ["skills_sh-1"]
            assert "已安装 marketplace skill：analyze-csv" in conversation
            assert "命令：$analyze-csv" in conversation
            assert "https://skills.sh/office/analyze-csv" in conversation

    asyncio.run(run_test())


def test_tui_skills_install_reports_marketplace_errors(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        marketplace_results=[
            SimpleNamespace(
                result_id="skillsmp-1",
                provider="skillsmp",
                name="csv-helper",
                source="data-team",
                summary="Find CSV workflows.",
                detail_url="https://skillsmp.com/skills/csv-helper",
                installable=False,
                quality={},
            ),
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills install skillsmp-1"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert service.installed_marketplace_ids == ["skillsmp-1"]
            assert "skills 操作失败" in _text(app, "#conversation")
            assert "only skills_sh results are installable" in _text(app, "#conversation")

            input_widget.value = "/skills wat"
            await pilot.press("enter")
            assert "/skills search <query>" in _text(app, "#conversation")
            assert "/skills install <result-id>" in _text(app, "#conversation")

    asyncio.run(run_test())


def test_tui_skill_command_starts_prompt_with_explicit_skill_context(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skill review check this"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.read_skill_names == ["review"]
            assert service.prompts
            assert service.prompts[0].startswith("Use skill review explicitly.")
            assert "Follow this workflow." in service.prompts[0]
            assert "check this" in service.prompts[0]

    asyncio.run(run_test())


def test_tui_skill_command_without_name_opens_skill_picker(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        skills=[
            {
                "name": "review",
                "description": "Review workflow.",
                "source": "user",
                "command_name": "review",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
            {
                "name": "csv-helper",
                "description": "CSV analysis workflow.",
                "source": "user",
                "command_name": "csv-helper",
                "user_invocable": True,
                "disable_model_invocation": False,
            },
        ],
    )


def _edit_diff_request() -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="edit_diff",
        tool_name="file_write",
        question="Approve file edit?",
        reason="file_write will modify notes.txt",
        risk_level="high",
        args_summary={
            "path": "notes.txt",
            "change_type": "modified",
            "additions": 1,
            "deletions": 1,
            "diff_preview": "--- notes.txt\n+++ notes.txt\n@@\n-old\n+new",
        },
    )

    async def run_test() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skill"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "选择 Skill" in _all_text(app)
            assert "2 skills" in _all_text(app)
            assert "review" in _all_text(app)
            assert "csv-helper" in _all_text(app)

            await pilot.press("c", "s", "v")
            await pilot.pause(0.1)
            assert "搜索: csv" in _all_text(app)
            assert "csv-helper" in _all_text(app)
            assert "review" not in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == "/skill csv-helper "
            assert service.prompts == []

    asyncio.run(run_test())


def test_status_line_shows_permission_mode(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path, permission_mode="auto_approve")
    status = service.get_workspace_status()

    assert "perm:auto" in status_line(status, ui_state="idle", width=120)


def test_untrusted_absolute_path_detection_ignores_authorized_roots(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    other = tmp_path / "other"
    project.mkdir()
    external.mkdir()
    other.mkdir()
    prompt = f'介绍 "{external}" 和 "{other}"'

    matches = find_untrusted_absolute_paths(
        prompt,
        project_root=project,
        external_roots=[{"path": str(external), "access": "read", "source": "user"}],
    )

    assert matches == [other.resolve()]


def test_tui_external_directory_read_decision_continues_prompt(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project")
    service.workspace_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = f'介绍 "{external}"'
            input_widget = app.query_one("#prompt-input")
            input_widget.value = prompt
            await pilot.press("enter")
            await pilot.press("enter")

            assert service.external_roots == [
                {"path": str(external.resolve()), "access": "read", "source": "user"},
            ]
            assert service.next_turn_target_paths == [str(external.resolve())]
            assert service.prompts == [prompt]

    asyncio.run(run())


def test_tui_permissions_command_shows_current_external_roots(tmp_path: Path) -> None:
    external = tmp_path / "external"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        external_roots=[{"path": str(external), "access": "full", "source": "user"}],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            modal_text = _all_text(app)
            assert "权限设置" in modal_text
            assert "请求批准" in modal_text
            assert "自动批准" in modal_text
            assert "完全访问权限" in modal_text
            assert "external" in modal_text
            assert "完全信任" in modal_text

    asyncio.run(run())


def test_tui_ctrl_p_opens_permissions_modal(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()

            assert "权限设置" in _all_text(app)

    asyncio.run(run())


def test_tui_permissions_modal_changes_permission_modes(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()

            assert service.permission_mode == "auto_approve"
            assert "perm:auto" in _text(app, "#status-bar")

            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()
            assert "完全访问权限" in _all_text(app)

            await pilot.press("y")
            await pilot.pause()
            assert service.permission_mode == "full_access"
            assert "perm:full" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_permissions_modal_changes_access_removes_and_clears_roots(tmp_path: Path) -> None:
    external_a = tmp_path / "external-a"
    external_b = tmp_path / "external-b"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        external_roots=[
            {"path": str(external_a), "access": "read", "source": "user"},
            {"path": str(external_b), "access": "full", "source": "user"},
        ],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("f")
            assert service.external_roots[0]["access"] == "full"

            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("r")
            assert service.external_roots == [{"path": str(external_a.resolve()), "access": "full", "source": "user"}]

            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("c")
            await pilot.press("y")
            assert service.external_roots == []

    asyncio.run(run())


def test_tui_full_access_mode_does_not_prompt_for_external_absolute_path(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project", permission_mode="full_access")
    service.workspace_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = f'介绍 "{external}"'
            input_widget = app.query_one("#prompt-input")
            input_widget.value = prompt
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert service.external_roots == []
            assert service.next_turn_target_paths == [str(external.resolve())]
            assert service.prompts == [prompt]
            assert "完全访问权限已启用" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_auto_approve_mode_still_prompts_for_untrusted_external_path(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project", permission_mode="auto_approve")
    service.workspace_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = f'介绍 "{external}"'
            input_widget = app.query_one("#prompt-input")
            input_widget.value = prompt
            await pilot.press("enter")
            await pilot.pause()

            assert "检测到工作区外目录" in _all_text(app)
            assert service.prompts == []

    asyncio.run(run())


def test_tui_wide_external_full_access_requires_confirmation(tmp_path: Path, monkeypatch) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project")
    service.workspace_root.mkdir()
    wide_root = tmp_path / "home"
    wide_root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: wide_root)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = f'整理 "{wide_root}"'
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("f")
            assert service.external_roots == []
            await pilot.press("y")
            assert service.external_roots == [{"path": str(wide_root.resolve()), "access": "full", "source": "user"}]
            assert service.prompts == [f'整理 "{wide_root}"']

    asyncio.run(run())


def test_tui_web_command_toggles_networking_inside_app(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert "web:off" in _text(app, "#status-bar")

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/web on"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is False
            assert "用法：/web" in _text(app, "#conversation")

            prompt_input.value = "/web"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is True
            assert "web:on" in _text(app, "#status-bar")
            assert "联网已开启" in _text(app, "#conversation")

            prompt_input.value = "/web"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.enable_web is False
            assert "web:off" in _text(app, "#status-bar")
            assert "联网已关闭" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_conversation_search_state_tracks_matches_and_navigation() -> None:
    state = ConversationSearchState(["You\n  inspect docs", "Tool file_read done", "Assistant\n  docs ready"])

    matched = state.update_query("docs")
    second = state.next_match()
    first = state.previous_match()
    empty = state.update_query("missing")

    assert matched.count == 2
    assert matched.current_line == 0
    assert second.current_line == 2
    assert first.current_line == 0
    assert empty.count == 0
    assert empty.status_text == "无匹配：missing"


def test_tui_session_overlay_state_filters_and_selects_sessions(tmp_path: Path) -> None:
    sessions = [
        _session_summary(tmp_path, "session-alpha", "整理会议纪要", 3),
        _session_summary(tmp_path, "session-beta", "分析 CSV", 1),
    ]
    state = SessionOverlayState(sessions=sessions)

    filtered = state.with_query("csv")
    selected = filtered.move(1)
    empty = filtered.with_query("none")

    assert [item.session_id for item in filtered.visible_sessions] == ["session-beta"]
    assert selected.selected_session.session_id == "session-beta"
    assert empty.selected_session is None
    assert "无匹配会话" in empty.render()


def test_tui_model_overlay_switches_session_and_sets_default(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            assert "router" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.switched_model_profile == "router"
            assert "模型已切换到当前会话：router" in _text(app, "#conversation")

            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("p")
            await pilot.pause(0.1)
            assert service.default_model_profile == "router"
            assert "默认模型 profile 已设为：router" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_model_overlay_deletes_profile_after_confirmation(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        ),
        SimpleNamespace(
            name="local",
            provider="openai-chat",
            model="deepseek-chat",
            active=True,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        ),
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("d")
            await pilot.pause(0.1)
            assert "删除模型 profile：router" in _all_text(app)

            await pilot.press("n")
            await pilot.pause(0.1)
            assert service.deleted_model_profile is None

            await pilot.press("d")
            await pilot.pause(0.1)
            await pilot.press("y")
            await pilot.pause(0.1)

            assert service.deleted_model_profile == "router"
            text = _all_text(app)
            assert "模型 profile 已删除：router" in text
            assert "router" not in service.model_profiles
            assert "local" in text

    asyncio.run(run())


def test_tui_model_setup_wizard_masks_key_and_saves_keyring_profile(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "t", "e", "s", "t", "-", "s", "e", "c", "r", "e", "t")
            await pilot.pause(0.1)
            assert "sk-test-secret" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.configured_model_profile.name == "requesty-openai-gpt-5-2-chat"
            assert service.configured_api_key == "sk-test-secret"

    asyncio.run(run())


def test_tui_manual_model_setup_can_save_keyring_profile(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("m")
            await pilot.pause(0.1)
            assert "手动模型配置" in _all_text(app)

            for value in [
                "deepseek",
                "openai-chat",
                "https://api.deepseek.com",
                "deepseek-chat",
                "DEEPSEEK_API_KEY",
                "keyring",
                "sk-manual-secret",
            ]:
                await pilot.press(*list(value), "enter")
                await pilot.pause(0.05)

            request = service.configured_model_profile
            assert request.name == "deepseek"
            assert request.provider == "openai-chat"
            assert request.base_url == "https://api.deepseek.com"
            assert request.model == "deepseek-chat"
            assert request.api_key_env == "DEEPSEEK_API_KEY"
            assert request.credential_source == "keyring"
            assert service.configured_api_key == "sk-manual-secret"
            assert "sk-manual-secret" not in _all_text(app)

    asyncio.run(run())


def test_tui_manual_model_setup_env_source_does_not_require_api_key(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("m")
            await pilot.pause(0.1)

            for value in [
                "local-env",
                "openai",
                "https://api.openai.com/v1",
                "gpt-5.2",
                "OPENAI_API_KEY",
                "env",
            ]:
                await pilot.press(*list(value), "enter")
                await pilot.pause(0.05)

            request = service.configured_model_profile
            assert request.name == "local-env"
            assert request.provider == "openai"
            assert request.credential_source == "env"
            assert service.configured_api_key is None

    asyncio.run(run())


def test_tui_manual_model_setup_insecure_file_requires_explicit_confirmation(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("m")
            await pilot.pause(0.1)

            for value in [
                "plain",
                "openai-chat",
                "https://api.example/v1",
                "plain-model",
                "PLAIN_API_KEY",
                "insecure_file",
            ]:
                await pilot.press(*list(value), "enter")
                await pilot.pause(0.05)

            assert "必须输入 YES 才会继续" in _all_text(app)
            await pilot.press("n", "o", "enter")
            await pilot.pause(0.05)
            assert service.configured_model_profile is None

            await pilot.press("Y", "E", "S", "enter")
            await pilot.pause(0.05)
            await pilot.press("s", "k", "-", "p", "l", "a", "i", "n", "enter")
            await pilot.pause(0.1)

            request = service.configured_model_profile
            assert request.name == "plain"
            assert request.credential_source == "insecure_file"
            assert service.configured_api_key == "sk-plain"

    asyncio.run(run())


def test_tui_model_new_profile_keeps_model_center_visible_while_catalog_loads(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_refresh_release = threading.Event()
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            text = _all_text(app)
            assert isinstance(app.screen, ModelCatalogLoadingOverlay)
            assert "模型中心" in text
            assert "正在读取模型目录" in text

            service.catalog_refresh_release.set()
            await pilot.pause(0.2)
            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 0
            assert "provider: Requesty" in _all_text(app)

    asyncio.run(run())


def test_tui_model_new_profile_reuses_in_memory_catalog(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            await pilot.press("escape")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 0

    asyncio.run(run())


def test_tui_model_new_profile_refreshes_once_when_cached_catalog_has_no_configurable_models(
    tmp_path: Path,
) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.cached_catalog_providers = [
        SimpleNamespace(
            id="fresh",
            name="Fresh",
            env_names=["FRESH_API_KEY"],
            api_base_url="https://fresh.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[],
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.3)

            assert service.got_catalog_count == 1
            assert service.refreshed_catalog_count == 1
            text = _all_text(app)
            assert "provider: Requesty" in text
            assert "模型目录没有可配置模型" not in text

    asyncio.run(run())


def test_tui_model_new_profile_reports_empty_catalog_instead_of_opening_empty_wizard(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            text = _all_text(app)
            assert "没有可配置的目录模型" not in text
            assert "模型目录没有可配置模型" in text
            assert "请刷新目录或检查网络" in text

    asyncio.run(run())


def test_tui_model_setup_wizard_filters_out_adapter_required_catalog_providers(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="mistral",
            name="Mistral",
            env_names=["MISTRAL_API_KEY"],
            api_base_url="https://api.mistral.ai",
            provider_package="@ai-sdk/mistral",
            models=[SimpleNamespace(id="mistral-large-latest", name="Mistral Large")],
        ),
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        ),
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            text = _all_text(app)
            assert "Mistral" not in text
            assert "Requesty" in text

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "t", "e", "s", "t")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.provider == "openai-chat"
            assert service.configured_model_profile.base_url == "https://router.requesty.ai/v1"
            assert service.configured_model_profile.model == "openai/gpt-5.2-chat"

    asyncio.run(run())


def test_tui_model_setup_wizard_can_configure_native_anthropic_provider(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="anthropic",
            name="Anthropic",
            env_names=["ANTHROPIC_API_KEY"],
            api_base_url="https://api.anthropic.com",
            provider_package="@ai-sdk/anthropic",
            models=[SimpleNamespace(id="claude-sonnet-4-5", name="Claude Sonnet 4.5")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            assert "provider: Anthropic" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "a", "n", "t")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.name == "anthropic-claude-sonnet-4-5"
            assert service.configured_model_profile.provider == "anthropic"
            assert service.configured_model_profile.base_url == "https://api.anthropic.com"
            assert service.configured_model_profile.model == "claude-sonnet-4-5"
            assert service.configured_model_profile.api_key_env == "ANTHROPIC_API_KEY"

    asyncio.run(run())


def test_tui_model_setup_wizard_can_configure_native_google_provider(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="google",
            name="Google",
            env_names=["GEMINI_API_KEY"],
            api_base_url="https://generativelanguage.googleapis.com/v1beta",
            provider_package="@ai-sdk/google",
            models=[SimpleNamespace(id="gemini-2.5-pro", name="Gemini 2.5 Pro")],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            assert "provider: Google" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("g", "e", "m", "i", "n", "i", "-", "t", "e", "s", "t")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.name == "google-gemini-2-5-pro"
            assert service.configured_model_profile.provider == "google"
            assert service.configured_model_profile.base_url == "https://generativelanguage.googleapis.com/v1beta"
            assert service.configured_model_profile.model == "gemini-2.5-pro"
            assert service.configured_model_profile.api_key_env == "GEMINI_API_KEY"

    asyncio.run(run())


def test_tui_model_setup_wizard_can_select_provider_and_model(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[
                SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat"),
                SimpleNamespace(id="openai/gpt-5.2-coder", name="GPT 5.2 Coder"),
            ],
        ),
        SimpleNamespace(
            id="deepseek",
            name="DeepSeek",
            env_names=["DEEPSEEK_API_KEY"],
            api_base_url="https://api.deepseek.com",
            provider_package="@ai-sdk/openai-compatible",
            models=[
                SimpleNamespace(id="deepseek-chat", name="DeepSeek Chat"),
                SimpleNamespace(id="deepseek-reasoner", name="DeepSeek Reasoner"),
            ],
        ),
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            assert "provider: Requesty" in _all_text(app)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "provider: DeepSeek" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "model: DeepSeek Chat" in _all_text(app)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "model: DeepSeek Reasoner" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "d", "s")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.name == "deepseek-deepseek-reasoner"
            assert service.configured_model_profile.provider == "openai-chat"
            assert service.configured_model_profile.base_url == "https://api.deepseek.com"
            assert service.configured_model_profile.model == "deepseek-reasoner"
            assert service.configured_model_profile.api_key_env == "DEEPSEEK_API_KEY"

    asyncio.run(run())


def test_tui_model_setup_wizard_filters_provider_and_model_lists(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id="openai/gpt-5.2-chat", name="GPT 5.2 Chat")],
        ),
        SimpleNamespace(
            id="deepseek",
            name="DeepSeek",
            env_names=["DEEPSEEK_API_KEY"],
            api_base_url="https://api.deepseek.com",
            provider_package="@ai-sdk/openai-compatible",
            models=[
                SimpleNamespace(id="deepseek-chat", name="DeepSeek Chat"),
                SimpleNamespace(id="deepseek-reasoner", name="DeepSeek Reasoner"),
                SimpleNamespace(id="deepseek-coder", name="DeepSeek Coder"),
            ],
        ),
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            await pilot.press("d", "e", "e", "p")
            await pilot.pause(0.1)
            text = _all_text(app)
            assert "provider 搜索: deep" in text
            assert "DeepSeek" in text
            assert "Requesty" not in text

            await pilot.press("enter")
            await pilot.pause(0.1)
            text = _all_text(app)
            assert "DeepSeek Chat" in text
            assert "DeepSeek Reasoner" in text
            assert "DeepSeek Coder" in text
            assert "模型 1/3" in text

            await pilot.press("r", "e", "a")
            await pilot.pause(0.1)
            text = _all_text(app)
            assert "model 搜索: rea" in text
            assert "DeepSeek Reasoner" in text
            assert "DeepSeek Chat" not in text
            assert "DeepSeek Coder" not in text

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "d", "s")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.name == "deepseek-deepseek-reasoner"
            assert service.configured_model_profile.model == "deepseek-reasoner"

    asyncio.run(run())


def test_tui_model_setup_wizard_scrolls_provider_window_with_selection(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id=f"provider-{index:02d}",
            name=f"Provider {index:02d}",
            env_names=[f"PROVIDER_{index:02d}_API_KEY"],
            api_base_url=f"https://provider-{index:02d}.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[SimpleNamespace(id=f"model-{index:02d}", name=f"Model {index:02d}")],
        )
        for index in range(15)
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            for _ in range(13):
                await pilot.press("down")
            await pilot.pause(0.1)

            text = _all_text(app)
            assert "Provider 14/15" in text
            assert "> Provider 13" in text
            assert "Provider 00" not in text

    asyncio.run(run())


def test_tui_model_setup_wizard_scrolls_model_window_with_selection(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="many",
            name="Many Models",
            env_names=["MANY_API_KEY"],
            api_base_url="https://many.example/v1",
            provider_package="@ai-sdk/openai-compatible",
            models=[
                SimpleNamespace(id=f"model-{index:02d}", name=f"Model {index:02d}")
                for index in range(20)
            ],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)
            await pilot.press("enter")
            await pilot.pause(0.1)
            for _ in range(17):
                await pilot.press("down")
            await pilot.pause(0.1)

            text = _all_text(app)
            assert "模型 18/20" in text
            assert "> Model 17" in text
            assert "Model 00" not in text

    asyncio.run(run())


def test_tui_model_setup_wizard_supports_official_openai_catalog_provider(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.catalog_providers = [
        SimpleNamespace(
            id="openai",
            name="OpenAI",
            env_names=["OPENAI_API_KEY"],
            api_base_url="https://api.openai.com/v1",
            provider_package=None,
            models=[
                SimpleNamespace(id="gpt-5.2-chat", name="GPT 5.2 Chat"),
                SimpleNamespace(id="gpt-5.2-coder", name="GPT 5.2 Coder"),
            ],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.2)

            assert "provider: OpenAI" in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "model: GPT 5.2 Coder" in _all_text(app)

            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("s", "k", "-", "o", "a", "i")
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.configured_model_profile.name == "openai-gpt-5-2-coder"
            assert service.configured_model_profile.provider == "openai"
            assert service.configured_model_profile.base_url == "https://api.openai.com/v1"
            assert service.configured_model_profile.model == "gpt-5.2-coder"
            assert service.configured_model_profile.api_key_env == "OPENAI_API_KEY"

    asyncio.run(run())


def test_tui_model_overlay_runs_connection_test_without_showing_secret(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.connection_test_result = SimpleNamespace(
        ok=True,
        profile_name="router",
        provider="openai-chat",
        model="openai/gpt-5.2-chat",
        message="OK",
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("t")
            await pilot.pause(0.2)
            assert service.tested_model_profile == "router"
            assert "OK" in _all_text(app)
            assert "sk-test-secret" not in _all_text(app)

    asyncio.run(run())


def test_tui_model_overlay_refreshes_catalog_via_worker(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)
    service.model_profiles = [
        SimpleNamespace(
            name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            active=False,
            credential_available=True,
            capability=SimpleNamespace(status="runnable"),
        )
    ]
    service.catalog_providers = [
        SimpleNamespace(
            id="requesty",
            name="Requesty",
            env_names=["REQUESTY_API_KEY"],
            api_base_url="https://router.requesty.ai/v1",
            models=[],
        )
    ]

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("/")
            await pilot.press("m", "o", "d", "e", "l", "enter")
            await pilot.pause(0.1)
            await pilot.press("r")
            await pilot.pause(0.2)
            assert service.refreshed_catalog_count >= 1
            assert "模型目录已刷新：1 个 provider" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_file_reference_fuzzy_search_stays_inside_workspace(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    target = docs / "Project Plan.md"
    target.write_text("plan", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    outside = tmp_path.parent / "outside-haagent-ref.txt"
    outside.write_text("outside", encoding="utf-8")

    matches = fuzzy_file_matches(tmp_path, "plan")
    no_matches = fuzzy_file_matches(tmp_path, "missing")
    token = path_reference_token(tmp_path, target)

    assert [match.display_path for match in matches] == ["docs/Project Plan.md"]
    assert no_matches == []
    assert token == '@file("docs/Project Plan.md")'


def test_tui_file_reference_index_uses_fast_file_walker(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Project Plan.md").write_text("plan", encoding="utf-8")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hi')", encoding="utf-8")
    candidates = [
        FileReferenceMatch(path=docs / "Project Plan.md", display_path="docs/Project Plan.md"),
        FileReferenceMatch(path=src / "main.py", display_path="src/main.py"),
    ]

    monkeypatch.setattr("haagent.tui.files.refs._iter_file_reference_candidates", lambda root: iter(candidates))

    index = build_file_reference_index(tmp_path)
    assert [item.display_path for item in index.matches("plan")] == ["docs/Project Plan.md"]
    assert [item.display_path for item in index.matches("main")] == ["src/main.py"]


def test_tui_file_reference_overlay_scrolls_selected_match_into_view(tmp_path: Path) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    overlay = FileReferenceOverlay(tmp_path, "")
    overlay.index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=tuple(
            FileReferenceMatch(path=tmp_path / f"file-{index:02}.txt", display_path=f"file-{index:02}.txt")
            for index in range(12)
        ),
    )
    overlay.loading = False
    overlay._reload()

    for _ in range(4):
        overlay._move(1)

    body = overlay._body()
    assert "> file-04.txt" in body
    assert "  file-00.txt" not in body
    assert "  file-03.txt" in body


def test_tui_file_reference_overlay_uses_preloaded_index_without_loading(tmp_path: Path) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=(FileReferenceMatch(path=tmp_path / "README.md", display_path="README.md"),),
    )
    overlay = FileReferenceOverlay(tmp_path, "", index)
    overlay.on_mount()

    assert overlay.loading is False
    assert "正在搜索文件" not in overlay._body()
    assert "README.md" in overlay._body()


def test_tui_file_reference_overlay_filters_loaded_index_without_rescanning(tmp_path: Path, monkeypatch) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    def fail_rglob(self, pattern):
        raise AssertionError("query updates should not rescan workspace")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    overlay = FileReferenceOverlay(tmp_path, "")
    overlay.index = FileReferenceIndex(
        root=tmp_path.resolve(),
        files=(
            FileReferenceMatch(path=tmp_path / "README.md", display_path="README.md"),
            FileReferenceMatch(path=tmp_path / "docs" / "Project Plan.md", display_path="docs/Project Plan.md"),
        ),
    )

    overlay.update_query("plan")

    assert [item.display_path for item in overlay.matches] == ["docs/Project Plan.md"]


def test_tui_file_reference_overlay_ignores_index_after_unmount(tmp_path: Path, monkeypatch) -> None:
    from haagent.tui.files.overlay import FileReferenceOverlay

    overlay = FileReferenceOverlay(tmp_path, "")
    index = build_file_reference_index(tmp_path)
    overlay.on_unmount()

    def fail_reload():
        raise AssertionError("unmounted overlay should not refresh stale worker results")

    monkeypatch.setattr(overlay, "_reload", fail_reload)

    overlay._handle_index_ready(index)

    assert overlay.index is None


def test_tui_failure_next_steps_are_conservative() -> None:
    steps = failure_next_steps(
        failed_stage="executing",
        failure_category="Tool Argument Failure",
        reason="path does not exist",
        episode_path=".runs/episode-failed",
    )
    joined = "\n".join(steps)

    assert "重试" in joined
    assert "调整请求" in joined
    assert ".runs/episode-failed" in joined
    assert "/tools" not in joined
    assert "工具详情" not in joined
    assert "工具时间线" not in joined
    assert "修复完成" not in joined
    assert "一定是" not in joined


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
            conversation = _text(app, "#conversation")
            assert "ws:" in status
            assert str(tmp_path) not in status
            assert "profile: local" in status
            assert "openai-chat/deepseek-chat" in status
            assert "key: ok" in status
            assert "DEEPSEEK_API_KEY" not in status
            assert "session-test" in status
            assert list(app.query("#side-bar")) == []
            assert "Shift+Enter 换行" in conversation
            footer = _text(app, "#footer-bar")
            assert "[Ctrl+Q]退出" in footer
            assert "[q]退出" not in footer
            assert "[Enter]发送" in str(app.query_one("#footer-bar").render())
            assert "[Shift+Enter]换行" in str(app.query_one("#footer-bar").render())
            assert isinstance(app.query_one("#prompt-input"), TextArea)

    asyncio.run(run())


def test_tui_default_theme_applies_semantic_classes_and_chinese_titles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status_widget = app.query_one("#status-bar")

            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            assert status_widget.has_class("status-default")
            assert "状态: - 空闲" in _text(app, "#status-bar")
            assert list(app.query("#side-bar")) == []
            conversation = _text(app, "#conversation")
            assert "Shift+Enter 换行" in conversation

    asyncio.run(run())


def test_tui_light_theme_can_be_enabled_without_losing_status_classes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("HAAGENT_TUI_THEME", "light")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert app.theme == "haagent-light"
            assert app.screen.has_class("theme-light")
            assert app.query_one("#status-bar").has_class("status-default")
            assert "状态: - 空闲" in _text(app, "#status-bar")
            assert list(app.query("#side-bar")) == []

    asyncio.run(run())


def test_tui_theme_can_be_cycled_with_keyboard_shortcut(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("HAAGENT_TUI_THEME", raising=False)

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-light"
            assert app.screen.has_class("theme-light")
            assert "主题已切换：浅色" in _text(app, "#conversation")

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert "主题已切换：单色" in _text(app, "#conversation")

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            assert "主题已切换：暗色" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_no_color_prevents_keyboard_theme_cycle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-monochrome"

            await pilot.press("ctrl+t")
            await pilot.pause(0.1)
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert "NO_COLOR 已启用，主题保持单色" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_no_color_mode_keeps_symbols_text_and_selection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    candidates = [_memory_candidate("cand_first", "第一条"), _memory_candidate("cand_second", "第二条")]

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=candidates)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.theme == "haagent-monochrome"
            assert app.screen.has_class("theme-monochrome")
            assert "状态: - 空闲" in _text(app, "#status-bar")

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")

            assert list(app.query("#side-bar")) == []
            assert "记忆候选" in conversation
            assert "> cand_second" in conversation
            assert "确认" in _text(app, "#footer-bar")

    asyncio.run(run())


def test_tui_m_key_no_longer_opens_memory_mode(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)

            assert "记忆候选" not in _text(app, "#conversation")
            assert list(app.query("#side-bar")) == []

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "记忆候选" in _text(app, "#conversation")

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
            assert list(app.query("#side-bar")) == []

    async def run_120() -> None:
        service = FakeAssistantService(
            workspace_root=long_workspace,
            model=long_model,
            current_session_id=long_session,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            status = _text(app, "#status-bar")
            assert len(status) <= 120
            assert "ws:" in status
            assert "profile: local" in status
            assert "key: ok" in status
            assert "sid:" in status
            assert str(long_workspace) not in status
            assert long_model not in status
            assert long_session not in status
            assert "DEEPSEEK_API_KEY" not in status
            assert list(app.query("#side-bar")) == []

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
            assert list(app.query("#side-bar")) == []
            assert "[Ctrl+Q]退出" in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert app.query_one("#resize-message").has_class("hidden")
            assert not app.query_one("#main").has_class("hidden")
            assert list(app.query("#side-bar")) == []
            assert "Shift+Enter 换行" in _text(app, "#conversation")

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
            profile_error="未找到默认模型配置，请运行 haagent 后在 TUI 内输入 /model 完成配置",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = _text(app, "#conversation")
            assert "未找到默认模型配置" in conversation
            assert "/model" in conversation
            assert "uv run haagent setup" not in conversation

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
            assert "/model" in conversation
            assert "不会在 TUI 中输入" not in conversation

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
            conversation = _text(app, "#conversation")
            assert "key: ok" in status
            assert "Shift+Enter 换行" in conversation

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
            assert "系统凭据库不可用：backend unavailable" in conversation
            assert "/model" in conversation
            assert "uv run haagent setup" not in conversation

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


def test_tui_plain_s_does_not_open_sessions_or_search(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            assert input_widget.has_focus
            await pilot.press("s")
            await pilot.pause(0.1)
            assert input_widget.value == "s"
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)
            assert "范围: conversation" not in _all_text(app)

    asyncio.run(run())


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
            assert "HaAgent 帮助" in rendered
            assert "聊天模式" in rendered
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "HaAgent 帮助" not in _all_text(app)

    asyncio.run(run())


def test_tui_help_modal_is_contextual_for_memory_modes(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("?")
            await pilot.pause(0.1)
            assert "记忆候选列表" in _all_text(app)
            assert "↑/↓" in _all_text(app)
            assert "j/k" not in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)

            app.action_memory_enter()
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


def test_tui_text_area_shift_enter_inserts_newline_and_enter_submits_prompt(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Summarize this folder"
            await pilot.press("shift+enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert input_widget.value == "Summarize this folder\n"
            input_widget.value = "Summarize this folder\nwith constraints"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts == ["Summarize this folder\nwith constraints"]
            assert input_widget.value == ""
            conversation = _text(app, "#conversation")
            assert "你" in conversation
            assert "Summarize this folder" in conversation
            assert "with constraints" in conversation
            assert "assistant: Summarize this folder\nwith constraints" in conversation

    asyncio.run(run())


def test_tui_text_area_blank_input_does_not_submit(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "  \n  "
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert input_widget.value == "  \n  "

    asyncio.run(run())


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


def test_tui_sessions_overlay_search_resume_continue_new_and_escape(tmp_path: Path) -> None:
    sessions = [
        _session_summary(tmp_path, "session-alpha", "整理会议纪要", 3),
        _session_summary(tmp_path, "session-beta", "分析 CSV", 1),
    ]

    async def run_resume() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            sessions=sessions,
            session_histories={
                "session-beta": [
                    SimpleNamespace(
                        turn_index=1,
                        request="分析 CSV",
                        summary="用户要分析 sales.csv，助手已说明会检查列名和异常值。",
                        status="completed",
                    ),
                ],
            },
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" in _all_text(app)
            await pilot.press("c", "s", "v")
            await pilot.pause(0.1)
            assert "session-beta" in _all_text(app)
            assert "session-alpha" not in str(app.screen.render())
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.resumed_sessions == [str(sessions[1].session_path)]
            assert "session-beta" in _text(app, "#status-bar")
            conversation = _text(app, "#conversation")
            assert "分析 CSV" in conversation
            assert "用户要分析 sales.csv" in conversation
            assert "当前会话：session-beta" not in conversation
            assert "整理会议纪要" not in conversation
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)

    async def run_continue_new_escape() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, sessions=sessions)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("l")
            await pilot.pause(0.1)
            assert service.continued_latest_count == 1
            assert service.current_session_id == "session-alpha"

            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("n")
            await pilot.pause(0.1)
            assert service.created_sessions == ["session-new-1"]
            assert "当前会话：session-new-1" not in _text(app, "#conversation")
            assert "sid:session-n" in _text(app, "#status-bar")

            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)

    asyncio.run(run_resume())
    asyncio.run(run_continue_new_escape())


def test_tui_new_session_command_clears_previous_timeline(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path, assistant_content="旧回答")

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "旧问题"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "旧问题" in _text(app, "#conversation")
            assert "旧回答" in _text(app, "#conversation")

            input_widget.value = "/new"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert service.created_sessions == ["session-new-1"]
            assert "旧问题" not in conversation
            assert "旧回答" not in conversation
            assert "新建会话：session-new-1" not in conversation
            assert "sid:session-n" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_mcp_command_renders_server_status(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        mcp_status={
            "configured_count": 2,
            "connected_count": 1,
            "failed_count": 1,
            "servers": [
                {
                    "name": "fixture",
                    "state": "connected",
                    "detail": "",
                    "tool_count": 1,
                    "resource_count": 1,
                },
                {
                    "name": "broken",
                    "state": "failed",
                    "detail": "connection refused",
                    "tool_count": 0,
                    "resource_count": 0,
                },
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/mcp"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "MCP servers:" in conversation
            assert "fixture: connected (tools: 1, resources: 1)" in conversation
            assert "broken: failed - connection refused" in conversation

    asyncio.run(run())


def test_tui_mcp_command_renders_configured_not_loaded_without_session(tmp_path: Path) -> None:
    service = FakeAssistantService(
        workspace_root=tmp_path,
        mcp_status={
            "configured_count": 1,
            "connected_count": 0,
            "failed_count": 0,
            "servers": [
                {
                    "name": "exa",
                    "state": "configured",
                    "detail": "not loaded; create or resume a session to connect",
                    "tool_count": 0,
                    "resource_count": 0,
                }
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/mcp"
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "MCP servers:" in conversation
            assert "exa: configured - not loaded; create or resume a session to connect" in conversation

    asyncio.run(run())


def test_tui_restored_session_renders_final_response_not_raw_turn_summary(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-raw", "恢复旧会话", 1)]
    raw_summary = "\n".join(
        [
            "- user_request: 恢复旧会话",
            "  status: completed",
            f"  episode_path: {tmp_path / '.runs' / 'episode-1'}",
            "  assistant_final_response: 这是恢复后应该看到的回答。",
            "  verification: success",
        ],
    )
    service = FakeAssistantService(
        workspace_root=tmp_path,
        sessions=sessions,
        session_histories={
            "session-raw": [
                SimpleNamespace(
                    turn_index=1,
                    request="恢复旧会话",
                    summary=raw_summary,
                    status="completed",
                ),
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "恢复旧会话" in conversation
            assert "这是恢复后应该看到的回答。" in conversation
            assert "user_request:" not in conversation
            assert "episode_path:" not in conversation
            assert "verification:" not in conversation

    asyncio.run(run())


def test_tui_restored_session_prefers_assistant_display_text(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-display", "恢复长回答", 1)]
    raw_summary = "\n".join(
        [
            "- user_request: 恢复长回答",
            "  status: completed",
            f"  episode_path: {tmp_path / '.runs' / 'episode-1'}",
            "  assistant_final_response: 摘要里的短回答... [truncated]",
            "  verification: success",
        ],
    )
    full_display_text = "这是用于恢复展示的较完整回答，不应该退回到摘要里的截断文本。"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        sessions=sessions,
        session_histories={
            "session-display": [
                SimpleNamespace(
                    turn_index=1,
                    request="恢复长回答",
                    summary=raw_summary,
                    status="completed",
                    assistant_display_text=full_display_text,
                ),
            ],
        },
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert full_display_text in conversation
            assert "摘要里的短回答" not in conversation

    asyncio.run(run())


def test_tui_restores_initial_resume_session_on_mount(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        service.initial_resume = "session-from-cli"
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert service.resumed_sessions == ["session-from-cli"]
            assert service.current_session_id == "session-from-cli"
            assert "sid:session-f" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_continues_initial_latest_session_on_mount(tmp_path: Path) -> None:
    sessions = [_session_summary(tmp_path, "session-alpha", "整理会议纪要", 3)]

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, sessions=sessions)
        service.initial_continue = True
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert service.continued_latest_count == 1
            assert service.current_session_id == "session-alpha"
            assert "sid:session-a" in _text(app, "#status-bar")

    asyncio.run(run())


def test_tui_search_overlay_finds_conversation_and_does_not_pollute_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            app._append_block("Assistant", "Alpha docs\nBeta docs")
            app._append_line("Tool file_read done")
            app._refresh_conversation()
            await pilot.pause()
            before = _text(app, "#conversation")

            await pilot.press("ctrl+f")
            await pilot.pause(0.1)
            await pilot.press("d", "o", "c", "s")
            await pilot.pause(0.1)
            assert "范围: conversation" in _all_text(app)
            assert "1/2" in _all_text(app)
            await pilot.press("n")
            await pilot.pause(0.1)
            assert "2/2" in _all_text(app)
            await pilot.press("shift+n")
            await pilot.pause(0.1)
            assert "1/2" in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert _text(app, "#conversation") == before

            await pilot.press("ctrl+f")
            await pilot.press("x", "x", "x")
            await pilot.pause(0.1)
            assert "无匹配" in _all_text(app)

    asyncio.run(run())


def test_tui_slash_command_suggestions_filter_execute_and_do_not_pollute_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, sessions=[_session_summary(tmp_path, "session-old", "继续任务")])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            before = _text(app, "#conversation")

            await pilot.press("/")
            await pilot.pause(0.1)
            assert "快捷命令" in _all_text(app)
            assert "/help" in _all_text(app)
            assert "/memory" in _all_text(app)
            assert "/resume" not in _all_text(app)
            assert app.command_suggestions_is_open()
            assert app.query_one("#command-suggestions-dialog").parent.id == "input-panel"
            assert app.query_one("#prompt-input").value == "/"
            await pilot.press("h", "e")
            await pilot.pause(0.1)
            assert "过滤: /he" in _all_text(app)
            assert "/help" in _all_text(app)
            assert "/resume" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "HaAgent 帮助" in _all_text(app)
            assert _text(app, "#conversation") == before
            await pilot.press("escape")
            await pilot.pause(0.1)

            await pilot.press("/")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" in _all_text(app)
            assert service.prompts == []
            await pilot.press("escape")
            await pilot.pause(0.1)

            input_widget = app.query_one("#prompt-input")
            input_widget.value = "整理 /tmp"
            await pilot.press("/")
            await pilot.pause(0.1)
            assert not app.command_suggestions_is_open()
            assert app.query_one("#prompt-input").value == "整理 /tmp/"

            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/new"
            await pilot.press("enter")
            await pilot.pause(0.1)
            input_widget.value = "/resume"
            await pilot.press("enter")
            await pilot.pause(0.1)

            input_widget.value = "/unknown"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert service.prompts == []
            assert service.created_sessions == ["session-new-1"]
            assert service.continued_latest_count == 1
            assert "未知命令：/unknown" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_slash_command_suggestions_scroll_visible_window() -> None:
    from haagent.tui.commands.suggestions import CommandSuggestionState

    state = CommandSuggestionState(commands=command_registry().commands())
    for _ in range(8):
        state = state.move(1)

    rendered = state.render()

    assert "/web" in rendered
    assert "/help" not in rendered
    assert "/skill" in rendered
    assert "> /web" in rendered


def test_tui_file_reference_overlay_selects_workspace_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Project Plan.md").write_text("plan", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")

    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "Read "
            await pilot.press("@")
            await pilot.pause(0.1)
            assert input_widget.value == "Read @"
            assert type(app.screen) is Screen
            assert app.query_one("#prompt-input", PromptInput) is input_widget
            assert "文件引用" in _all_text(app)
            assert "docs/Project Plan.md" in _all_text(app)
            assert "README.md" in _all_text(app)
            assert "> README.md" in _all_text(app)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> README.md" not in _all_text(app)
            await pilot.press("p", "l", "a")
            await pilot.pause(0.1)
            assert "搜索: pla" in _all_text(app)
            assert "README.md" not in _all_text(app)
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert input_widget.value == 'Read @file("docs/Project Plan.md")'

            input_widget.value = "Read @missing"
            await pilot.press("@")
            await pilot.pause(0.1)
            assert "无匹配文件" in _all_text(app)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "文件引用" not in _all_text(app)
            assert app.query_one("#prompt-input", PromptInput) is input_widget
            assert input_widget.has_focus

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
            assert "工具 1 项" in conversation
            assert "1 待确认" in conversation
            assert "shell" in conversation

    asyncio.run(run())
def test_tui_tool_events_and_failure_stay_visible_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            interaction_request=_approval_request({"command": "uv run pytest -q", "cwd": ".", "timeout_seconds": 30}),
            extra_events=[
                _runtime_event(
                    "tool_started",
                    1,
                    tool_name="file_write",
                    args={"path": "notes.md", "mode": "create"},
                ),
                _runtime_event(
                    "tool_finished",
                    1,
                    tool_name="file_write",
                    result={"status": "success", "path": str(tmp_path / "notes.md"), "mode": "create", "bytes_written": 12, "created": True},
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "写一个 notes"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("n")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert list(app.query("#side-bar")) == []
            assert "工具 2 项" in conversation
            assert "1 成功" in conversation
            assert "1 失败" in conversation
            assert "file_write" in conversation
            assert "shell" in conversation
            assert "审批已拒绝：shell" in conversation
            assert "查看工具详情" not in conversation
            assert "任务工作台" not in conversation
            assert "工具时间线" not in conversation

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


def test_tui_running_task_rejects_plain_submit_and_keeps_input(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            input_widget.value = "Second task"
            await pilot.press("enter")
            await pilot.pause(0.2)

            try:
                assert service.prompts == ["Long task"]
                assert input_widget.value == "Second task"
                conversation = _text(app, "#conversation")
                assert "当前任务仍在运行，请等待或使用 /cancel" not in conversation
                assert "[命令]" not in conversation
                assert "Second task" not in conversation
            finally:
                service.release.set()
                await pilot.pause(0.2)

    asyncio.run(run())


def test_tui_layout_sizes_do_not_render_side_bar(tmp_path: Path) -> None:
    async def run_80() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            assert list(app.query("#side-bar")) == []
            assert "/tools" not in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            assert list(app.query("#side-bar")) == []
            assert not app.query_one("#main").has_class("hidden")

    async def run_200() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(200, 60)):
            assert list(app.query("#side-bar")) == []
            assert "Shift+Enter 换行" in _text(app, "#conversation")

    asyncio.run(run_80())
    asyncio.run(run_120())
    asyncio.run(run_200())


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
            assert "回答已提交：request_user_input" in conversation
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
            assert "工具 1 项" in conversation
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


def test_tui_memory_candidate_event_shows_notice(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[_memory_candidate()],
            extra_events=[
                MemoryNoticeEvent(
                    session_id="session-test",
                    turn_index=1,
                    count=1,
                    message="发现 1 条可记忆候选，已放入候选队列，等待你确认。",
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
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation
            assert "[a/y]确认" in _text(app, "#footer-bar")

    asyncio.run(run())


def test_tui_memory_panel_lists_and_shows_candidate_details(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            conversation = _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation
            assert "用户身份与爱好" in conversation
            assert "[Enter]详情" in footer

            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "source_summary: 用户明确要求记住自己的名字和爱好。" in conversation
            assert "basis: 用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in conversation
            assert "category_rationale: 这是跨 workspace 可复用的用户偏好和身份信息。" in conversation

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
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "> cand_second" in conversation
            assert "  cand_first" in conversation
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
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")
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
            await _open_memory_panel(app, pilot)
            await pilot.press("G")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#conversation")

            await pilot.press("enter")
            await pilot.pause(0.1)
            assert "candidate_id: cand_last" in _text(app, "#conversation")
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "> cand_last" in _text(app, "#conversation")

            await pilot.press("g")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "[↑/↓]移动" in footer
            assert "j/k" not in footer
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
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_third" in _text(app, "#conversation")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_j_k_do_not_move_memory_selection(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            memory_candidates=[
                _memory_candidate("cand_first", "第一条偏好"),
                _memory_candidate("cand_second", "第二条偏好"),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("j")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#conversation")
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")
            await pilot.press("k")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_memory_mode_is_readable_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(90, 30)) as pilot:
            await _open_memory_panel(app, pilot)
            conversation = _text(app, "#conversation")
            assert "记忆候选" in conversation
            assert "cand_abc123" in conversation

            await pilot.press("enter")
            await pilot.pause(0.1)
            conversation = _text(app, "#conversation")
            assert "记忆候选详情" in conversation
            assert "basis: 用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。" in conversation

    asyncio.run(run())


def test_tui_memory_confirm_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("a")
            await pilot.pause(0.1)
            assert service.confirmed_candidate_ids == ["cand_abc123"]
            assert "已确认记忆候选：cand_abc123" in _text(app, "#conversation")
            assert "暂无待确认候选" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_memory_reject_uses_service_and_removes_pending_candidate(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            await pilot.press("r")
            await pilot.pause(0.1)
            assert service.rejected_candidate_ids == [("cand_abc123", "rejected from TUI")]
            assert "已拒绝记忆候选：cand_abc123" in _text(app, "#conversation")
            assert "暂无待确认候选" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_memory_panel_shows_empty_and_load_errors(tmp_path: Path) -> None:
    async def empty_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            assert "暂无待确认候选" in _text(app, "#conversation")

    async def error_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_error=RuntimeError("queue broken"))
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            assert "记忆候选不可用：queue broken" in _text(app, "#conversation")

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


def test_tui_conversation_does_not_auto_scroll_when_user_reads_history(tmp_path: Path) -> None:
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

            conversation.scroll_to(y=0, animate=False, force=True)
            await pilot.pause()
            app._append_block("Assistant", "new line while reading")
            app._refresh_conversation()
            await pilot.pause()

            assert conversation.scroll_y == 0

    asyncio.run(run())


def test_tui_end_key_scrolls_conversation_to_bottom_when_prompt_is_empty(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            conversation.scroll_to(y=0, animate=False, force=True)
            await pilot.pause()

            await pilot.press("end")
            await pilot.pause()

            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())


def test_tui_end_key_keeps_prompt_cursor_behavior_when_prompt_has_text(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            conversation = app.query_one("#conversation")
            for index in range(30):
                app._append_block("Assistant", f"line {index}")
            app._refresh_conversation()
            await pilot.pause()
            conversation.scroll_to(y=0, animate=False, force=True)
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "abc"
            input_widget.cursor_location = (0, 0)
            await pilot.pause()

            await pilot.press("end")
            await pilot.pause()

            assert conversation.scroll_y == 0
            assert input_widget.cursor_location == (0, 3)

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
            answer = conversation.query_one(".timeline-body")
            assert answer.region.width <= conversation.content_size.width
            assert conversation.virtual_size.height > len(app._conversation_lines)
            assert conversation.scroll_y == conversation.max_scroll_y

    asyncio.run(run())


def test_tui_assistant_body_renders_markdown_without_opening_links(tmp_path: Path) -> None:
    async def run() -> None:
        markdown_reply = (
            "# 结果\n\n"
            "- **重点**\n"
            "- `inline code`\n\n"
            "| 文件 | 状态 |\n"
            "| --- | --- |\n"
            "| README.md | 已读 |\n\n"
            "```python\nprint('ok')\n```\n\n"
            "[资料](https://example.com)"
        )
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content=markdown_reply)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "总结资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body")
            assert isinstance(assistant_body, Markdown)
            assert getattr(assistant_body, "_open_links") is False
            table_cells = [
                widget
                for widget in assistant_body.walk_children()
                if widget.has_class("cell") or widget.has_class("header")
            ]
            assert table_cells
            assert all(widget.tooltip is None for widget in table_cells)
            assert markdown_reply in conversation.plain_text

    asyncio.run(run())


def test_tui_assistant_final_message_replaces_streamed_markdown(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _assistant_event("assistant_delta", 1, "# 草稿"),
            ],
            assistant_content="# 最终\n\n- 完成",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "生成清单"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            assert assistant_body.source == "# 最终\n\n- 完成"
            assert "# 草稿" not in conversation.plain_text
            assert "# 最终\n\n- 完成" in conversation.plain_text

    asyncio.run(run())


def test_tui_streamed_markdown_table_cells_do_not_show_tooltips(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| 来源 | 链接 |\n"))
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| --- | --- |\n"))
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "| TASS | tass.com |\n"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "生成表格"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)
            await pilot.pause(0.3)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            table_cells = [
                widget
                for widget in assistant_body.walk_children()
                if widget.has_class("cell") or widget.has_class("header")
            ]
            assert table_cells
            assert all(widget.tooltip is None for widget in table_cells)
            release_event.set()
            await pilot.pause(0.2)

    asyncio.run(run())


def test_tui_merges_assistant_delta_events_into_single_response_block(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _assistant_event("assistant_delta", 1, "Ha"),
                _assistant_event("assistant_delta", 1, "Agent"),
            ],
            assistant_content="HaAgent",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍能力"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = _text(app, "#conversation")
            assert conversation.count("[HaAgent]") == 1
            assert "HaAgent" in conversation

    asyncio.run(run())


def test_tui_assistant_delta_updates_current_turn_without_overwriting_previous_turn(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="第一轮完整回答")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "第一问"
            await pilot.press("enter")
            await pilot.pause(0.2)

            app._handle_chat_event(_assistant_event("assistant_delta", 2, "第二"))
            app._handle_chat_event(_assistant_event("assistant_delta", 2, "轮"))
            app._handle_chat_event(_assistant_event("assistant_message", 2, "第二轮完整回答"))
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "第一轮完整回答" in conversation
            assert "第二轮完整回答" in conversation
            assert conversation.index("第一轮完整回答") < conversation.index("第二轮完整回答")

    asyncio.run(run())


def test_tui_final_message_on_new_turn_closes_previous_streaming_turn(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            app._handle_chat_event(_assistant_event("assistant_delta", 2, "worker 已启动，请稍候"))
            app._handle_chat_event(_assistant_event("assistant_message", 3, "worker 已完成，结论如下"))
            await asyncio.sleep(0)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_items = [item for item in conversation._items if item.role == "assistant"]
            assert [(item.turn_index, item.status) for item in assistant_items] == [
                (2, "done"),
                (3, "done"),
            ]

    asyncio.run(run())


def test_tui_streaming_turn_keeps_existing_widget_instances_stable(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "正在"))
                event_sink(_tool_event("tool_started", len(self.prompts), "file_read"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            if event_sink is not None:
                event_sink(_assistant_event("assistant_message", len(self.prompts), "最终回答"))
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assistant_blocks = list(conversation.query(".timeline-assistant"))
            assert assistant_blocks
            assistant_block = assistant_blocks[-1]
            children_before = list(conversation.children)

            rendered = _text(app, "#conversation")
            assert "file_read" in rendered
            assert "生成中" in rendered
            assert list(conversation.query(".timeline-assistant"))[-1] is assistant_block
            assert list(conversation.children) == children_before

            release_event.set()
            await pilot.pause(0.2)
            assert "最终回答" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_streaming_assistant_shows_rotating_cursor_and_cleans_up(tmp_path: Path) -> None:
    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "正在整理"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            if event_sink is not None:
                event_sink(_assistant_event("assistant_message", len(self.prompts), "最终回答"))
            return SimpleNamespace(status="completed")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            active = conversation.query_one(".timeline-assistant .timeline-active")
            first_frame = str(active.content)
            assistant_body = conversation.query_one(".timeline-assistant .timeline-body", Markdown)
            assert first_frame in {"|", "/", "-", "\\"}
            assert assistant_body.source == "正在整理"
            assert "生成中" in conversation.plain_text

            await pilot.pause(0.3)
            second_frame = str(active.content)
            assert second_frame in {"|", "/", "-", "\\"}
            assert second_frame != first_frame
            assert assistant_body.source == "正在整理"
            assert len(list(conversation.query(".timeline-assistant"))) == 1

            release_event.set()
            await pilot.pause(0.2)
            assert str(active.content) == ""
            assert active.display is False
            assert assistant_body.source == "最终回答"

    asyncio.run(run())


def test_tui_streaming_cursor_keeps_text_semantics_in_no_color_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    class BlockingAssistantService(FakeAssistantService):
        def __init__(self, *args, release_event: threading.Event, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.release_event = release_event
            self.stream_ready = threading.Event()

        def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
            self.prompts.append(prompt)
            self.started.set()
            if event_sink is not None:
                event_sink(_assistant_event("assistant_delta", len(self.prompts), "处理中"))
            self.stream_ready.set()
            self.release_event.wait(timeout=2)
            return SimpleNamespace(status="cancelled")

    async def run() -> None:
        release_event = threading.Event()
        service = BlockingAssistantService(workspace_root=tmp_path, release_event=release_event)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "查看资料"
            await pilot.press("enter")
            await asyncio.to_thread(service.stream_ready.wait, 2)
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            active = conversation.query_one(".timeline-assistant .timeline-active")
            assert "生成中" in str(active.content)
            assert "生成中" in conversation.plain_text

            release_event.set()
            await pilot.pause(0.2)
            assert str(active.content) == ""
            assert active.display is False

    asyncio.run(run())


def test_tui_compact_tool_summary_keeps_active_tool_visible_when_details_off(tmp_path: Path) -> None:
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
            assert "工具 3 项" in compact
            assert "2 成功" in compact
            assert "1 运行中" in compact
            assert "web_fetch" in compact

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


def test_tui_user_and_assistant_blocks_keep_spacing_and_alignment(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="已经完成。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "帮我总结"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            user = conversation.query_one(".timeline-user")
            assistant = conversation.query_one(".timeline-assistant")
            assert user.region.x <= conversation.region.x + 4
            assert assistant.region.x <= conversation.region.x + 4
            assert assistant.region.y >= user.region.y + user.region.height
            assert assistant.region.y > user.region.y

    asyncio.run(run())


def test_tui_conversation_scrollbar_states_do_not_use_black_track(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="已经完成。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "帮我总结"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assert conversation.styles.scrollbar_background == conversation.styles.background
            assert conversation.styles.scrollbar_background_hover == conversation.styles.background
            assert conversation.styles.scrollbar_background_active == conversation.styles.background
            assert conversation.styles.scrollbar_size_vertical == 0
            assert conversation.styles.scrollbar_size_horizontal == 0
            assert conversation.styles.scrollbar_visibility == "hidden"

    asyncio.run(run())


def test_tui_tool_events_attach_to_turn_summary_and_keep_answer_readable(tmp_path: Path) -> None:
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
            assert "工具 2 项" in conversation
            assert "1 成功" in conversation
            assert "1 失败" in conversation
            assert "运行中" not in conversation
            assert "file_read" in conversation
            assert "web_search" in conversation
            assert "我已经整理好结论。" in conversation
            assert conversation.index("工具 2 项") < conversation.index("我已经整理好结论。")

    asyncio.run(run())


def test_tui_tool_summary_defaults_to_single_line_even_for_few_tools(tmp_path: Path) -> None:
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
            assert "工具 2 项" in conversation
            assert "1 成功" in conversation
            assert "1 失败" in conversation
            assert "reading very long local file" not in conversation
            assert "web_search failed with timeout" not in conversation

    asyncio.run(run())


def test_tui_tool_summary_counts_calls_not_events_for_repeated_tools(tmp_path: Path) -> None:
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
            assert "工具 5 项" in conversation
            assert "3 成功" in conversation
            assert "1 运行中" in conversation
            assert "1 失败" in conversation
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
            assert "工具 3 项" in conversation
            assert "1 运行中" in conversation
            assert "2 失败" in conversation
            assert "待确认" not in conversation
            assert "审批已允许：shell" not in conversation

    asyncio.run(run())


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
            assert "工具 3 项" in compact
            assert "web_fetch" in compact
            assert "result compacted" not in compact
            assert "File change succeeded" not in compact
            assert "旧工具消息降级" not in compact

            input_widget.value = "/details"
            await pilot.press("enter")
            await pilot.pause(0.1)
            detailed = _text(app, "#conversation")
            assert "工具详情已开启" in detailed
            assert "web_fetch" in detailed
            assert "工具 file_read ok" in detailed
            assert "工具 web_search ok" in detailed
            assert "旧工具消息降级：web_search 1854 chars -> 929 chars" in detailed
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
            await _wait_for_conversation_bottom(app, pilot)
            assert app.query_one("#conversation").scroll_y == app.query_one("#conversation").max_scroll_y

    asyncio.run(run())


def test_tui_conversation_text_is_read_only_and_selectable(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = app.query_one("#conversation")
            assert isinstance(conversation, ConversationTimeline)
            app._append_block("Assistant", "这段回复应该可以被选中复制")
            app._refresh_conversation()
            assert "这段回复应该可以被选中复制" in conversation.plain_text

    asyncio.run(run())


def test_tui_no_color_timeline_keeps_role_and_status_labels(tmp_path: Path, monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[_tool_event("tool_failed", 1, "web_fetch", message="404")],
            assistant_content="已尝试获取网页。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "获取网页"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "[你]" in conversation
            assert "[HaAgent]" in conversation
            assert "[工具]" in conversation
            assert "失败" in conversation

    asyncio.run(run())


def test_tui_failure_event_shows_reason_episode_in_conversation(tmp_path: Path) -> None:
    async def run() -> None:
        episode_path = tmp_path / ".runs" / "episode-failed"
        service = FakeAssistantService(
            workspace_root=tmp_path,
            failure_event=FailureNoticeEvent(
                session_id="session-test",
                turn_index=1,
                status="failed",
                failed_stage="executing",
                failure_category="Loop Limit Failure",
                reason="exceeded max_turns=20",
                episode_path=str(episode_path),
            ),
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "介绍一下项目"
            await pilot.press("enter")
            await pilot.pause(0.2)
            conversation = _text(app, "#conversation")
            assert "本轮没有完成：模型连续调用工具但没有给出最终回答。" in conversation
            assert "stage=executing" in conversation
            assert "category=Loop Limit Failure" in conversation
            assert "reason=exceeded max_turns=20" in conversation
            assert str(episode_path) in conversation.replace("\n", "")
            assert "state: failed" in _text(app, "#status-bar")
            assert list(app.query("#side-bar")) == []

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


def test_tui_shows_assistant_placeholder_immediately_after_submit(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Search slowly"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            conversation = _text(app, "#conversation")
            assert "[你]" in conversation
            assert "[HaAgent]" in conversation

            service.release.set()
            await pilot.pause(0.2)

    asyncio.run(run())


def test_tui_ctrl_v_queues_image_token_in_prompt_and_next_prompt_sends_then_clears(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="收到图片。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            await pilot.press("ctrl+v")
            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "已添加图片附件" not in conversation
            assert "待发送附件" not in conversation
            assert input_widget.value == "[image 1] [image 2]"

            input_widget.value = f"{input_widget.value} 描述这张图"
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert service.prompts == ["描述这张图"]
            assert service.prompt_attachments == [[service.clipboard_attachments[0], service.clipboard_attachments[1]]]
            assert app._pending_attachments == []
            assert input_widget.value == ""
            conversation_after_send = _text(app, "#conversation")
            assert "[image 1] [image 2] 描述这张图" in conversation_after_send
            assert "待发送附件" not in conversation_after_send

    asyncio.run(run())


def test_tui_ctrl_v_image_paste_is_rejected_while_running(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "长任务"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            assert service.clipboard_attachments == []
            assert "运行中不能修改待发送附件" in _text(app, "#conversation")
            service.release.set()
            await pilot.pause(0.2)

    asyncio.run(run())


def test_tui_image_prompt_requires_vision_capable_model(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            model="deepseek-chat",
            image_input_supported=False,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            input_widget.value = f"{input_widget.value} 厉害吗"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.prompts == []
            assert service.prompt_attachments == []
            assert input_widget.value == "[image 1] 厉害吗"
            conversation = _text(app, "#conversation")
            assert "当前模型不支持图片输入" in conversation
            assert "deepseek-chat" in conversation

    asyncio.run(run())
