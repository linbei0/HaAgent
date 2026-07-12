"""
tests/tui/support.py - HaAgent TUI 集成测试共享 Fake 与 helpers

供 tests/tui 下按领域拆分的集成测试复用，避免重复构造 FakeAssistantService。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

from haagent import cli
from haagent.app.assistant_types import (
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


class FakeSandbox:
    def __init__(self, owner) -> None:
        self._owner = owner

    def status(self):
        return self._owner._get_sandbox_status()

    def doctor_report(self):
        return self._owner._get_sandbox_doctor_report()

    def enable_docker(self, *, fail_if_unavailable: bool = True):
        return self._owner._enable_docker_sandbox(fail_if_unavailable=fail_if_unavailable)

    def disable(self):
        return self._owner._disable_sandbox()


class FakeWorkspace:
    def __init__(self, owner) -> None:
        self._owner = owner
        self.sandbox = FakeSandbox(owner)

    def status(self):
        return self._owner._get_workspace_status()

    def current_session(self):
        return self._owner._session_status(self._owner.current_session_id)

    def mcp_status(self):
        return self._owner._get_mcp_status()

    def list_agents(self):
        return self._owner._list_agents()

    def set_web_enabled(self, enabled: bool):
        return self._owner._set_web_enabled(enabled)

    def turn_limit_status(self):
        return self._owner._get_turn_limit_status()

    def set_interactive_max_turns(self, max_turns: int):
        return self._owner._set_interactive_max_turns(max_turns)

    def set_current_turns_unlimited(self):
        return self._owner._set_current_turns_unlimited()


class FakePermissions:
    def __init__(self, owner) -> None:
        self._owner = owner

    def set_mode(self, mode):
        return self._owner._set_permission_mode(mode)

    def set_next_turn_targets(self, paths):
        return self._owner._set_next_turn_target_paths(paths)

    def add_external_root(self, path, access):
        return self._owner._add_external_root(path, access)

    def remove_external_root(self, path):
        return self._owner._remove_external_root(path)

    def set_external_root_access(self, path, access):
        return self._owner._set_external_root_access(path, access)

    def clear_external_roots(self):
        return self._owner._clear_external_roots()

    def switch_project_root(self, path):
        return self._owner._switch_project_root(path)


class FakeSessions:
    def __init__(self, owner) -> None:
        self._owner = owner
        self.permissions = FakePermissions(owner)

    @property
    def initial_resume(self):
        return getattr(self._owner, "initial_resume", None)

    @property
    def initial_continue(self):
        return bool(getattr(self._owner, "initial_continue", False))

    def create(self):
        return self._owner._create_session()

    def resume(self, session):
        return self._owner._resume_session(session)

    def continue_latest(self):
        return self._owner._continue_latest_session()

    def list(self):
        return self._owner._list_sessions()

    def history(self):
        return self._owner._current_session_history()

    def compact(self):
        return self._owner._compact_current_session()

    def run_prompt_events(self, prompt, **kwargs):
        return self._owner._run_prompt_events(prompt, **kwargs)

    def paste_clipboard_image(self, **kwargs):
        return self._owner._paste_clipboard_image(**kwargs)

    def cancel_current_run(self):
        return self._owner._cancel_current_run()


class FakeModels:
    def __init__(self, owner) -> None:
        self._owner = owner

    def list_connections(self):
        return self._owner._list_model_connections()

    def configure_connection(self, request):
        return self._owner._configure_model_connection(request)

    def delete_connection(self, connection_id):
        return self._owner._delete_model_connection(connection_id)

    def set_default_selection(self, request):
        return self._owner._set_default_model_selection(request)

    def get_catalog(self, **kwargs):
        return self._owner._get_model_catalog(**kwargs)

    def refresh_catalog(self, **kwargs):
        return self._owner._refresh_model_catalog(**kwargs)

    def test_connection(self, connection_id, *, model=None):
        return self._owner._test_model_connection(connection_id, model=model)

    def switch_current_session_selection(self, request):
        return self._owner._switch_current_session_model_selection(request)


class FakeSkills:
    def __init__(self, owner) -> None:
        self._owner = owner

    def list(self):
        return self._owner._list_skills()

    def trust_project(self):
        return self._owner._trust_project_skills()

    def untrust_project(self):
        return self._owner._untrust_project_skills()

    def read_for_user(self, name):
        return self._owner._read_skill_for_user(name)

    def search_marketplace(self, query, **kwargs):
        return self._owner._search_skill_marketplace(query, **kwargs)

    def install_marketplace(self, result_id):
        return self._owner._install_marketplace_skill(result_id)


class FakeMemory:
    def __init__(self, owner) -> None:
        self._owner = owner

    def list_candidates(self, *, status="pending"):
        return self._owner._list_memory_candidates(status=status)

    def get_candidate(self, candidate_id):
        return self._owner._get_memory_candidate(candidate_id)

    def confirm_candidate(self, candidate_id):
        return self._owner._confirm_memory_candidate(candidate_id)

    def reject_candidate(self, candidate_id, reason):
        return self._owner._reject_memory_candidate(candidate_id, reason)


class FakeChannels:
    def __init__(self, owner) -> None:
        self._owner = owner
        self.pairing_codes: list[tuple[str, str]] = []

    def list_instances(self):
        return list(self._owner.channel_instances)

    def set_enabled(self, instance_id, enabled):
        for item in self._owner.channel_instances:
            if item.id == instance_id:
                item.enabled = enabled
                return item
        raise RuntimeError(f"missing channel {instance_id}")

    def delete_instance(self, instance_id):
        self._owner.channel_instances = [
            item for item in self._owner.channel_instances if item.id != instance_id
        ]

    def issue_pairing_code(self, instance_id, *, expires_in_seconds=600):
        del expires_in_seconds
        code = "FAKEPAIR"
        self.pairing_codes.append((instance_id, code))
        return code

    def set_workspace_root(self, instance_id, workspace_root):
        for item in self._owner.channel_instances:
            if item.id == instance_id:
                item.workspace_root = workspace_root
                return item
        raise RuntimeError(f"missing channel {instance_id}")


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
        self.session_summaries = list(sessions or [])
        self.session_histories = dict(session_histories or {})
        self.enable_web = enable_web
        self.external_roots = list(external_roots or [])
        self.permission_mode = permission_mode
        self.skill_entries = list(skills or [])
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
        self.model_connections: list[SimpleNamespace] = []
        self.switched_model_connection: str | None = None
        self.switched_model: str | None = None
        self.current_session_model_connection: str | None = None
        self.default_model_selection: str | None = None
        self.deleted_model_connection: str | None = None
        self.catalog_providers: list[SimpleNamespace] = []
        self.cached_catalog_providers: list[SimpleNamespace] | None = None
        self.configured_model_connection = None
        self.configured_api_key: str | None = None
        self.connection_test_result = SimpleNamespace(
            ok=True,
            profile_name="router",
            provider="openai-chat",
            model="openai/gpt-5.2-chat",
            message="OK",
        )
        self.tested_model_connection: str | None = None
        self.tested_model: str | None = None
        self.refreshed_catalog_count = 0
        self.got_catalog_count = 0
        self.catalog_refresh_release: threading.Event | None = None
        self.workspace = FakeWorkspace(self)
        self.sessions = FakeSessions(self)
        self.models = FakeModels(self)
        self.skills = FakeSkills(self)
        self.memory = FakeMemory(self)
        self.channel_instances: list[SimpleNamespace] = []
        self.channels = FakeChannels(self)

    def _get_workspace_status(self) -> AssistantWorkspaceStatus:
        current_profile = next(
            (
                connection
                for connection in self._list_model_connections()
                if connection.id == self.current_session_model_connection
            ),
            None,
        )
        profile_name = getattr(current_profile, "name", self.profile_name)
        provider = getattr(current_profile, "gateway_provider", self.provider)
        model = getattr(current_profile, "model", self.model)
        return AssistantWorkspaceStatus(
            workspace_root=self.workspace_root,
            runs_root=self.workspace_root / ".runs",
            profile_name=profile_name,
            provider=provider,
            base_url="https://api.deepseek.com",
            model=model,
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

    def _get_mcp_status(self) -> dict[str, object]:
        return dict(self.mcp_status)

    def _list_agents(self) -> list[dict[str, object]]:
        return list(self.agents)

    def _get_sandbox_status(self) -> AssistantSandboxStatus:
        return self.sandbox_status

    def _get_sandbox_doctor_report(self) -> SandboxDoctorReport:
        return self.sandbox_doctor_report

    def _enable_docker_sandbox(self, *, fail_if_unavailable: bool = True) -> AssistantSandboxStatus:
        self.sandbox_enabled_count += 1
        self.sandbox_status = AssistantSandboxStatus(
            backend="docker",
            degraded=False,
            reason="" if fail_if_unavailable else "fallback allowed",
        )
        return self.sandbox_status

    def _disable_sandbox(self) -> AssistantSandboxStatus:
        self.sandbox_disabled_count += 1
        self.sandbox_status = AssistantSandboxStatus(
            backend="local_subprocess",
            degraded=True,
            reason="docker sandbox disabled",
        )
        return self.sandbox_status

    def _run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None, attachments=None):
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

    def _paste_clipboard_image(self, *, existing=None):
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

    def _create_session(self) -> AssistantSessionStatus:
        session_id = f"session-new-{len(self.created_sessions) + 1}"
        self.created_sessions.append(session_id)
        self.current_session_id = session_id
        return self._session_status(session_id)

    def _resume_session(self, session: str | Path) -> AssistantSessionStatus:
        session_text = str(session)
        self.resumed_sessions.append(session_text)
        match = next((item for item in self.session_summaries if str(item.session_path) == session_text), None)
        self.current_session_id = match.session_id if match is not None else session_text
        return self._session_status(self.current_session_id)

    def _continue_latest_session(self) -> AssistantSessionStatus:
        self.continued_latest_count += 1
        if self.session_summaries:
            self.current_session_id = self.session_summaries[0].session_id
        return self._session_status(self.current_session_id)

    def _compact_current_session(self):
        self.compacted_count += 1
        return SimpleNamespace(
            applied=True,
            reason="applied",
            original_turn_count=8,
            compacted_turn_count=2,
            preserved_recent_count=6,
            saved_chars=1200,
        )

    def _cancel_current_run(self):
        self.cancelled_count += 1
        return SimpleNamespace(status="cancelled", reason="user_cancelled")

    def _list_sessions(self) -> list[AssistantSessionSummary]:
        return list(self.session_summaries)

    def _current_session_history(self):
        return list(self.session_histories.get(self.current_session_id, []))

    def _list_model_connections(self):
        return [_connection_record(connection) for connection in self.model_connections]

    def _refresh_model_catalog(self):
        self.refreshed_catalog_count += 1
        if self.catalog_refresh_release is not None:
            self.catalog_refresh_release.wait(timeout=5)
        return SimpleNamespace(providers=list(self.catalog_providers), used_cache=False, error=None)

    def _get_model_catalog(self):
        self.got_catalog_count += 1
        if self.catalog_refresh_release is not None:
            self.catalog_refresh_release.wait(timeout=5)
        providers = self.catalog_providers if self.cached_catalog_providers is None else self.cached_catalog_providers
        return SimpleNamespace(providers=list(providers), used_cache=True, error=None)

    def _configure_model_connection(self, request):
        self.configured_model_connection = request
        self.configured_api_key = request.api_key
        return request

    def _test_model_connection(self, connection_id: str, model: str | None = None):
        self.tested_model_connection = connection_id
        self.tested_model = model
        return self.connection_test_result

    def _switch_current_session_model_selection(self, request) -> AssistantSessionStatus:
        connection_id = request.connection_id
        self.switched_model_connection = connection_id
        self.switched_model = request.model
        self.current_session_model_connection = connection_id
        for index, profile in enumerate(self.model_connections):
            normalized = _connection_record(profile)
            self.model_connections[index] = SimpleNamespace(
                **{**normalized.__dict__, "current_session": normalized.id == connection_id},
            )
        selected = next((item for item in self._list_model_connections() if item.id == connection_id), None)
        return AssistantSessionStatus(
            session_id=self.current_session_id,
            workspace_root=self.workspace_root,
            runs_root=self.workspace_root / ".runs",
            session_path=self.workspace_root / ".runs" / "sessions" / self.current_session_id,
            turn_count=len(self.prompts),
            max_turns=None,
            provider=self.provider or "-",
            model_profile_name=f"{connection_id}:{request.model}",
            model_connection_id=connection_id,
            model=request.model,
            base_url=getattr(selected, "base_url", "https://api.deepseek.com"),
            external_roots=list(self.external_roots),
            permission_mode=self.permission_mode,
        )

    def _set_default_model_selection(self, request) -> None:
        self.default_model_selection = request

    def _set_web_enabled(self, enabled: bool) -> AssistantWorkspaceStatus:
        self.enable_web = enabled
        return self._get_workspace_status()

    def _delete_model_connection(self, connection_id: str) -> None:
        self.deleted_model_connection = connection_id
        self.model_connections = [
            connection for connection in self.model_connections if _connection_record(connection).id != connection_id
        ]

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

    def _set_permission_mode(self, mode: str) -> AssistantSessionStatus:
        self.permission_mode = mode
        return self._session_status(self.current_session_id)

    def _set_next_turn_target_paths(self, paths: list[str | Path]) -> AssistantSessionStatus:
        self.next_turn_target_paths = [str(Path(path).resolve()) for path in paths]
        return self._session_status(self.current_session_id)

    def _add_external_root(self, path: str | Path, access: str) -> AssistantSessionStatus:
        resolved = str(Path(path).resolve())
        self.external_roots = [root for root in self.external_roots if root["path"] != resolved]
        self.external_roots.append({"path": resolved, "access": access, "source": "user"})
        return self._session_status(self.current_session_id)

    def _remove_external_root(self, path: str | Path) -> AssistantSessionStatus:
        resolved = str(Path(path).resolve())
        self.external_roots = [root for root in self.external_roots if root["path"] != resolved]
        return self._session_status(self.current_session_id)

    def _set_external_root_access(self, path: str | Path, access: str) -> AssistantSessionStatus:
        self._add_external_root(path, access)
        return self._session_status(self.current_session_id)

    def _clear_external_roots(self) -> AssistantSessionStatus:
        self.external_roots = []
        return self._session_status(self.current_session_id)

    def _switch_project_root(self, path: str | Path) -> AssistantSessionStatus:
        self.workspace_root = Path(path).resolve()
        self.external_roots = []
        return self._session_status(self.current_session_id)

    def _list_skills(self):
        return SimpleNamespace(
            skills=list(self.skill_entries),
            blocked_project_skill_roots=list(self.blocked_project_skill_roots),
        )

    def _trust_project_skills(self):
        self.trusted_skills_count += 1
        self.blocked_project_skill_roots = []
        return self._list_skills()

    def _untrust_project_skills(self):
        self.untrusted_skills_count += 1
        return self._list_skills()

    def _read_skill_for_user(self, name: str):
        self.read_skill_names.append(name)
        match = next((item for item in self.skill_entries if item.get("name") == name or item.get("command_name") == name), None)
        if match is None:
            raise RuntimeError(f"skill not found: {name}")
        return SimpleNamespace(
            name=str(match.get("name")),
            command_name=str(match.get("command_name") or match.get("name")),
            content=f"# {match.get('name')}\nFollow this workflow.",
        )

    def _search_skill_marketplace(self, query: str, *, providers=None, limit: int = 10):
        provider_tuple = tuple(providers) if providers is not None else None
        self.searched_marketplace_queries.append((query, provider_tuple, limit))
        return SimpleNamespace(
            status="success" if self.marketplace_results else "error",
            query=query,
            results=list(self.marketplace_results),
            warnings=list(self.marketplace_warnings),
        )

    def _install_marketplace_skill(self, result_id: str):
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

    def _list_memory_candidates(self, status: str | None = "pending") -> list[MemoryCandidate]:
        if self.memory_error is not None:
            raise self.memory_error
        if status is None:
            return list(self.memory_candidates)
        return [candidate for candidate in self.memory_candidates if candidate.status == status]

    def _get_memory_candidate(self, candidate_id: str) -> MemoryCandidate:
        if self.memory_error is not None:
            raise self.memory_error
        for candidate in self.memory_candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise RuntimeError(f"memory candidate not found: {candidate_id}")

    def _confirm_memory_candidate(self, candidate_id: str) -> MemoryRecord:
        candidate = self._get_memory_candidate(candidate_id)
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

    def _reject_memory_candidate(self, candidate_id: str, reason: str) -> MemoryCandidate:
        candidate = self._get_memory_candidate(candidate_id)
        self.rejected_candidate_ids.append((candidate_id, reason))
        self.memory_candidates = [item for item in self.memory_candidates if item.candidate_id != candidate_id]
        raw = candidate.to_dict()
        raw["status"] = "rejected"
        raw["updated_at"] = "2026-06-26T00:00:00+00:00"
        return MemoryCandidate.from_dict(raw)


def _connection_record(connection: SimpleNamespace) -> SimpleNamespace:
    name = str(getattr(connection, "name", getattr(connection, "id", "local")))
    connection_id = str(getattr(connection, "id", name))
    gateway_provider = str(getattr(connection, "gateway_provider", getattr(connection, "provider", "openai-chat")))
    return SimpleNamespace(
        id=connection_id,
        name=name,
        provider_id=str(getattr(connection, "provider_id", connection_id)),
        provider_name=str(getattr(connection, "provider_name", name)),
        gateway_provider=gateway_provider,
        base_url=str(getattr(connection, "base_url", "https://api.deepseek.com")),
        api_key_env=str(getattr(connection, "api_key_env", "DEEPSEEK_API_KEY")),
        credential_source=str(getattr(connection, "credential_source", "keyring")),
        credential_available=bool(getattr(connection, "credential_available", True)),
        credential_source_used=getattr(connection, "credential_source_used", None),
        model=str(getattr(connection, "model", "deepseek-chat")),
    )


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


