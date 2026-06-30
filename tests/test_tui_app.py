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
from haagent.app.assistant_service import AssistantSessionStatus, AssistantSessionSummary, AssistantWorkspaceStatus
from haagent.memory import CandidateEvidence, MemoryCandidate, MemoryRecord
from haagent.runtime.chat_session import ChatEvent
from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.app import HaAgentTuiApp, find_untrusted_absolute_paths
from haagent.tui.commands import SlashCommandResult, command_registry, parse_slash_command
from haagent.tui.changes import changed_files_from_tool_event, path_stays_in_workspace
from haagent.tui.failures import failure_next_steps
from haagent.tui.file_refs import FileReferenceIndex, FileReferenceMatch, build_file_reference_index, fuzzy_file_matches, path_reference_token
from haagent.tui.keys import APP_BINDINGS, footer_text, help_body, key_help_lines
from haagent.tui.models import ModelCatalogLoadingOverlay
from haagent.tui.copy import MODAL_TITLES, PANEL_TITLES
from haagent.tui.renderers import memory_panel_text, side_bar, status_line
from haagent.tui.search import ConversationSearchState
from haagent.tui.sessions import SessionOverlayState
from haagent.tui.state import ResponsiveLayout, layout_for_size
from haagent.tui.theme import (
    SemanticToken,
    TuiThemeMode,
    no_color_enabled,
    select_theme,
    semantic_tokens,
    status_semantic,
)
from haagent.tui.tool_timeline import ToolTimelineState, redact_mapping_for_display
from haagent.tui.widgets import PromptInput
from textual.widgets import RichLog, TextArea
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
        failure_event: ChatEvent | None = None,
        interaction_request: HumanInteractionRequest | None = None,
        extra_events: list[ChatEvent] | None = None,
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
        self.trusted_skills_count = 0
        self.untrusted_skills_count = 0
        self.read_skill_names: list[str] = []
        self.next_turn_target_paths: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.prompts: list[str] = []
        self.interaction_responses: list[HumanInteractionResponse] = []
        self.confirmed_candidate_ids: list[str] = []
        self.rejected_candidate_ids: list[tuple[str, str]] = []
        self.created_sessions: list[str] = []
        self.resumed_sessions: list[str] = []
        self.continued_latest_count = 0
        self.cancelled_count = 0
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
        )

    def run_prompt_events(self, prompt: str, *, event_sink=None, interaction_handler=None):
        self.prompts.append(prompt)
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

    def cancel_current_run(self):
        self.cancelled_count += 1
        self.block_until_released = False
        self.release.set()
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
    if isinstance(widget, TextArea):
        return widget.text
    if isinstance(widget, RichLog):
        return "\n".join("".join(segment.text for segment in line) for line in widget.lines)
    return str(widget.content)


def _all_text(app: HaAgentTuiApp) -> str:
    widgets = list(app.query("*"))
    if app.screen is not None:
        widgets.extend(app.screen.query("*"))
    pieces = []
    for widget in widgets:
        if isinstance(widget, TextArea):
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
    assert "web: on" in side_bar(online, ui_state="idle", last_failure=None)


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


def test_tui_chat_memory_entry_is_only_slash_command() -> None:
    chat_footer = footer_text("chat")
    chat_help = help_body("chat")
    binding_keys = {binding.key if hasattr(binding, "key") else binding[0] for binding in APP_BINDINGS}
    input_binding_keys = {binding.key for binding in PromptInput.BINDINGS}

    assert "/memory" in chat_help
    assert "[m]记忆" not in chat_footer
    assert "m" not in binding_keys
    assert "m" not in input_binding_keys


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
    assert PANEL_TITLES["workbench"] == "任务工作台"
    assert PANEL_TITLES["sessions"] == "会话"
    assert PANEL_TITLES["tools"] == "工具"
    assert PANEL_TITLES["memory"] == "记忆候选"
    assert PANEL_TITLES["search"] == "搜索"
    assert MODAL_TITLES["approval"] == "工具审批"
    assert MODAL_TITLES["tool_details"] == "工具详情"


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
    assert layout_for_size(79, 24) == ResponsiveLayout(too_small=True, show_side_bar=False)
    assert layout_for_size(80, 23) == ResponsiveLayout(too_small=True, show_side_bar=False)
    assert layout_for_size(80, 24) == ResponsiveLayout(too_small=False, show_side_bar=False)
    assert layout_for_size(120, 24) == ResponsiveLayout(too_small=False, show_side_bar=True)


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
    assert {command.name for command in registry.commands()} >= {
        "help",
        "sessions",
        "memory",
        "tools",
        "skills",
        "skill",
        "new",
        "resume",
        "model",
        "web",
        "permissions",
    }
    assert "models" not in {command.name for command in registry.commands()}


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
        async with app.run_test() as pilot:
            input_widget = app.query_one("#prompt-input", PromptInput)
            input_widget.value = "/skills"
            await pilot.press("enter")
            assert "review [user]" in _text(app, "#conversation")
            assert "项目 skills 未信任" in _text(app, "#conversation")

            input_widget.value = "/skills trust"
            await pilot.press("enter")
            assert service.trusted_skills_count == 1
            assert "已信任当前 workspace 的项目 skills" in _text(app, "#conversation")

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


def test_side_bar_shows_external_roots(tmp_path: Path) -> None:
    external = tmp_path / "external"
    status = FakeAssistantService(
        workspace_root=tmp_path,
        external_roots=[{"path": str(external), "access": "read", "source": "user"}],
    ).get_workspace_status()

    text = side_bar(status, ui_state="idle", last_failure=None)

    assert "外部目录" in text
    assert str(external) in text
    assert "只读参考" in text


def test_status_and_sidebar_show_permission_mode(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path, permission_mode="auto_approve")
    status = service.get_workspace_status()

    assert "perm:auto" in status_line(status, ui_state="idle", width=120)
    side = side_bar(status, ui_state="idle", last_failure=None)
    assert "权限模式" in side
    assert "自动批准" in side


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

    monkeypatch.setattr("haagent.tui.file_refs._iter_file_reference_candidates", lambda root: iter(candidates))

    index = build_file_reference_index(tmp_path)
    assert [item.display_path for item in index.matches("plan")] == ["docs/Project Plan.md"]
    assert [item.display_path for item in index.matches("main")] == ["src/main.py"]


def test_tui_file_reference_overlay_scrolls_selected_match_into_view(tmp_path: Path) -> None:
    from haagent.tui.file_ref_modal import FileReferenceOverlay

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
    from haagent.tui.file_ref_modal import FileReferenceOverlay

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
    from haagent.tui.file_ref_modal import FileReferenceOverlay

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
    from haagent.tui.file_ref_modal import FileReferenceOverlay

    overlay = FileReferenceOverlay(tmp_path, "")
    index = build_file_reference_index(tmp_path)
    overlay.on_unmount()

    def fail_reload():
        raise AssertionError("unmounted overlay should not refresh stale worker results")

    monkeypatch.setattr(overlay, "_reload", fail_reload)

    overlay._handle_index_ready(index)

    assert overlay.index is None


def test_tui_tool_timeline_state_tracks_started_done_failed_and_redacts_secret(tmp_path: Path) -> None:
    secret = "sk-test1234567890abcdef1234567890abcdef"
    state = ToolTimelineState()

    state.apply_event(
        ChatEvent(
            event_type="tool_started",
            session_id="session-test",
            turn_index=1,
            message="starting tool shell",
            payload={
                "tool_name": "shell",
                "args_summary": {"command": f"echo {secret}", "cwd": "."},
                "reason": "Run check",
            },
        ),
    )
    state.apply_event(
        ChatEvent(
            event_type="tool_finished",
            session_id="session-test",
            turn_index=1,
            message="finished tool shell",
            payload={
                "tool_name": "shell",
                "status": "success",
                "result_summary": {"exit_code": 0, "stdout_excerpt": f"ok {secret}", "stderr_excerpt": ""},
                "episode_path": str(tmp_path / ".runs" / "episode-ok"),
            },
        ),
    )
    state.apply_event(
        ChatEvent(
            event_type="tool_failed",
            session_id="session-test",
            turn_index=1,
            message="failed tool file_read",
            payload={
                "tool_name": "file_read",
                "args_summary": {"path": "missing.md"},
                "error_type": "tool_argument_invalid",
                "message": "path does not exist: missing.md",
            },
        ),
    )

    rendered = state.render()
    detail = state.selected_item().detail_text()

    assert "shell" in rendered
    assert "done" in rendered
    assert "file_read" in rendered
    assert "failed" in rendered
    assert secret not in rendered
    assert secret not in detail
    assert "[REDACTED_TOKEN]" in detail


def test_tui_changed_file_summary_extracts_file_write_and_patch_without_git(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    write_summary = changed_files_from_tool_event(
        "file_write",
        args_summary={"path": "notes/today.md", "mode": "create", "content_chars": 20},
        result_summary={"path": str(workspace / "notes" / "today.md"), "mode": "create", "bytes_written": 20, "created": True},
        workspace_root=workspace,
    )
    patch_summary = changed_files_from_tool_event(
        "apply_patch_set",
        args_summary={"paths": ["docs/a.md", "docs/b.md"], "replacement_count": 2},
        result_summary={"paths": ["docs/a.md", "docs/b.md"], "replacement_count": 2},
        workspace_root=workspace,
    )

    assert [item.path for item in write_summary] == ["notes/today.md"]
    assert write_summary[0].change_type == "added"
    assert "20 bytes" in write_summary[0].summary
    assert [item.path for item in patch_summary] == ["docs/a.md", "docs/b.md"]
    assert all(item.change_type == "modified" for item in patch_summary)


def test_tui_workspace_path_containment_is_normalized(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inside = workspace / "docs" / "plan.md"
    outside = tmp_path / "outside.md"

    assert path_stays_in_workspace("docs/plan.md", workspace)
    assert path_stays_in_workspace(str(inside), workspace)
    assert not path_stays_in_workspace("../outside.md", workspace)
    assert not path_stays_in_workspace(str(outside), workspace)


def test_tui_failure_next_steps_are_conservative() -> None:
    steps = failure_next_steps(
        failed_stage="executing",
        failure_category="Tool Argument Failure",
        reason="path does not exist",
        episode_path=".runs/episode-failed",
    )
    joined = "\n".join(steps)

    assert "查看工具详情" in joined
    assert "重试" in joined
    assert "调整请求" in joined
    assert ".runs/episode-failed" in joined
    assert "修复完成" not in joined
    assert "一定是" not in joined


def test_tui_redact_mapping_for_display_hides_secret_values() -> None:
    secret = "sk-test1234567890abcdef1234567890abcdef"

    text = redact_mapping_for_display({"command": f"echo {secret}", "path": "notes.md"})

    assert secret not in text
    assert "[REDACTED_TOKEN]" in text
    assert "notes.md" in text


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
            assert "模型配置" in side
            assert "base_url: https://api.deepseek.com" in side
            assert "api_key_env: DEEPSEEK_API_KEY" in side
            assert "Shift+Enter 换行" in _text(app, "#conversation")
            footer = _text(app, "#footer-bar")
            assert "[Ctrl+Q]退出" in footer
            assert "[q]退出" not in footer
            assert "[Enter]发送" in str(app.query_one("#footer-bar").render())
            assert "[Shift+Enter]换行" in str(app.query_one("#footer-bar").render())
            assert "[Tab]焦点" in str(app.query_one("#footer-bar").render())
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
            side = _text(app, "#side-bar")

            assert app.theme == "haagent-dark"
            assert app.screen.has_class("theme-dark")
            assert status_widget.has_class("status-default")
            assert "状态: - 空闲" in _text(app, "#status-bar")
            assert "任务工作台" in side
            assert "当前阶段" in side
            assert "工具时间线" in side
            assert "待处理事项" in side
            assert "变更文件" in side
            assert "最近失败" in side
            assert "工作区" in side
            assert "模型配置" in side
            assert "当前会话" in side

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
            assert "任务工作台" in _text(app, "#side-bar")

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
            side = _text(app, "#side-bar")

            assert app.query_one("#side-bar").has_class("panel-focused")
            assert "记忆候选" in side
            assert "> cand_second" in side
            assert "确认" in _text(app, "#footer-bar")

    asyncio.run(run())


def test_tui_m_key_no_longer_opens_memory_mode(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m")
            await pilot.pause(0.1)

            side = _text(app, "#side-bar")
            assert "记忆候选" not in side
            assert not app.query_one("#side-bar").has_class("panel-focused")

            prompt_input = app.query_one("#prompt-input")
            prompt_input.value = "/memory"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert "记忆候选" in _text(app, "#side-bar")
            assert app.query_one("#side-bar").has_class("panel-focused")

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
            assert "模型配置" in _text(app, "#side-bar")

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
            assert "/model" in conversation
            assert "uv run haagent setup" not in conversation
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
            assert "session-new-1" in _text(app, "#side-bar")

            input_widget.value = "/sessions"
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert "输入过滤  ↑/↓ 移动  Enter 恢复" not in _all_text(app)

    asyncio.run(run_resume())
    asyncio.run(run_continue_new_escape())


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
            assert "/resume" in _all_text(app)
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
            side = _text(app, "#side-bar")
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
            assert "shell ? 待审批 (pending approval)" in side
            assert "工具 shell ? 待审批" in conversation

    asyncio.run(run())


def test_tui_workbench_shows_phase_timeline_pending_changes_and_failure(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            interaction_request=_approval_request({"command": "uv run pytest -q", "cwd": ".", "timeout_seconds": 30}),
            extra_events=[
                ChatEvent(
                    event_type="tool_started",
                    session_id="session-test",
                    turn_index=1,
                    message="starting tool file_write",
                    payload={"tool_name": "file_write", "args_summary": {"path": "notes.md", "mode": "create"}},
                ),
                ChatEvent(
                    event_type="tool_finished",
                    session_id="session-test",
                    turn_index=1,
                    message="finished tool file_write",
                    payload={
                        "tool_name": "file_write",
                        "status": "success",
                        "result_summary": {"path": str(tmp_path / "notes.md"), "mode": "create", "bytes_written": 12, "created": True},
                        "episode_path": str(tmp_path / ".runs" / "episode-ok"),
                    },
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "写一个 notes"
            await pilot.press("enter")
            await pilot.pause(0.2)
            side = _text(app, "#side-bar")
            await pilot.press("n")
            await pilot.pause(0.2)
            failed_side = _text(app, "#side-bar")

            assert "任务工作台" in side
            assert "当前阶段" in side
            assert "waiting approval" in side
            assert "工具时间线" in side
            assert "file_write" in side
            assert "done" in side
            assert "待处理事项" in side
            assert "shell" in side
            assert "变更文件" in side
            assert "notes.md" in side
            assert "新增" in side
            assert "最近失败" in failed_side
            assert "request_user_input" not in failed_side
            assert "查看工具详情" in _text(app, "#conversation")

    asyncio.run(run())


def test_tui_tool_detail_overlay_opens_scrolls_closes_and_redacts_secret(tmp_path: Path) -> None:
    async def run() -> None:
        secret = "sk-test1234567890abcdef1234567890abcdef"
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                ChatEvent(
                    event_type="tool_started",
                    session_id="session-test",
                    turn_index=1,
                    message="starting tool shell",
                    payload={
                        "tool_name": "shell",
                        "args_summary": {"command": f"echo {secret}", "cwd": "."},
                        "reason": "检查命令输出",
                    },
                ),
                ChatEvent(
                    event_type="tool_finished",
                    session_id="session-test",
                    turn_index=1,
                    message="finished tool shell",
                    payload={
                        "tool_name": "shell",
                        "status": "success",
                        "result_summary": {
                            "exit_code": 0,
                            "stdout_excerpt": ("line\n" * 40) + secret,
                            "stderr_excerpt": "",
                        },
                        "episode_path": str(tmp_path / ".runs" / "episode-shell"),
                    },
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "运行命令"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("tab")
            await pilot.pause(0.1)
            await pilot.press("enter")
            await pilot.pause(0.1)
            rendered = _all_text(app)
            await pilot.press("pagedown")
            await pilot.pause(0.1)
            await pilot.press("escape")
            await pilot.pause(0.1)

            assert "工具详情" in rendered
            assert "tool name: shell" in rendered
            assert "status: ok 成功 (done)" in rendered
            assert "reason: 检查命令输出" in rendered
            assert "args:" in rendered
            assert "stdout:" in rendered
            assert "episode:" in rendered
            assert str(tmp_path / ".runs" / "episode-shell") in rendered
            assert secret not in rendered
            assert "[REDACTED_TOKEN]" in rendered
            assert "工具详情" not in _all_text(app)

    asyncio.run(run())


def test_tui_tool_timeline_keyboard_selection_opens_failed_detail(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                ChatEvent(
                    event_type="tool_started",
                    session_id="session-test",
                    turn_index=1,
                    message="starting tool file_read",
                    payload={"tool_name": "file_read", "args_summary": {"path": "README.md"}},
                ),
                ChatEvent(
                    event_type="tool_finished",
                    session_id="session-test",
                    turn_index=1,
                    message="finished tool file_read",
                    payload={"tool_name": "file_read", "status": "success", "result_summary": {"path": "README.md"}},
                ),
                ChatEvent(
                    event_type="tool_failed",
                    session_id="session-test",
                    turn_index=1,
                    message="failed tool file_write",
                    payload={
                        "tool_name": "file_write",
                        "args_summary": {"path": "blocked.md"},
                        "error_type": "tool_argument_invalid",
                        "message": "parent directory does not exist",
                    },
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "读取再写入"
            await pilot.press("enter")
            await pilot.pause(0.2)
            await pilot.press("tab")
            await pilot.pause(0.1)
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> file_write" in _text(app, "#side-bar")
            await pilot.press("enter")
            await pilot.pause(0.1)
            rendered = _all_text(app)

            assert "工具详情" in rendered
            assert "tool name: file_write" in rendered
            assert "status: ! 失败 (failed)" in rendered
            assert "parent directory does not exist" in rendered

    asyncio.run(run())


def test_tui_tools_overlay_available_when_sidebar_collapsed(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                ChatEvent(
                    event_type="tool_started",
                    session_id="session-test",
                    turn_index=1,
                    message="starting tool file_read",
                    payload={"tool_name": "file_read", "args_summary": {"path": "README.md"}},
                ),
            ],
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "读文件"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert app.query_one("#side-bar").has_class("hidden")
            input_widget.value = "/tools"
            await pilot.press("enter")
            await pilot.pause(0.1)
            rendered = _all_text(app)

            assert "任务工作台" in rendered
            assert "工具时间线" in rendered
            assert "file_read" in rendered

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
            assert "state: cancelled" in _text(app, "#status-bar")
            assert "当前阶段" in _text(app, "#side-bar")
            assert "cancelled" in _text(app, "#side-bar")
            assert app._pending_interaction is None
            assert "任务已取消" in _text(app, "#conversation")

            service.started.clear()
            input_widget.value = "Second task"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert service.prompts[-1] == "Second task"

    asyncio.run(run())


def test_tui_layout_sizes_keep_workbench_stable(tmp_path: Path) -> None:
    async def run_80() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(80, 24)):
            assert app.query_one("#side-bar").has_class("hidden")
            assert "/tools" in _text(app, "#footer-bar")

    async def run_120() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            side = _text(app, "#side-bar")
            assert "任务工作台" in side
            assert "当前阶段" in side
            assert "工具时间线" in side
            assert "变更文件" in side

    async def run_200() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(200, 60)):
            side = _text(app, "#side-bar")
            assert "任务工作台" in side
            assert "工作区" in side
            assert "模型配置" in side

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
            assert "审批已允许：shell" in _text(app, "#conversation")
            assert "shell ok 已允许 (approved)" in _text(app, "#side-bar")
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
            assert "shell ! 已拒绝 (denied)" in _text(app, "#side-bar")
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
            assert "工具 request_user_input ! 失败 (failed)" in conversation
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
            assert "记忆候选" in _text(app, "#side-bar")
            assert "cand_abc123" in _text(app, "#side-bar")
            assert "[a/y]确认" in _text(app, "#footer-bar")

    asyncio.run(run())


def test_tui_memory_panel_lists_and_shows_candidate_details(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, memory_candidates=[_memory_candidate()])
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            side = _text(app, "#side-bar")
            footer = _text(app, "#footer-bar")
            assert "记忆候选" in side
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
            await _open_memory_panel(app, pilot)
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
            await _open_memory_panel(app, pilot)
            await pilot.press("down")
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
            await _open_memory_panel(app, pilot)
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
            assert "> cand_second" in _text(app, "#side-bar")

            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_third" in _text(app, "#side-bar")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")

            await pilot.press("up")
            await pilot.pause(0.1)
            assert "> cand_first" in _text(app, "#side-bar")

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
            assert "> cand_first" in _text(app, "#side-bar")
            await pilot.press("down")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")
            await pilot.press("k")
            await pilot.pause(0.1)
            assert "> cand_second" in _text(app, "#side-bar")

    asyncio.run(run())


def test_tui_memory_mode_is_readable_without_sidebar(tmp_path: Path) -> None:
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
            assert "暂无待确认候选" in _text(app, "#side-bar")

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
            assert "暂无待确认候选" in _text(app, "#side-bar")

    asyncio.run(run())


def test_tui_memory_panel_shows_empty_and_load_errors(tmp_path: Path) -> None:
    async def empty_run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_memory_panel(app, pilot)
            assert "暂无待确认候选" in _text(app, "#side-bar")

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
            wrapped_lines = [section for line in conversation.wrapped_document.lines for section in line]
            longest_line = max(len(line) for line in wrapped_lines)
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


def test_tui_conversation_text_is_read_only_and_selectable(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)):
            conversation = app.query_one("#conversation")
            assert isinstance(conversation, TextArea)
            assert conversation.read_only is True
            app._append_block("Assistant", "这段回复应该可以被选中复制")
            app._refresh_conversation()
            conversation.select_all()
            assert "这段回复应该可以被选中复制" in conversation.selected_text

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
            assert "最近失败" in side
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

