"""
haagent/app/assistant_service.py - 个人助手应用服务层

封装 CLI 与未来 TUI 共用的 workspace、profile、session 和事件流能力。
"""

from __future__ import annotations

import os
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from haagent.models import provider_profile as provider_profile_module
from haagent.mcp.settings import load_mcp_settings
from haagent.models.catalog import (
    DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
    CatalogFetchResult,
    CatalogTransport,
    fetch_model_catalog,
)
from haagent.models.credentials import CredentialError
from haagent.models.gateway import ModelGateway
from haagent.models.gateway import ModelCallError
from haagent.models.gateway_registry import GatewayCapability, gateway_capability_for_profile, gateway_from_profile
from haagent.models.provider_profile import (
    ProviderProfile,
    ProviderProfileError,
    ProviderProfileRecord,
    active_provider_credential_status,
    list_provider_profile_records,
    load_active_profile_name,
    load_active_provider_profile,
    load_active_provider_profile_record,
    load_provider_profile,
    load_provider_profile_record,
    provider_profile_credential_status,
    delete_provider_profile,
    save_active_profile,
    save_provider_profile_with_key,
    user_config_dir,
)
from haagent.runtime.session.agent import (
    CHAT_MAX_TURNS,
    AgentSession,
    ChatSessionError,
    ChatTurnResult,
    SessionSummary,
    SessionTurnSummary,
    find_latest_session,
    list_sessions,
)
from haagent.runtime.events import RuntimeUiEvent
from haagent.runtime.execution.human_interaction import HumanInteractionHandler
from haagent.runtime.execution.path_policy import PathAccess, PermissionMode
from haagent.skills import trust_project_root, untrust_project_root
from haagent.skills.marketplace import MarketplaceError, MarketplaceSkillCard, install_marketplace_skill_card, search_marketplace
from haagent.tools.skills import skill_list, skill_read
from haagent.memory import (
    CandidateQueue,
    MemoryCandidate,
    MemoryRecord,
    MemoryStore,
)
from haagent.app import memory_usecases, model_profile_usecases, session_usecases, skill_usecases


GatewayFactory = Callable[[ProviderProfile], ModelGateway]
EventSink = Callable[[RuntimeUiEvent], None]


class AssistantServiceError(RuntimeError):
    """AssistantService 无法完成显式请求时抛出。"""


@dataclass(frozen=True)
class AssistantWorkspaceStatus:
    workspace_root: Path
    runs_root: Path
    profile_name: str | None
    provider: str | None
    base_url: str | None
    model: str | None
    api_key_env: str | None
    api_key_available: bool
    credential_source_configured: str | None = None
    credential_source_used: str | None = None
    credential_store_available: bool | None = None
    credential_store_error: str | None = None
    profile_error: str | None = None
    current_session_id: str | None = None
    current_turn_count: int | None = None
    web_enabled: bool = False
    external_roots: list[dict[str, str]] | None = None
    permission_mode: PermissionMode = "request_approval"


@dataclass(frozen=True)
class AssistantSessionStatus:
    session_id: str
    workspace_root: Path
    runs_root: Path
    session_path: Path
    turn_count: int
    provider: str
    model_profile_name: str | None = None
    model: str | None = None
    base_url: str | None = None
    web_enabled: bool = False
    external_roots: list[dict[str, str]] | None = None
    permission_mode: PermissionMode = "request_approval"


@dataclass(frozen=True)
class AssistantSessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    workspace_root: Path
    turn_count: int
    first_request: str
    session_path: Path


@dataclass(frozen=True)
class AssistantSessionTurn:
    turn_index: int
    request: str
    summary: str
    status: str
    episode_path: Path
    verification_status: str


@dataclass(frozen=True)
class AssistantSessionCompactResult:
    applied: bool
    reason: str
    original_turn_count: int
    compacted_turn_count: int
    preserved_recent_count: int
    saved_chars: int


@dataclass(frozen=True)
class AssistantCancelResult:
    status: str
    reason: str


@dataclass(frozen=True)
class AssistantModelProfile:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    credential_source: str
    active: bool
    credential_available: bool
    credential_source_used: str | None
    capability: GatewayCapability


@dataclass(frozen=True)
class AssistantModelTestResult:
    ok: bool
    profile_name: str
    provider: str
    model: str
    message: str


@dataclass(frozen=True)
class AssistantSkillList:
    skills: list[dict[str, object]]
    blocked_project_skill_roots: list[str]


@dataclass(frozen=True)
class AssistantSkillContent:
    name: str
    command_name: str
    content: str


@dataclass(frozen=True)
class AssistantMarketplaceSkill:
    result_id: str
    provider: str
    name: str
    source: str
    summary: str
    detail_url: str
    installable: bool
    quality: dict[str, int | float | str]


@dataclass(frozen=True)
class AssistantMarketplaceSearch:
    status: str
    query: str
    results: list[AssistantMarketplaceSkill]
    warnings: list[str]


@dataclass(frozen=True)
class AssistantMarketplaceInstall:
    name: str
    command_name: str
    skill_dir: Path
    skill_file: Path
    source_url: str


@dataclass(frozen=True)
class ModelProfileConfigureRequest:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    credential_source: str
    api_key: str | None = None


class AssistantService:
    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        runs_root: Path = Path(".runs"),
        environ: Mapping[str, str] | None = None,
        gateway_factory: GatewayFactory | None = None,
        session_cls: type[AgentSession] = AgentSession,
        max_turns: int = CHAT_MAX_TURNS,
        enable_web: bool = False,
        initial_resume: str | Path | None = None,
        initial_continue: bool = False,
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.runs_root = runs_root
        self.environ = os.environ if environ is None else environ
        self.gateway_factory = gateway_factory or gateway_from_profile
        self.session_cls = session_cls
        self.max_turns = max_turns
        self.enable_web = enable_web
        self._session: AgentSession | None = None
        self._pending_model_profile_name: str | None = None
        self._marketplace_results: dict[str, MarketplaceSkillCard] = {}
        self.initial_resume = initial_resume
        self.initial_continue = initial_continue
        self.error_cls = AssistantServiceError
        self.cancel_result_cls = AssistantCancelResult
        self.session_compact_result_cls = AssistantSessionCompactResult
        self.session_status_cls = AssistantSessionStatus
        self.model_profile_cls = AssistantModelProfile
        self.model_test_result_cls = AssistantModelTestResult
        self.skill_list_cls = AssistantSkillList
        self.skill_content_cls = AssistantSkillContent
        self.marketplace_search_cls = AssistantMarketplaceSearch
        self.marketplace_install_cls = AssistantMarketplaceInstall
        self.chat_session_error_cls = ChatSessionError
        self.gateway_capability_for_profile = gateway_capability_for_profile
        self.marketplace_skill_mapper = _marketplace_skill
        self._session_status = _session_status
        self._session_summary = _session_summary
        self._session_turn = _session_turn
        self.load_profile_record_for_result = _load_profile_record_for_result
        self.secret_candidates = _secret_candidates
        self.redact_secret_text = _redact_secret_text
        self.skill_list_fn = lambda args, workspace_root, skill_settings=None: skill_list(
            args,
            workspace_root,
            skill_settings,
        )
        self.skill_read_fn = lambda args, workspace_root, user_invoked=False, skill_settings=None: skill_read(
            args,
            workspace_root,
            skill_settings,
            user_invoked=user_invoked,
        )
        self.trust_project_root_fn = lambda workspace_root: trust_project_root(workspace_root)
        self.untrust_project_root_fn = lambda workspace_root: untrust_project_root(workspace_root)
        self.search_marketplace_fn = lambda query, *, providers=None, limit=10: search_marketplace(
            query,
            providers=providers,
            limit=limit,
        )
        self.install_marketplace_skill_card_fn = lambda card: install_marketplace_skill_card(card)

    def get_workspace_status(self) -> AssistantWorkspaceStatus:
        profile_name: str | None = None
        provider: str | None = None
        base_url: str | None = None
        model: str | None = None
        api_key_env: str | None = None
        api_key_available = False
        credential_source_configured: str | None = None
        credential_source_used: str | None = None
        credential_store_available: bool | None = None
        credential_store_error: str | None = None
        profile_error: str | None = None
        try:
            record = load_active_provider_profile_record()
            profile_name = record.name
            provider = record.provider
            base_url = record.base_url
            model = record.model
            api_key_env = record.api_key_env
            credential = active_provider_credential_status(environ=self.environ)
            api_key_available = credential.api_key_available
            credential_source_configured = credential.credential_source_configured
            credential_source_used = credential.credential_source_used
            credential_store_available = credential.credential_store_available
            credential_store_error = credential.credential_store_error
        except ProviderProfileError as error:
            profile_error = str(error)
        session_status = self.current_session()
        return AssistantWorkspaceStatus(
            workspace_root=self.workspace_root,
            runs_root=self.runs_root,
            profile_name=profile_name,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            api_key_available=api_key_available,
            credential_source_configured=credential_source_configured,
            credential_source_used=credential_source_used,
            credential_store_available=credential_store_available,
            credential_store_error=credential_store_error,
            profile_error=profile_error,
            current_session_id=session_status.session_id if session_status is not None else None,
            current_turn_count=session_status.turn_count if session_status is not None else None,
            web_enabled=self.enable_web,
            external_roots=session_status.external_roots if session_status is not None else [],
            permission_mode=session_status.permission_mode if session_status is not None else "request_approval",
        )

    def current_session(self) -> AssistantSessionStatus | None:
        if self._session is None:
            return None
        return _session_status(self._session)

    def get_mcp_status(self) -> dict[str, object]:
        if self._session is None:
            settings = load_mcp_settings()
            servers = [
                {
                    "name": name,
                    "state": "configured",
                    "detail": "not loaded; create or resume a session to connect",
                    "tool_count": 0,
                    "resource_count": 0,
                }
                for name in settings.servers
            ]
            return {
                "configured_count": len(servers),
                "connected_count": 0,
                "failed_count": 0,
                "servers": servers,
            }
        mcp_status = getattr(self._session, "mcp_status", None)
        if callable(mcp_status):
            return mcp_status()
        return {
            "configured_count": 0,
            "connected_count": 0,
            "failed_count": 0,
            "servers": [],
        }

    def set_web_enabled(self, enabled: bool) -> AssistantWorkspaceStatus:
        self.enable_web = enabled
        if self._session is not None:
            self._session.enable_web = enabled
        return self.get_workspace_status()

    def create_session(self) -> AssistantSessionStatus:
        return session_usecases.create_session(self)

    def resume_session(self, session: str | Path) -> AssistantSessionStatus:
        return session_usecases.resume_session(self, session)

    def continue_latest_session(self) -> AssistantSessionStatus:
        return session_usecases.continue_latest_session(self)

    def list_sessions(self) -> list[AssistantSessionSummary]:
        return session_usecases.list_sessions_for_workspace(self)

    def current_session_history(self) -> list[AssistantSessionTurn]:
        return session_usecases.current_session_history(self)

    def compact_current_session(self) -> AssistantSessionCompactResult:
        return session_usecases.compact_current_session(self)

    def list_model_profiles(self) -> list[AssistantModelProfile]:
        return model_profile_usecases.list_model_profiles(self)

    def set_default_model_profile(self, profile_name: str) -> None:
        model_profile_usecases.set_default_model_profile(self, profile_name)

    def configure_model_profile(self, request: ModelProfileConfigureRequest) -> ProviderProfileRecord:
        return model_profile_usecases.configure_model_profile(self, request)

    def delete_model_profile(self, profile_name: str) -> None:
        model_profile_usecases.delete_model_profile_for_user(self, profile_name)

    def refresh_model_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        return model_profile_usecases.refresh_model_catalog(self, transport=transport)

    def get_model_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        return model_profile_usecases.get_model_catalog(self, transport=transport)

    def test_model_profile(self, profile_name: str) -> AssistantModelTestResult:
        return model_profile_usecases.test_model_profile(self, profile_name)

    def switch_current_session_model(self, profile_name: str) -> AssistantSessionStatus:
        return model_profile_usecases.switch_current_session_model(self, profile_name)

    def set_permission_mode(self, mode: PermissionMode) -> AssistantSessionStatus:
        return session_usecases.set_permission_mode(self, mode)

    def set_next_turn_target_paths(self, paths: list[str | Path]) -> AssistantSessionStatus:
        return session_usecases.set_next_turn_target_paths(self, paths)

    def add_external_root(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        return session_usecases.add_external_root(self, path, access)

    def remove_external_root(self, path: str | Path) -> AssistantSessionStatus:
        return session_usecases.remove_external_root(self, path)

    def set_external_root_access(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        return session_usecases.set_external_root_access(self, path, access)

    def clear_external_roots(self) -> AssistantSessionStatus:
        return session_usecases.clear_external_roots(self)

    def switch_project_root(self, path: str | Path) -> AssistantSessionStatus:
        return session_usecases.switch_project_root(self, path)

    def list_skills(self) -> AssistantSkillList:
        return skill_usecases.list_skills_for_user(self)

    def trust_project_skills(self) -> AssistantSkillList:
        return skill_usecases.trust_project_skills(self)

    def untrust_project_skills(self) -> AssistantSkillList:
        return skill_usecases.untrust_project_skills(self)

    def read_skill_for_user(self, name: str) -> AssistantSkillContent:
        return skill_usecases.read_skill_for_user(self, name)

    def search_skill_marketplace(
        self,
        query: str,
        *,
        providers: list[str] | None = None,
        limit: int = 10,
    ) -> AssistantMarketplaceSearch:
        return skill_usecases.search_skill_marketplace(
            self,
            query,
            providers=providers,
            limit=limit,
        )

    def install_marketplace_skill(self, result_id: str) -> AssistantMarketplaceInstall:
        return skill_usecases.install_marketplace_skill(self, result_id)

    def run_prompt_events(
        self,
        prompt: str,
        *,
        event_sink: EventSink | None = None,
        include_session_events: bool = True,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        return session_usecases.run_prompt_events(
            self,
            prompt,
            event_sink=event_sink,
            include_session_events=include_session_events,
            interaction_handler=interaction_handler,
        )

    def cancel_current_run(self) -> AssistantCancelResult:
        return session_usecases.cancel_current_run(self)

    def list_memory_candidates(self, status: str | None = "pending") -> list[MemoryCandidate]:
        return memory_usecases.list_memory_candidates(self, status=status)

    def get_memory_candidate(self, candidate_id: str) -> MemoryCandidate:
        return memory_usecases.get_memory_candidate(self, candidate_id)

    def confirm_memory_candidate(self, candidate_id: str) -> MemoryRecord:
        return memory_usecases.confirm_memory_candidate(self, candidate_id)

    def reject_memory_candidate(self, candidate_id: str, reason: str) -> MemoryCandidate:
        return memory_usecases.reject_memory_candidate(self, candidate_id, reason)

    def _load_session_profile(self) -> ProviderProfile:
        if self._pending_model_profile_name is not None:
            return load_provider_profile(
                self._pending_model_profile_name,
                environ=self.environ,
                config_dir=user_config_dir(),
            )
        return load_active_provider_profile(environ=self.environ)

    def _load_resume_profile(self, session: str | Path) -> ProviderProfile:
        profile_name = _session_model_profile_name(session, self.runs_root)
        if profile_name is not None:
            return load_provider_profile(
                profile_name,
                environ=self.environ,
                config_dir=user_config_dir(),
            )
        return load_active_provider_profile(environ=self.environ)

    def _memory_queue(self) -> CandidateQueue:
        if self._session is None:
            latest = find_latest_session(self.runs_root, self.workspace_root)
            if latest is None:
                raise AssistantServiceError("当前 workspace 没有可审查的 memory candidate session")
            session_path = latest.session_path
        else:
            session_path = self._session.session_path
        return CandidateQueue(session_path)

    def _memory_store(self) -> MemoryStore:
        return MemoryStore(workspace_root=self.workspace_root)


def _marketplace_skill(card: MarketplaceSkillCard) -> AssistantMarketplaceSkill:
    return AssistantMarketplaceSkill(
        result_id=card.result_id,
        provider=card.provider.value,
        name=card.name,
        source=card.source,
        summary=card.summary,
        detail_url=card.detail_url,
        installable=card.installable,
        quality=dict(card.quality),
    )


def _session_status(session: AgentSession) -> AssistantSessionStatus:
    return AssistantSessionStatus(
        session_id=session.session_id,
        workspace_root=session.workspace_root,
        runs_root=session.runs_root,
        session_path=session.session_path.resolve(),
        turn_count=session.turn_count,
        provider=session.provider_name,
        model_profile_name=getattr(session, "model_profile_name", None),
        model=getattr(session, "model_name", None),
        base_url=getattr(session, "model_base_url", None),
        web_enabled=getattr(session, "enable_web", False),
        external_roots=_external_root_summaries(session),
        permission_mode=_session_permission_mode(session),
    )


def _session_summary(summary: SessionSummary) -> AssistantSessionSummary:
    return AssistantSessionSummary(
        session_id=summary.session_id,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        workspace_root=summary.workspace_root,
        turn_count=summary.turn_count,
        first_request=summary.first_request,
        session_path=summary.session_path,
    )


def _session_turn(turn: SessionTurnSummary) -> AssistantSessionTurn:
    return AssistantSessionTurn(
        turn_index=turn.turn_index,
        request=turn.request,
        summary=turn.summary,
        status=turn.status,
        episode_path=turn.episode_path,
        verification_status=turn.verification_status,
    )


def _external_root_summaries(session: AgentSession) -> list[dict[str, str]]:
    policy = getattr(session, "path_policy", None)
    if policy is None:
        return []
    return [
        {
            "path": str(root.path.resolve()),
            "access": root.access,
            "source": root.source,
        }
        for root in policy.external_roots
    ]


def _session_permission_mode(session: AgentSession) -> PermissionMode:
    policy = getattr(session, "path_policy", None)
    mode = getattr(policy, "permission_mode", "request_approval")
    if mode in {"request_approval", "auto_approve", "full_access"}:
        return mode
    return "request_approval"


def _session_model_profile_name(session: str | Path, runs_root: Path) -> str | None:
    raw = Path(session)
    if raw.is_absolute() or raw.exists() or raw.name != str(session):
        session_path = raw.resolve()
    else:
        session_path = (runs_root / "sessions" / str(session)).resolve()
    metadata_path = session_path / "session.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    profile_name = metadata.get("model_profile_name")
    if isinstance(profile_name, str) and profile_name:
        return profile_name
    return None


def _load_profile_record_for_result(profile_name: str):
    try:
        return load_provider_profile_record(profile_name)
    except ProviderProfileError:
        return None


def _secret_candidates(environ: Mapping[str, str]) -> list[str]:
    return [value for value in environ.values() if isinstance(value, str) and value.strip()]


def _redact_secret_text(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted
