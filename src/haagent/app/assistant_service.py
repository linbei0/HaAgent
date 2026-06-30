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
from haagent.runtime.chat_session import (
    CHAT_MAX_TURNS,
    AgentSession,
    ChatEvent,
    ChatSessionError,
    ChatTurnResult,
    SessionSummary,
    SessionTurnSummary,
    find_latest_session,
    list_sessions,
)
from haagent.runtime.human_interaction import HumanInteractionHandler
from haagent.runtime.path_policy import PathAccess, PermissionMode
from haagent.skills import trust_project_root, untrust_project_root
from haagent.tools.skills import skill_list, skill_read
from haagent.memory import (
    CandidateQueue,
    CandidateQueueError,
    MemoryCandidate,
    MemoryRecord,
    MemoryStore,
    MemoryStoreError,
)
from haagent.memory.governance import MemoryGovernanceError


GatewayFactory = Callable[[ProviderProfile], ModelGateway]
EventSink = Callable[[ChatEvent], None]


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
        self.initial_resume = initial_resume
        self.initial_continue = initial_continue

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

    def set_web_enabled(self, enabled: bool) -> AssistantWorkspaceStatus:
        self.enable_web = enabled
        if self._session is not None:
            self._session.enable_web = enabled
        return self.get_workspace_status()

    def create_session(self) -> AssistantSessionStatus:
        try:
            profile = self._load_session_profile()
            self._session = self.session_cls(
                workspace_root=self.workspace_root,
                runs_root=self.runs_root,
                model_gateway=self.gateway_factory(profile),
                model_profile_name=profile.name,
                model_name=profile.model,
                model_base_url=profile.base_url,
                max_turns=self.max_turns,
                enable_web=self.enable_web,
            )
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error
        return _session_status(self._session)

    def resume_session(self, session: str | Path) -> AssistantSessionStatus:
        try:
            profile = self._load_resume_profile(session)
            self._session = self.session_cls.resume(
                session,
                runs_root=self.runs_root,
                model_gateway=self.gateway_factory(profile),
                model_profile_name=profile.name,
                model_name=profile.model,
                model_base_url=profile.base_url,
                max_turns=self.max_turns,
                enable_web=self.enable_web,
            )
        except (ChatSessionError, ProviderProfileError) as error:
            raise AssistantServiceError(str(error)) from error
        return _session_status(self._session)

    def continue_latest_session(self) -> AssistantSessionStatus:
        try:
            latest = find_latest_session(self.runs_root, self.workspace_root)
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error
        if latest is None:
            raise AssistantServiceError("当前 workspace 没有可恢复会话")
        return self.resume_session(latest.session_path)

    def list_sessions(self) -> list[AssistantSessionSummary]:
        try:
            return [_session_summary(summary) for summary in list_sessions(self.runs_root, self.workspace_root)]
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error

    def current_session_history(self) -> list[AssistantSessionTurn]:
        if self._session is None:
            return []
        try:
            return [_session_turn(turn) for turn in self._session.turn_summaries()]
        except ChatSessionError as error:
            raise AssistantServiceError(str(error)) from error

    def list_model_profiles(self) -> list[AssistantModelProfile]:
        try:
            active_profile_name = load_active_profile_name()
        except ProviderProfileError:
            active_profile_name = None
        profiles: list[AssistantModelProfile] = []
        for record in list_provider_profile_records():
            credential = provider_profile_credential_status(
                record.name,
                environ=self.environ,
                config_dir=user_config_dir(),
            )
            profiles.append(
                AssistantModelProfile(
                    name=record.name,
                    provider=record.provider,
                    base_url=record.base_url,
                    model=record.model,
                    api_key_env=record.api_key_env,
                    credential_source=record.credential_source,
                    active=record.name == active_profile_name,
                    credential_available=credential.api_key_available,
                    credential_source_used=credential.credential_source_used,
                    capability=gateway_capability_for_profile(record),
                )
            )
        return profiles

    def set_default_model_profile(self, profile_name: str) -> None:
        load_provider_profile_record(profile_name)
        save_active_profile(profile_name, config_dir=user_config_dir())

    def configure_model_profile(self, request: ModelProfileConfigureRequest) -> ProviderProfileRecord:
        record = ProviderProfileRecord(
            name=request.name,
            provider=request.provider,
            base_url=request.base_url,
            model=request.model,
            api_key_env=request.api_key_env,
            credential_source=request.credential_source,
        )
        try:
            save_provider_profile_with_key(
                record,
                request.api_key,
                credential_store=provider_profile_module.DEFAULT_CREDENTIAL_STORE,
                config_dir=user_config_dir(),
            )
        except (ProviderProfileError, CredentialError) as error:
            raise AssistantServiceError(str(error)) from error
        return record

    def delete_model_profile(self, profile_name: str) -> None:
        try:
            delete_provider_profile(profile_name, config_dir=user_config_dir())
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error

    def refresh_model_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        try:
            return fetch_model_catalog(transport=transport, force_refresh=True)
        except Exception as error:
            raise AssistantServiceError(str(error)) from error

    def get_model_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        try:
            return fetch_model_catalog(
                transport=transport,
                max_cache_age=DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
            )
        except Exception as error:
            raise AssistantServiceError(str(error)) from error

    def test_model_profile(self, profile_name: str) -> AssistantModelTestResult:
        try:
            profile = load_provider_profile(
                profile_name,
                environ=self.environ,
                config_dir=user_config_dir(),
            )
            gateway = self.gateway_factory(profile)
            response = gateway.generate(
                [{"role": "user", "content": "Reply with OK."}],
                [],
            )
            return AssistantModelTestResult(
                ok=True,
                profile_name=profile.name,
                provider=profile.provider,
                model=profile.model,
                message=_redact_secret_text(response.content, [profile.api_key]),
            )
        except (ProviderProfileError, CredentialError, ModelCallError) as error:
            record = _load_profile_record_for_result(profile_name)
            return AssistantModelTestResult(
                ok=False,
                profile_name=profile_name,
                provider=record.provider if record is not None else "",
                model=record.model if record is not None else "",
                message=_redact_secret_text(str(error), _secret_candidates(self.environ)),
            )

    def switch_current_session_model(self, profile_name: str) -> AssistantSessionStatus:
        try:
            profile = load_provider_profile(
                profile_name,
                environ=self.environ,
                config_dir=user_config_dir(),
            )
            if self._session is None:
                self._pending_model_profile_name = profile.name
                return AssistantSessionStatus(
                    session_id="pending",
                    workspace_root=self.workspace_root,
                    runs_root=self.runs_root,
                    session_path=self.runs_root,
                    turn_count=0,
                    provider=profile.provider,
                    model_profile_name=profile.name,
                    model=profile.model,
                    base_url=profile.base_url,
                    web_enabled=self.enable_web,
                    permission_mode="request_approval",
                )
            gateway = self.gateway_factory(profile)
            self._session.switch_model_gateway(
                profile_name=profile.name,
                provider=profile.provider,
                model=profile.model,
                base_url=profile.base_url,
                gateway=gateway,
            )
        except (ProviderProfileError, ChatSessionError) as error:
            raise AssistantServiceError(str(error)) from error
        return _session_status(self._session)

    def set_permission_mode(self, mode: PermissionMode) -> AssistantSessionStatus:
        if mode not in {"request_approval", "auto_approve", "full_access"}:
            raise AssistantServiceError("permission mode must be request_approval, auto_approve, or full_access")
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self._session.set_permission_mode(mode)
        return _session_status(self._session)

    def set_next_turn_target_paths(self, paths: list[str | Path]) -> AssistantSessionStatus:
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self._session.set_next_turn_target_paths([Path(path) for path in paths])
        return _session_status(self._session)

    def add_external_root(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        if access not in {"read", "full"}:
            raise AssistantServiceError("external root access must be read or full")
        if self._session is None:
            self.create_session()
        assert self._session is not None
        root = Path(path).resolve()
        if not root.exists():
            raise AssistantServiceError(f"外部目录不存在：{root}")
        if not root.is_dir():
            raise AssistantServiceError(f"外部路径必须是目录：{root}")
        self._session.add_external_root(root, access)
        return _session_status(self._session)

    def remove_external_root(self, path: str | Path) -> AssistantSessionStatus:
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self._session.remove_external_root(Path(path))
        return _session_status(self._session)

    def set_external_root_access(self, path: str | Path, access: PathAccess) -> AssistantSessionStatus:
        if access not in {"read", "full"}:
            raise AssistantServiceError("external root access must be read or full")
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self._session.set_external_root_access(Path(path), access)
        return _session_status(self._session)

    def clear_external_roots(self) -> AssistantSessionStatus:
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self._session.clear_external_roots()
        return _session_status(self._session)

    def switch_project_root(self, path: str | Path) -> AssistantSessionStatus:
        root = Path(path).resolve()
        if not root.exists():
            raise AssistantServiceError(f"项目目录不存在：{root}")
        if not root.is_dir():
            raise AssistantServiceError(f"项目路径必须是目录：{root}")
        if self._session is None:
            self.create_session()
        assert self._session is not None
        self.workspace_root = root
        self._session.switch_project_root(root)
        return _session_status(self._session)

    def list_skills(self) -> AssistantSkillList:
        result = skill_list({}, self.workspace_root)
        if result.get("status") != "success":
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            raise AssistantServiceError(str(error.get("message", "failed to list skills")))
        return AssistantSkillList(
            skills=list(result.get("skills", [])),
            blocked_project_skill_roots=[
                str(path) for path in result.get("blocked_project_skill_roots", [])
            ],
        )

    def trust_project_skills(self) -> AssistantSkillList:
        trust_project_root(self.workspace_root)
        return self.list_skills()

    def untrust_project_skills(self) -> AssistantSkillList:
        untrust_project_root(self.workspace_root)
        return self.list_skills()

    def read_skill_for_user(self, name: str) -> AssistantSkillContent:
        result = skill_read({"name": name}, self.workspace_root, user_invoked=True)
        if result.get("status") != "success":
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            raise AssistantServiceError(str(error.get("message", f"skill not found: {name}")))
        return AssistantSkillContent(
            name=str(result["name"]),
            command_name=str(result.get("command_name") or result["name"]),
            content=str(result["content"]),
        )

    def run_prompt_events(
        self,
        prompt: str,
        *,
        event_sink: EventSink | None = None,
        include_session_events: bool = True,
        interaction_handler: HumanInteractionHandler | None = None,
    ) -> ChatTurnResult:
        if self._session is None:
            self.create_session()
        assert self._session is not None
        return self._session.run_prompt_events(
            prompt,
            event_sink=event_sink,
            include_session_events=include_session_events,
            interaction_handler=interaction_handler,
        )

    def cancel_current_run(self) -> AssistantCancelResult:
        if self._session is None:
            return AssistantCancelResult(status="idle", reason="no_active_session")
        self._session.cancel_current_run()
        return AssistantCancelResult(status="cancelled", reason="user_cancelled")

    def list_memory_candidates(self, status: str | None = "pending") -> list[MemoryCandidate]:
        queue = self._memory_queue()
        return queue.list(status=status)

    def get_memory_candidate(self, candidate_id: str) -> MemoryCandidate:
        return self._memory_queue().get(candidate_id)

    def confirm_memory_candidate(self, candidate_id: str) -> MemoryRecord:
        try:
            return self._memory_store().confirm_candidate(
                self._memory_queue(),
                candidate_id,
                actor="user",
            )
        except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
            raise AssistantServiceError(str(error)) from error

    def reject_memory_candidate(self, candidate_id: str, reason: str) -> MemoryCandidate:
        try:
            return self._memory_store().reject_candidate(
                self._memory_queue(),
                candidate_id,
                reason=reason,
                actor="user",
            )
        except (CandidateQueueError, MemoryStoreError, MemoryGovernanceError) as error:
            raise AssistantServiceError(str(error)) from error

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
