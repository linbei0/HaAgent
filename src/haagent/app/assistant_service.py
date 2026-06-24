"""
haagent/app/assistant_service.py - 个人助手应用服务层

封装 CLI 与未来 TUI 共用的 workspace、profile、session 和事件流能力。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from haagent.models.gateway import (
    ModelGateway,
    OpenAIChatCompletionsGateway,
    OpenAIResponsesGateway,
)
from haagent.models.provider_profile import (
    ProviderProfile,
    ProviderProfileError,
    active_provider_credential_status,
    load_active_provider_profile,
    load_active_provider_profile_record,
)
from haagent.runtime.chat_session import (
    CHAT_MAX_TURNS,
    AgentSession,
    ChatEvent,
    ChatSessionError,
    ChatTurnResult,
    SessionSummary,
    find_latest_session,
    list_sessions,
)
from haagent.runtime.human_interaction import HumanInteractionHandler


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


@dataclass(frozen=True)
class AssistantSessionStatus:
    session_id: str
    workspace_root: Path
    runs_root: Path
    session_path: Path
    turn_count: int
    provider: str


@dataclass(frozen=True)
class AssistantSessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    workspace_root: Path
    turn_count: int
    first_request: str
    session_path: Path


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
    ) -> None:
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.runs_root = runs_root
        self.environ = os.environ if environ is None else environ
        self.gateway_factory = gateway_factory or _gateway_from_profile
        self.session_cls = session_cls
        self.max_turns = max_turns
        self._session: AgentSession | None = None

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
        )

    def current_session(self) -> AssistantSessionStatus | None:
        if self._session is None:
            return None
        return _session_status(self._session)

    def create_session(self) -> AssistantSessionStatus:
        try:
            self._session = self.session_cls(
                workspace_root=self.workspace_root,
                runs_root=self.runs_root,
                model_gateway=self._build_model_gateway(),
                max_turns=self.max_turns,
            )
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error
        return _session_status(self._session)

    def resume_session(self, session: str | Path) -> AssistantSessionStatus:
        try:
            self._session = self.session_cls.resume(
                session,
                runs_root=self.runs_root,
                model_gateway=self._build_model_gateway(),
                max_turns=self.max_turns,
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

    def _build_model_gateway(self) -> ModelGateway:
        profile = load_active_provider_profile(environ=self.environ)
        return self.gateway_factory(profile)


def _gateway_from_profile(profile: ProviderProfile) -> ModelGateway:
    gateway_kwargs = {
        "api_key": profile.api_key,
        "model": profile.model,
        "base_url": profile.base_url,
    }
    if profile.provider == "openai":
        return OpenAIResponsesGateway(**gateway_kwargs)
    if profile.provider == "openai-chat":
        return OpenAIChatCompletionsGateway(**gateway_kwargs)
    raise ProviderProfileError(f"unsupported provider in profile: {profile.provider}")


def _session_status(session: AgentSession) -> AssistantSessionStatus:
    return AssistantSessionStatus(
        session_id=session.session_id,
        workspace_root=session.workspace_root,
        runs_root=session.runs_root,
        session_path=session.session_path.resolve(),
        turn_count=session.turn_count,
        provider=session.provider_name,
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
