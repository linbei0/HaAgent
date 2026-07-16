"""
haagent/app/workspace_usecases.py - Workspace 状态与运行控制 Module

聚合 workspace 状态，并管理 MCP、sandbox、联网和 turn limit 等运行控制。
"""

from __future__ import annotations

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantSandboxStatus,
    AssistantServiceError,
    AssistantTurnLimitStatus,
    AssistantWorkspaceStatus,
    SandboxDoctorReport,
)
from haagent.mcp.settings import load_mcp_settings
from haagent.models.config.connections import ProviderProfileError, user_config_dir
from haagent.multi_agent.team_store import TeamStore
from haagent.runtime.settings import RuntimeSettingsError, load_runtime_settings, set_interactive_max_turns
from haagent.runtime.sandbox.status import (
    disable_sandbox as disable_runtime_sandbox,
    enable_docker_sandbox,
    sandbox_doctor_report,
    sandbox_user_status,
)


def sandbox_status() -> AssistantSandboxStatus:
    status = sandbox_user_status()
    return AssistantSandboxStatus(
        backend=status.backend,
        degraded=status.degraded,
        reason=status.reason,
    )


class AssistantSandbox:
    def status(self) -> AssistantSandboxStatus:
        return sandbox_status()

    def doctor_report(self) -> SandboxDoctorReport:
        return sandbox_doctor_report(check_disabled=True)

    def enable_docker(self, *, fail_if_unavailable: bool = True) -> AssistantSandboxStatus:
        enable_docker_sandbox(fail_if_unavailable=fail_if_unavailable)
        return sandbox_status()

    def disable(self) -> AssistantSandboxStatus:
        disable_runtime_sandbox()
        return sandbox_status()


class AssistantWorkspace:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context
        self.sandbox = AssistantSandbox()
        # token 热路径会频繁读 status；缓存后避免每次打 keyring / 重读 providers。
        self._status_cache: AssistantWorkspaceStatus | None = None
        self._status_cache_key: tuple[object, ...] | None = None

    def mcp_status(self) -> dict[str, object]:
        if self._context.session is None:
            servers = [
                {
                    "name": name,
                    "state": "configured",
                    "detail": "not loaded; create or resume a session to connect",
                    "tool_count": 0,
                    "resource_count": 0,
                }
                for name in load_mcp_settings().servers
            ]
            return {
                "configured_count": len(servers),
                "connected_count": 0,
                "failed_count": 0,
                "servers": servers,
            }
        return self._context.session.mcp_status()

    def list_agents(self) -> list[dict[str, object]]:
        if self._context.session is None:
            return []
        store = TeamStore(user_config_dir() / "teams")
        return [
            {
                "team_id": team.team_id,
                "agent_id": worker.agent_id,
                "task_id": worker.task_id,
                "subagent_type": worker.subagent_type,
                "description": worker.description,
                "status": worker.status,
                "episode_path": worker.episode_path,
            }
            for team in store.list_teams_for_leader(self._context.session.session_id)
            for worker in team.agents
        ]

    def set_web_enabled(self, enabled: bool) -> AssistantWorkspaceStatus:
        self._context.enable_web = enabled
        if self._context.session is not None:
            self._context.session.enable_web = enabled
        self._context.status_generation += 1
        self._status_cache = None
        self._status_cache_key = None
        return self.status()

    def turn_limit_status(self) -> AssistantTurnLimitStatus:
        current = self._context.session.max_turns if self._context.session is not None else self._context.max_turns
        return AssistantTurnLimitStatus(
            current_max_turns=current,
            configured_interactive_max_turns=load_runtime_settings().interactive_max_turns,
        )

    def set_interactive_max_turns(self, max_turns: int) -> AssistantTurnLimitStatus:
        try:
            settings = set_interactive_max_turns(max_turns)
        except RuntimeSettingsError as error:
            raise AssistantServiceError(str(error)) from error
        self._context.max_turns = settings.interactive_max_turns
        if self._context.session is not None:
            self._context.session.set_max_turns(settings.interactive_max_turns)
        return self.turn_limit_status()

    def set_current_turns_unlimited(self) -> AssistantTurnLimitStatus:
        if self._context.session is None:
            raise AssistantServiceError("当前没有 session；先发送一条消息再使用 /turns unlimited。")
        self._context.session.set_max_turns(None)
        return self.turn_limit_status()

    def status(self) -> AssistantWorkspaceStatus:
        cache_key = self._status_cache_key_for_current()
        if self._status_cache is not None and self._status_cache_key == cache_key:
            return self._status_cache
        status = self._build_status()
        self._status_cache = status
        self._status_cache_key = cache_key
        return status

    def _status_cache_key_for_current(self) -> tuple[object, ...]:
        """本地可观察字段 + generation；凭据文件变化靠 invalidate 抬 generation。"""

        session = self._context.session
        policy = getattr(session, "path_policy", None) if session is not None else None
        external = ()
        permission = None
        if policy is not None:
            permission = getattr(policy, "permission_mode", None)
            roots = getattr(policy, "external_roots", ()) or ()
            external = tuple(
                (str(getattr(root, "path", "")), getattr(root, "access", ""), getattr(root, "source", ""))
                for root in roots
            )
        if session is None:
            return (
                self._context.status_generation,
                None,
                None,
                None,
                None,
                permission,
                external,
                self._context.enable_web,
                self._context.workspace_root,
                self._context.runs_root,
            )
        return (
            self._context.status_generation,
            session.session_id,
            session.turn_count,
            session.model_ref,
            permission,
            external,
            self._context.enable_web,
            self._context.workspace_root,
            self._context.runs_root,
            id(session),
        )

    def _build_status(self) -> AssistantWorkspaceStatus:
        from haagent.app.session_usecases import session_status

        session = session_status(self._context.session) if self._context.session is not None else None
        override = self._context.session.model_ref if self._context.session is not None else None
        profile_name = provider = base_url = model = api_key_env = None
        model_variant = None
        credential_source_configured = credential_source_used = None
        credential_store_available = None
        credential_store_error = profile_error = None
        api_key_available = False
        try:
            assert self._context.model_runtime is not None
            selection = override or self._context.model_runtime.selection_store.load_active()
            snapshot = self._context.model_runtime.snapshot
            connection = snapshot.connection(selection.connection_id)
            profile_name = connection.id
            provider = connection.gateway_provider
            base_url = connection.base_url
            model = selection.model
            model_variant = selection.variant
            api_key_env = connection.api_key_env
            credential = self._context.model_runtime.credential_status(connection.id)
            api_key_available = credential.api_key_available
            credential_source_configured = credential.credential_source_configured
            credential_source_used = credential.credential_source_used
            credential_store_available = credential.credential_store_available
            credential_store_error = credential.credential_store_error
        except ProviderProfileError as error:
            profile_error = str(error)
        return AssistantWorkspaceStatus(
            workspace_root=self._context.workspace_root,
            runs_root=self._context.runs_root,
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
            current_session_id=session.session_id if session is not None else None,
            current_turn_count=session.turn_count if session is not None else None,
            web_enabled=self._context.enable_web,
            external_roots=session.external_roots if session is not None else [],
            permission_mode=session.permission_mode if session is not None else "request_approval",
            sandbox_status=session.sandbox_status if session is not None else sandbox_status(),
            image_input_supported=_image_input_supported(provider, base_url, model),
            model_variant=model_variant,
        )


def _image_input_supported(provider: str | None, base_url: str | None, model: str | None) -> bool | None:
    provider_key = (provider or "").strip().lower()
    model_key = (model or "").strip().lower()
    base_url_key = (base_url or "").strip().lower()
    if provider_key in {"openai", "anthropic", "google"}:
        return True
    if provider_key != "openai-chat":
        return None
    if "deepseek" in base_url_key or model_key.startswith("deepseek"):
        return False
    return None
