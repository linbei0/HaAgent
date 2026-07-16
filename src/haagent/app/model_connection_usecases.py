"""
haagent/app/model_connection_usecases.py - 模型连接与选择应用 Module

管理供应商连接、credential 状态、模型目录、连接测试和当前 session 模型切换。
"""

from __future__ import annotations

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantModelConnection,
    AssistantModelTestResult,
    AssistantServiceError,
    AssistantSessionStatus,
    ModelConnectionConfigureRequest,
    ModelSelectionRequest,
)
from haagent.models.catalog import (
    DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
    CatalogFetchResult,
    CatalogTransport,
    fetch_model_catalog,
)
from haagent.models import model_connections as model_connections_module
from haagent.models.credentials import CredentialError
from haagent.models.model_connections import (
    ModelSelection,
    ProviderConnectionRecord,
    ProviderProfileError,
    delete_provider_connection,
    load_active_model_selection,
    load_model_selection_profile,
    load_providers_config_snapshot,
    provider_connection_credential_status,
    save_active_model_selection,
    save_fallback_model_selection,
    save_provider_connection_with_key,
)
from haagent.models.local_runtime import LocalRuntimeDiscovery, LocalRuntimeModel, discover_local_runtimes
from haagent.models.types import ModelCallError
from haagent.runtime.session.package import ChatSessionError


class AssistantModels:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context

    def list_connections(self) -> list[AssistantModelConnection]:
        connections = []
        snapshot = self._context.providers_snapshot
        for record in snapshot.list_connections():
            credential = provider_connection_credential_status(
                record,
                environ=self._context.environ,
                config_dir=snapshot.path.parent,
            )
            connections.append(
                AssistantModelConnection(
                    id=record.id,
                    name=record.name,
                    provider_id=record.provider_id,
                    provider_name=record.provider_name,
                    gateway_provider=record.gateway_provider,
                    base_url=record.base_url,
                    api_key_env=record.api_key_env,
                    credential_source=record.credential_source,
                    credential_available=credential.api_key_available,
                    credential_source_used=credential.credential_source_used,
                    runtime_kind=record.runtime_kind,
                    model_config_diagnostics=snapshot.diagnostics_for(record.id),
                ),
            )
        return connections

    def configure_connection(self, request: ModelConnectionConfigureRequest) -> ProviderConnectionRecord:
        record = ProviderConnectionRecord(
            id=request.id,
            name=request.name,
            provider_id=request.provider_id,
            provider_name=request.provider_name,
            gateway_provider=request.gateway_provider,
            base_url=request.base_url,
            api_key_env=request.api_key_env,
            credential_source=request.credential_source,
            runtime_kind=request.runtime_kind,
        )
        try:
            snapshot = self._context.providers_snapshot
            path = save_provider_connection_with_key(
                record,
                request.api_key,
                snapshot=snapshot,
                credential_store=model_connections_module.DEFAULT_CREDENTIAL_STORE,
            )
        except (ProviderProfileError, CredentialError) as error:
            raise AssistantServiceError(str(error)) from error
        _replace_snapshot_from_path(self._context, path)
        self._context.status_generation += 1
        return record

    def discover_local_runtimes(self) -> tuple[LocalRuntimeDiscovery, LocalRuntimeDiscovery]:
        discoveries = discover_local_runtimes(environ=self._context.environ)
        snapshot = self._context.providers_snapshot
        available: dict[str, set[str]] = {}
        for connection in snapshot.records:
            discovery = next(
                (
                    item
                    for item in discoveries
                    if item.runtime_kind == connection.runtime_kind and item.status == "available"
                ),
                None,
            )
            if discovery is not None:
                available[connection.id] = {model.id for model in discovery.models}
        self._context.providers_snapshot = snapshot.bind_available_models(available)
        return discoveries

    def save_local_model(
        self,
        discovery: LocalRuntimeDiscovery,
        model: LocalRuntimeModel,
    ) -> ModelSelection:
        if discovery.status != "available" or model not in discovery.models:
            raise AssistantServiceError("local model must come from an available discovery result")
        runtime_name = "Ollama" if discovery.runtime_kind == "ollama" else "LM Studio"
        connection_id = f"local-{discovery.runtime_kind.replace('_', '-')}"
        lm_studio_key = str(self._context.environ.get("LM_STUDIO_API_KEY", "")).strip()
        credential_source = "env" if discovery.runtime_kind == "lm_studio" and lm_studio_key else "none"
        record = ProviderConnectionRecord(
            id=connection_id,
            name=runtime_name,
            provider_id=discovery.runtime_kind.replace("_", "-"),
            provider_name=runtime_name,
            gateway_provider="openai",
            base_url=discovery.base_url,
            api_key_env="LM_STUDIO_API_KEY" if credential_source == "env" else "",
            credential_source=credential_source,
            runtime_kind=discovery.runtime_kind,
        )
        snapshot = self._context.providers_snapshot
        path = save_provider_connection_with_key(
            record,
            None,
            snapshot=snapshot,
        )
        _replace_snapshot_from_path(self._context, path)
        self._context.providers_snapshot = self._context.providers_snapshot.bind_available_models(
            {connection_id: {item.id for item in discovery.models}},
        )
        self._context.status_generation += 1
        return ModelSelection(connection_id=connection_id, model=model.id)

    def set_fallback_selection(
        self,
        request: ModelSelectionRequest,
        *,
        cloud_fallback_consent: bool = False,
    ) -> None:
        selection = ModelSelection(
            connection_id=request.connection_id,
            model=request.model,
            variant=request.variant,
        )
        _load_profile(self._context, selection)
        save_fallback_model_selection(
            selection,
            cloud_fallback_consent=cloud_fallback_consent,
            config_dir=self._context.providers_snapshot.path.parent,
        )

    def set_default_selection(self, request: ModelSelectionRequest) -> None:
        selection = ModelSelection(
            connection_id=request.connection_id,
            model=request.model,
            variant=request.variant,
        )
        try:
            _load_profile(self._context, selection)
            save_active_model_selection(
                selection,
                config_dir=self._context.providers_snapshot.path.parent,
            )
        except (ProviderProfileError, CredentialError) as error:
            raise AssistantServiceError(str(error)) from error
        self._context.status_generation += 1

    def list_model_variants(self, connection_id: str, model_id: str) -> list[str]:
        try:
            snapshot = self._context.providers_snapshot
            snapshot.require_valid()
            if (connection_id, model_id) in snapshot.invalid_model_configs:
                raise ProviderProfileError(
                    f"configured model is not available in catalog/discovery: "
                    f"connection={connection_id} model={model_id}",
                )
            config = snapshot.connection(connection_id).models.get(model_id)
            return [] if config is None else list(config.variants)
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error

    def delete_connection(self, connection_id: str) -> None:
        try:
            snapshot = self._context.providers_snapshot
            path = delete_provider_connection(
                connection_id,
                snapshot=snapshot,
            )
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error
        _replace_snapshot_from_path(self._context, path)
        self._context.status_generation += 1

    def refresh_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        try:
            result = fetch_model_catalog(transport=transport, force_refresh=True)
            bind_catalog_snapshot(self._context, result)
            return result
        except Exception as error:
            raise AssistantServiceError(str(error)) from error

    def get_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        try:
            result = fetch_model_catalog(
                transport=transport,
                max_cache_age=DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
            )
            bind_catalog_snapshot(self._context, result)
            return result
        except Exception as error:
            raise AssistantServiceError(str(error)) from error

    def test_connection(self, connection_id: str, *, model: str | None = None) -> AssistantModelTestResult:
        try:
            if model is None:
                model = load_active_model_selection(
                    config_dir=self._context.providers_snapshot.path.parent,
                ).model
            selection = ModelSelection(connection_id=connection_id, model=model)
            profile = _load_profile(self._context, selection)
            response = self._context.gateway_factory(profile).generate(
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
            return AssistantModelTestResult(
                ok=False,
                profile_name=connection_id,
                provider="",
                model="",
                message=_redact_secret_text(str(error), _secret_candidates(self._context.environ)),
            )

    def switch_current_session_selection(self, request: ModelSelectionRequest) -> AssistantSessionStatus:
        selection = ModelSelection(
            connection_id=request.connection_id,
            model=request.model,
            variant=request.variant,
        )
        try:
            profile = _load_profile(self._context, selection)
            _save_default_selection_if_missing(
                selection,
                config_dir=self._context.providers_snapshot.path.parent,
            )
            if self._context.session is None:
                self._context.pending_model_selection = selection
                # 无 session 时 status 读 active selection；抬 generation 避免凭据/默认模型旧缓存。
                self._context.status_generation += 1
                return AssistantSessionStatus(
                    session_id="pending",
                    workspace_root=self._context.workspace_root,
                    runs_root=self._context.runs_root,
                    session_path=self._context.runs_root,
                    turn_count=0,
                    max_turns=self._context.max_turns,
                    provider=profile.provider,
                    model_profile_name=profile.name,
                    model_connection_id=selection.connection_id,
                    model=profile.model,
                    model_variant=selection.variant,
                    base_url=profile.base_url,
                    web_enabled=self._context.enable_web,
                    permission_mode="request_approval",
                )
            gateway = self._context.gateway_factory(profile)
            self._context.session.switch_model_gateway(
                profile_name=profile.name,
                model_connection_id=selection.connection_id,
                model=profile.model,
                base_url=profile.base_url or "",
                gateway=gateway,
                model_variant=selection.variant,
            )
            self._context.status_generation += 1
        except (ProviderProfileError, ChatSessionError) as error:
            raise AssistantServiceError(str(error)) from error
        from haagent.app.session_usecases import session_status

        return session_status(self._context.session)


def _save_default_selection_if_missing(selection: ModelSelection, *, config_dir) -> None:
    try:
        load_active_model_selection(config_dir=config_dir)
    except ProviderProfileError:
        save_active_model_selection(selection, config_dir=config_dir)


def _replace_snapshot_from_path(context: AssistantContext, path) -> None:
    snapshot = load_providers_config_snapshot(path)
    snapshot.require_valid()
    context.providers_snapshot = snapshot


def _load_profile(context: AssistantContext, selection: ModelSelection):
    return load_model_selection_profile(
        selection,
        snapshot=context.providers_snapshot,
        environ=context.environ,
    )


def bind_catalog_snapshot(context: AssistantContext, catalog: CatalogFetchResult) -> None:
    snapshot = context.providers_snapshot
    available: dict[str, set[str]] = {}
    providers = {provider.id: provider for provider in catalog.providers}
    for connection in snapshot.records:
        if connection.runtime_kind != "remote":
            continue
        provider = providers.get(connection.provider_id)
        if provider is not None:
            available[connection.id] = {model.id for model in provider.models}
    context.providers_snapshot = snapshot.bind_available_models(available)


def _secret_candidates(environ) -> list[str]:
    return [value for value in environ.values() if isinstance(value, str) and value.strip()]


def _redact_secret_text(text: str, secrets: list[str]) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted
