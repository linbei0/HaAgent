"""
haagent/app/model_connection_usecases.py - 模型连接应用用例

只编排 ModelRuntime；不读取或解析 providers.json/settings.json。
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
from haagent.models.catalog import DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE, CatalogFetchResult, CatalogTransport, fetch_model_catalog
from haagent.models.config.credentials import CredentialError
from haagent.models.local_runtime import LocalRuntimeDiscovery, LocalRuntimeModel, discover_local_runtimes
from haagent.models.config.connections import ProviderConnectionRecord, ProviderProfileError, save_connection_api_key
from haagent.models.model_ref import ModelInvocation, ModelRef
from haagent.models.types import ModelCallError
from haagent.runtime.session.package import ChatSessionError


class AssistantModels:
    def __init__(self, context: AssistantContext) -> None:
        self._context = context

    @property
    def _runtime(self):
        assert self._context.model_runtime is not None
        return self._context.model_runtime

    def list_connections(self) -> list[AssistantModelConnection]:
        snapshot = self._runtime.snapshot
        return [
            AssistantModelConnection(
                id=record.id,
                name=record.name,
                provider_id=record.provider_id,
                provider_name=record.provider_name,
                gateway_provider=record.gateway_provider,
                base_url=record.base_url,
                api_key_env=record.api_key_env,
                credential_source=record.credential_source,
                credential_available=self._runtime.credential_status(record.id).api_key_available,
                credential_source_used=self._runtime.credential_status(record.id).credential_source_used,
                runtime_kind=record.runtime_kind,
                model_config_diagnostics=snapshot.diagnostics_for(record.id),
            )
            for record in self._runtime.list_connections()
        ]

    def list_choices(self):
        return self._runtime.list_choices()

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
            snapshot = self._runtime.config_store.save_connection(record, expected_digest=self._runtime.snapshot.digest)
            save_connection_api_key(record, request.api_key, config_dir=snapshot.path.parent)
            self._runtime.snapshot = snapshot
        except (ProviderProfileError, CredentialError) as error:
            raise AssistantServiceError(str(error)) from error
        self._context.status_generation += 1
        return record

    def delete_connection(self, connection_id: str) -> None:
        try:
            snapshot = self._runtime.config_store.delete_connection(
                connection_id,
                expected_digest=self._runtime.snapshot.digest,
            )
            self._runtime.selection_store.remove_connection(
                connection_id,
                [record.id for record in snapshot.records],
            )
            self._runtime.snapshot = snapshot
        except ProviderProfileError as error:
            raise AssistantServiceError(str(error)) from error
        self._context.status_generation += 1

    def discover_local_runtimes(self) -> tuple[LocalRuntimeDiscovery, LocalRuntimeDiscovery]:
        discoveries = discover_local_runtimes(environ=self._context.environ)
        available: dict[str, set[str]] = {}
        for connection in self._runtime.snapshot.records:
            discovery = next(
                (item for item in discoveries if item.runtime_kind == connection.runtime_kind and item.status == "available"),
                None,
            )
            if discovery is not None:
                available[connection.id] = {model.id for model in discovery.models}
        self._runtime.snapshot = self._runtime.snapshot.bind_available_models(available)
        return discoveries

    def save_local_model(self, discovery: LocalRuntimeDiscovery, model: LocalRuntimeModel) -> ModelRef:
        if discovery.status != "available" or model not in discovery.models:
            raise AssistantServiceError("local model must come from an available discovery result")
        connection_id = f"local-{discovery.runtime_kind.replace('_', '-')}"
        lm_studio_key = str(self._context.environ.get("LM_STUDIO_API_KEY", "")).strip()
        source = "env" if discovery.runtime_kind == "lm_studio" and lm_studio_key else "none"
        record = ProviderConnectionRecord(
            id=connection_id,
            name="Ollama" if discovery.runtime_kind == "ollama" else "LM Studio",
            provider_id=discovery.runtime_kind.replace("_", "-"),
            provider_name="Ollama" if discovery.runtime_kind == "ollama" else "LM Studio",
            gateway_provider="openai",
            base_url=discovery.base_url,
            api_key_env="LM_STUDIO_API_KEY" if source == "env" else "",
            credential_source=source,
            runtime_kind=discovery.runtime_kind,
        )
        self._runtime.snapshot = self._runtime.config_store.save_connection(
            record,
            expected_digest=self._runtime.snapshot.digest,
        ).bind_available_models({connection_id: {item.id for item in discovery.models}})
        self._context.status_generation += 1
        return ModelRef(connection_id, model.id)

    def set_fallback_selection(self, request: ModelSelectionRequest, *, cloud_fallback_consent: bool = False) -> None:
        ref = _ref(request)
        self._runtime.resolve(ref)
        self._runtime.set_fallback(ref, cloud_consent=cloud_fallback_consent)

    def set_default_selection(self, request: ModelSelectionRequest) -> None:
        ref = _ref(request)
        try:
            self._runtime.resolve(ref)
            self._runtime.set_active(ref)
        except (ProviderProfileError, CredentialError) as error:
            raise AssistantServiceError(str(error)) from error
        self._context.status_generation += 1

    def refresh_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        return self._catalog(transport=transport, force=True)

    def get_catalog(self, *, transport: CatalogTransport | None = None) -> CatalogFetchResult:
        return self._catalog(transport=transport, force=False)

    def _catalog(self, *, transport: CatalogTransport | None, force: bool) -> CatalogFetchResult:
        try:
            result = fetch_model_catalog(
                transport=transport,
                force_refresh=force,
                max_cache_age=None if force else DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
            )
            bind_catalog_snapshot(self._context, result)
            return result
        except Exception as error:
            raise AssistantServiceError(str(error)) from error

    def test_connection(self, connection_id: str, *, model: str | None = None) -> AssistantModelTestResult:
        try:
            selected = self._runtime.selection_store.load_active()
            ref = ModelRef(connection_id, model or selected.model)
            resolved = self._runtime.resolve(ref)
            response = self._runtime.create_gateway(ref).generate(
                ModelInvocation([{"role": "user", "content": "Reply with OK."}], [], resolved.settings),
            )
            return AssistantModelTestResult(
                True,
                f"{ref.connection_id}:{ref.model}",
                resolved.provider,
                ref.model,
                _redact_secret_text(response.content, [resolved.credential.api_key]),
            )
        except (ProviderProfileError, CredentialError, ModelCallError) as error:
            return AssistantModelTestResult(False, connection_id, "", "", _redact_secret_text(str(error), list(self._context.environ.values())))

    def switch_current_session_selection(self, request: ModelSelectionRequest) -> AssistantSessionStatus:
        ref = _ref(request)
        try:
            resolved = self._runtime.resolve(ref)
            try:
                self._runtime.selection_store.load_active()
            except ProviderProfileError:
                self._runtime.set_active(ref)
            if self._context.session is None:
                self._context.pending_model_selection = ref
                self._context.status_generation += 1
                return AssistantSessionStatus(
                    session_id="pending",
                    workspace_root=self._context.workspace_root,
                    runs_root=self._context.runs_root,
                    session_path=self._context.runs_root,
                    turn_count=0,
                    max_turns=self._context.max_turns,
                    provider=resolved.provider,
                    model_connection_id=ref.connection_id,
                    model=ref.model,
                    model_variant=ref.variant,
                    base_url=resolved.base_url,
                    web_enabled=self._context.enable_web,
                )
            self._context.session.switch_model_gateway(ref, self._runtime.create_gateway(ref))
            self._context.status_generation += 1
        except (ProviderProfileError, ChatSessionError) as error:
            raise AssistantServiceError(str(error)) from error
        from haagent.app.session_usecases import session_status
        return session_status(self._context.session)


def bind_catalog_snapshot(context: AssistantContext, catalog: CatalogFetchResult) -> None:
    assert context.model_runtime is not None
    providers = {provider.id: provider for provider in catalog.providers}
    available = {
        connection.id: {model.id for model in providers[connection.provider_id].models}
        for connection in context.model_runtime.snapshot.records
        if connection.runtime_kind == "remote" and connection.provider_id in providers
    }
    context.model_runtime.snapshot = context.model_runtime.snapshot.bind_available_models(available)


def _ref(request: ModelSelectionRequest) -> ModelRef:
    return ModelRef(request.connection_id, request.model, request.variant)


def _redact_secret_text(text: str, secrets) -> str:
    for secret in secrets:
        if isinstance(secret, str) and secret:
            text = text.replace(secret, "[REDACTED]")
    return text
