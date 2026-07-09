"""
haagent/app/model_connection_usecases.py - 模型连接与选择用例

集中封装模型目录、供应商连接、默认模型选择和当前会话模型切换逻辑。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haagent.models import model_connections as model_connections_module
from haagent.models.catalog import DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE, fetch_model_catalog
from haagent.models.credentials import CredentialError
from haagent.models.types import ModelCallError
from haagent.models.model_connections import (
    ModelSelection,
    ProviderConnectionRecord,
    ProviderProfileError,
    delete_provider_connection,
    list_provider_connection_records,
    load_active_model_selection,
    load_model_selection_profile,
    provider_connection_credential_status,
    save_active_model_selection,
    save_provider_connection_with_key,
    user_config_dir,
)

if TYPE_CHECKING:
    from haagent.app.assistant_service import (
        AssistantModelTestResult,
        AssistantService,
        AssistantSessionStatus,
        ModelConnectionConfigureRequest,
        ModelSelectionRequest,
    )


def list_model_connections(service: "AssistantService"):
    connections = []
    for record in list_provider_connection_records(config_path=user_config_dir() / "providers.json"):
        credential = provider_connection_credential_status(
            record.id,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        connections.append(
            service.model_connection_cls(
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
            )
        )
    return connections


def configure_model_connection(
    service: "AssistantService",
    request: "ModelConnectionConfigureRequest",
) -> ProviderConnectionRecord:
    record = ProviderConnectionRecord(
        id=request.id,
        name=request.name,
        provider_id=request.provider_id,
        provider_name=request.provider_name,
        gateway_provider=request.gateway_provider,
        base_url=request.base_url,
        api_key_env=request.api_key_env,
        credential_source=request.credential_source,
    )
    try:
        save_provider_connection_with_key(
            record,
            request.api_key,
            credential_store=model_connections_module.DEFAULT_CREDENTIAL_STORE,
            config_dir=user_config_dir(),
        )
    except (ProviderProfileError, CredentialError) as error:
        raise service.error_cls(str(error)) from error
    return record


def set_default_model_selection(service: "AssistantService", request: "ModelSelectionRequest") -> None:
    selection = ModelSelection(connection_id=request.connection_id, model=request.model)
    try:
        load_model_selection_profile(
            selection,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        save_active_model_selection(selection, config_dir=user_config_dir())
    except (ProviderProfileError, CredentialError) as error:
        raise service.error_cls(str(error)) from error


def delete_model_connection_for_user(service: "AssistantService", connection_id: str) -> None:
    try:
        delete_provider_connection(connection_id, config_dir=user_config_dir())
    except ProviderProfileError as error:
        raise service.error_cls(str(error)) from error


def refresh_model_catalog(service: "AssistantService", *, transport=None):
    try:
        return fetch_model_catalog(transport=transport, force_refresh=True)
    except Exception as error:
        raise service.error_cls(str(error)) from error


def get_model_catalog(service: "AssistantService", *, transport=None):
    try:
        return fetch_model_catalog(
            transport=transport,
            max_cache_age=DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE,
        )
    except Exception as error:
        raise service.error_cls(str(error)) from error


def test_model_connection(
    service: "AssistantService",
    connection_id: str,
    *,
    model: str | None = None,
) -> "AssistantModelTestResult":
    try:
        if model is None:
            active_selection = load_active_model_selection(config_dir=user_config_dir())
            model = active_selection.model
        selection = ModelSelection(connection_id=connection_id, model=model)
        profile = load_model_selection_profile(
            selection,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        gateway = service.gateway_factory(profile)
        response = gateway.generate(
            [{"role": "user", "content": "Reply with OK."}],
            [],
        )
        return service.model_test_result_cls(
            ok=True,
            profile_name=profile.name,
            provider=profile.provider,
            model=profile.model,
            message=service.redact_secret_text(response.content, [profile.api_key]),
        )
    except (ProviderProfileError, CredentialError, ModelCallError) as error:
        return service.model_test_result_cls(
            ok=False,
            profile_name=connection_id,
            provider="",
            model="",
            message=service.redact_secret_text(str(error), service.secret_candidates(service.environ)),
        )


def switch_current_session_model_selection(
    service: "AssistantService",
    request: "ModelSelectionRequest",
) -> "AssistantSessionStatus":
    selection = ModelSelection(connection_id=request.connection_id, model=request.model)
    try:
        profile = load_model_selection_profile(
            selection,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        _save_default_selection_if_missing(selection)
        if service._session is None:
            service._pending_model_selection = selection
            return service.session_status_cls(
                session_id="pending",
                workspace_root=service.workspace_root,
                runs_root=service.runs_root,
                session_path=service.runs_root,
                turn_count=0,
                max_turns=service.max_turns,
                provider=profile.provider,
                model_profile_name=profile.name,
                model_connection_id=selection.connection_id,
                model=profile.model,
                base_url=profile.base_url,
                web_enabled=service.enable_web,
                permission_mode="request_approval",
            )
        gateway = service.gateway_factory(profile)
        service._session.switch_model_gateway(
            profile_name=profile.name,
            model_connection_id=selection.connection_id,
            provider=profile.provider,
            model=profile.model,
            base_url=profile.base_url,
            gateway=gateway,
        )
    except (ProviderProfileError, service.chat_session_error_cls) as error:
        raise service.error_cls(str(error)) from error
    return service._session_status(service._session)


def _save_default_selection_if_missing(selection: ModelSelection) -> None:
    try:
        load_active_model_selection(config_dir=user_config_dir())
    except ProviderProfileError:
        save_active_model_selection(selection, config_dir=user_config_dir())
