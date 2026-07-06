"""
haagent/app/model_profile_usecases.py - 模型配置类应用用例

集中封装 AssistantService 的 profile、catalog 和当前会话模型切换逻辑。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from haagent.models import provider_profile as provider_profile_module
from haagent.models.catalog import DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE, fetch_model_catalog
from haagent.models.credentials import CredentialError
from haagent.models.gateway import ModelCallError
from haagent.models.provider_profile import (
    ProviderProfileError,
    ProviderProfileRecord,
    delete_provider_profile,
    list_provider_profile_records,
    load_active_profile_name,
    load_provider_profile,
    load_provider_profile_record,
    provider_profile_credential_status,
    save_active_profile,
    save_provider_profile_with_key,
    user_config_dir,
)

if TYPE_CHECKING:
    from haagent.app.assistant_service import (
        AssistantModelProfile,
        AssistantModelTestResult,
        AssistantService,
        AssistantSessionStatus,
        ModelProfileConfigureRequest,
    )


def list_model_profiles(service: "AssistantService") -> list["AssistantModelProfile"]:
    try:
        active_profile_name = load_active_profile_name()
    except ProviderProfileError:
        active_profile_name = None
    session_status = service.current_session()
    current_profile_name = session_status.model_profile_name if session_status is not None else None
    profiles: list["AssistantModelProfile"] = []
    for record in list_provider_profile_records():
        credential = provider_profile_credential_status(
            record.name,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        profiles.append(
            service.model_profile_cls(
                name=record.name,
                provider=record.provider,
                base_url=record.base_url,
                model=record.model,
                api_key_env=record.api_key_env,
                credential_source=record.credential_source,
                active=record.name == active_profile_name,
                credential_available=credential.api_key_available,
                credential_source_used=credential.credential_source_used,
                capability=service.gateway_capability_for_profile(record),
                current_session=record.name == current_profile_name,
            )
        )
    return profiles


def set_default_model_profile(service: "AssistantService", profile_name: str) -> None:
    load_provider_profile_record(profile_name)
    save_active_profile(profile_name, config_dir=user_config_dir())


def configure_model_profile(
    service: "AssistantService",
    request: "ModelProfileConfigureRequest",
) -> ProviderProfileRecord:
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
        raise service.error_cls(str(error)) from error
    return record


def delete_model_profile_for_user(service: "AssistantService", profile_name: str) -> None:
    try:
        delete_provider_profile(profile_name, config_dir=user_config_dir())
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


def test_model_profile(service: "AssistantService", profile_name: str) -> "AssistantModelTestResult":
    try:
        profile = load_provider_profile(
            profile_name,
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
        record = service.load_profile_record_for_result(profile_name)
        return service.model_test_result_cls(
            ok=False,
            profile_name=profile_name,
            provider=record.provider if record is not None else "",
            model=record.model if record is not None else "",
            message=service.redact_secret_text(str(error), service.secret_candidates(service.environ)),
        )


def switch_current_session_model(service: "AssistantService", profile_name: str) -> "AssistantSessionStatus":
    try:
        profile = load_provider_profile(
            profile_name,
            environ=service.environ,
            config_dir=user_config_dir(),
        )
        if service._session is None:
            service._pending_model_profile_name = profile.name
            return service.session_status_cls(
                session_id="pending",
                workspace_root=service.workspace_root,
                runs_root=service.runs_root,
                session_path=service.runs_root,
                turn_count=0,
                max_turns=service.max_turns,
                provider=profile.provider,
                model_profile_name=profile.name,
                model=profile.model,
                base_url=profile.base_url,
                web_enabled=service.enable_web,
                permission_mode="request_approval",
            )
        gateway = service.gateway_factory(profile)
        service._session.switch_model_gateway(
            profile_name=profile.name,
            provider=profile.provider,
            model=profile.model,
            base_url=profile.base_url,
            gateway=gateway,
        )
    except (ProviderProfileError, service.chat_session_error_cls) as error:
        raise service.error_cls(str(error)) from error
    return service._session_status(service._session)
