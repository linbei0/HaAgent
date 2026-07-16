"""
haagent/models/model_resolution.py - 模型运行时解析

把 ModelRef、配置快照和凭据解析为一次不可变 ResolvedModel。
"""

from __future__ import annotations

from collections.abc import Mapping

from haagent.models.config.connections import (
    DEFAULT_CREDENTIAL_STORE,
    ProviderProfileError,
    ProvidersConfigSnapshot,
    credential_record,
)
from haagent.models.config.credentials import CredentialError, CredentialStore, resolve_api_key
from haagent.models.model_options import ModelOptionsError
from haagent.models.model_ref import ModelRef, ResolvedCredential, ResolvedModel
from haagent.models.model_settings import ModelSettings


def resolve_model(
    ref: ModelRef,
    *,
    snapshot: ProvidersConfigSnapshot,
    environ: Mapping[str, str],
    credential_store: CredentialStore | None = None,
) -> ResolvedModel:
    connection = snapshot.connection(ref.connection_id)
    if (ref.connection_id, ref.model) in snapshot.invalid_model_configs:
        raise ProviderProfileError(
            f"configured model is not available in catalog/discovery: connection={ref.connection_id} model={ref.model}",
        )
    config = connection.models.get(ref.model)
    if config is None:
        if ref.variant is not None:
            raise ProviderProfileError(
                f"model variant is not available: connection={ref.connection_id} model={ref.model} variant={ref.variant}",
            )
        settings = ModelSettings.empty()
    else:
        if ref.variant is not None and ref.variant not in config.variants:
            raise ProviderProfileError(
                f"model variant is not available: connection={ref.connection_id} model={ref.model} variant={ref.variant}",
            )
        settings = ModelSettings.from_options(config.options, configured=bool(config.options or config.variants))
        if ref.variant is not None:
            settings = settings.resolve(config.variants[ref.variant])
    if connection.credential_source == "none":
        credential = ResolvedCredential("", connection.api_key_env, "none", "none")
    else:
        try:
            resolved = resolve_api_key(
                credential_record(connection),
                environ=environ,
                credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
                config_dir=snapshot.path.parent,
            )
        except CredentialError as error:
            raise ProviderProfileError(str(error)) from error
        if not resolved.api_key:
            detail = resolved.credential_store_error or f"configured source: {connection.credential_source}"
            raise ProviderProfileError(
                f"API key is not available for connection {ref.connection_id}: {detail}; api_key_env={connection.api_key_env}",
            )
        credential = ResolvedCredential(
            resolved.api_key,
            connection.api_key_env,
            connection.credential_source,
            resolved.credential_source_used or "",
        )
    return ResolvedModel(
        ref=ref,
        provider=connection.gateway_provider,
        base_url=connection.base_url,
        runtime_kind=connection.runtime_kind,
        settings=settings,
        credential=credential,
    )
