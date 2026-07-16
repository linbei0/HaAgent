"""
haagent/models/config/connections.py - 模型连接领域类型

只定义连接记录、配置快照及凭据操作；JSON 和模型解析分别由专用模块负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from haagent.models.config.credentials import (
    CredentialRecord,
    CredentialStatus,
    CredentialStore,
    KeyringCredentialStore,
    credential_status,
    save_connection_insecure_api_key,
    save_connection_keyring_api_key,
)
from haagent.models.model_options import ModelParameterConfig


USER_CONFIG_DIR_NAME = ".haagent"
USER_PROVIDERS_FILE = "providers.json"
USER_SETTINGS_FILE = "settings.json"
SUPPORTED_GATEWAY_PROVIDERS = {"anthropic", "google", "openai", "openai-chat"}
SUPPORTED_RUNTIME_KINDS = {"remote", "ollama", "lm_studio"}
DEFAULT_CREDENTIAL_SOURCE = "keyring"
DEFAULT_CREDENTIAL_STORE: CredentialStore = KeyringCredentialStore()
PROVIDERS_CONFIG_VERSION = 4


class ProviderProfileError(RuntimeError):
    """模型配置、选择或凭据不可用。"""


@dataclass(frozen=True)
class ProviderConnectionRecord:
    id: str
    name: str
    provider_id: str
    provider_name: str
    gateway_provider: str
    base_url: str
    api_key_env: str
    credential_source: str = DEFAULT_CREDENTIAL_SOURCE
    runtime_kind: str = "remote"
    models: dict[str, ModelParameterConfig] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "gateway_provider": self.gateway_provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_source": self.credential_source,
            "runtime_kind": self.runtime_kind,
        }
        if self.models:
            value["models"] = {
                model_id: {
                    **({"options": config.options} if config.options else {}),
                    **({"variants": config.variants} if config.variants else {}),
                }
                for model_id, config in self.models.items()
            }
        return value


@dataclass(frozen=True)
class ProvidersConfigSnapshot:
    path: Path
    records: tuple[ProviderConnectionRecord, ...]
    digest: str
    load_error: str | None = None
    invalid_model_configs: frozenset[tuple[str, str]] = frozenset()
    available_models: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def require_valid(self) -> None:
        if self.load_error is not None:
            raise ProviderProfileError(self.load_error)

    def list_connections(self) -> list[ProviderConnectionRecord]:
        self.require_valid()
        return list(self.records)

    def connection(self, connection_id: str) -> ProviderConnectionRecord:
        self.require_valid()
        record = next((item for item in self.records if item.id == connection_id), None)
        if record is None:
            raise ProviderProfileError(f"provider connection not found: {connection_id}")
        return record

    def diagnostics_for(self, connection_id: str) -> tuple[str, ...]:
        return tuple(
            f"connection={connection_id} model={model_id} is not available in catalog/discovery"
            for item_connection_id, model_id in sorted(self.invalid_model_configs)
            if item_connection_id == connection_id
        )

    def bind_available_models(self, available_models: Mapping[str, set[str]]) -> "ProvidersConfigSnapshot":
        validated = set(available_models)
        invalid = {item for item in self.invalid_model_configs if item[0] not in validated}
        for record in self.records:
            available = available_models.get(record.id)
            if available is not None:
                invalid.update((record.id, model) for model in record.models if model not in available)
        return ProvidersConfigSnapshot(
            path=self.path,
            records=self.records,
            digest=self.digest,
            load_error=self.load_error,
            invalid_model_configs=frozenset(invalid),
            available_models={key: tuple(sorted(value)) for key, value in available_models.items()},
        )


def user_config_dir() -> Path:
    return Path.home() / USER_CONFIG_DIR_NAME


def user_provider_connections_path() -> Path:
    return user_config_dir() / USER_PROVIDERS_FILE


def user_settings_path() -> Path:
    return user_config_dir() / USER_SETTINGS_FILE


def credential_record(connection: ProviderConnectionRecord) -> CredentialRecord:
    return CredentialRecord(
        profile_name=connection.id,
        api_key_env=connection.api_key_env,
        credential_source=connection.credential_source,
        credential_username=f"connection:{connection.id}",
    )


def provider_connection_credential_status(
    connection: ProviderConnectionRecord,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path,
) -> CredentialStatus:
    if connection.credential_source == "none":
        return CredentialStatus(True, "none", "none", None)
    return credential_status(
        credential_record(connection),
        environ=environ,
        credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
        config_dir=config_dir,
    )


def save_connection_api_key(
    connection: ProviderConnectionRecord,
    api_key: str | None,
    *,
    config_dir: Path,
    credential_store: CredentialStore | None = None,
) -> None:
    if api_key is None or not api_key.strip():
        return
    if connection.credential_source == "keyring":
        save_connection_keyring_api_key(
            connection.id,
            api_key,
            credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
        )
        return
    if connection.credential_source == "insecure_file":
        save_connection_insecure_api_key(connection.id, api_key, config_dir=config_dir)
        return
    raise ProviderProfileError(f"{connection.credential_source} credential_source does not allow saving api_key")
