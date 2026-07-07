"""
haagent/models/model_connections.py - 用户级模型连接与选择

读取和写入供应商连接、默认模型选择，并解析运行时模型 profile。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from haagent.models.credentials import (
    CredentialError,
    CredentialRecord,
    CredentialStatus,
    CredentialStore,
    KeyringCredentialStore,
    credential_status,
    resolve_api_key,
    save_connection_insecure_api_key,
    save_connection_keyring_api_key,
)


USER_CONFIG_DIR_NAME = ".haagent"
USER_PROVIDERS_FILE = "providers.json"
USER_SETTINGS_FILE = "settings.json"
SUPPORTED_GATEWAY_PROVIDERS = {"openai", "openai-chat"}
DEFAULT_CREDENTIAL_SOURCE = "keyring"
DEFAULT_CREDENTIAL_STORE: CredentialStore = KeyringCredentialStore()


class ProviderProfileError(RuntimeError):
    """模型连接配置或凭据不可用时抛出。"""


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    credential_source: str
    credential_source_used: str
    api_key: str = field(repr=False)


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

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "gateway_provider": self.gateway_provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "credential_source": self.credential_source,
        }


@dataclass(frozen=True)
class ModelSelection:
    connection_id: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {"connection_id": self.connection_id, "model": self.model}


def user_config_dir() -> Path:
    return Path.home() / USER_CONFIG_DIR_NAME


def user_provider_connections_path() -> Path:
    return user_config_dir() / USER_PROVIDERS_FILE


def user_settings_path() -> Path:
    return user_config_dir() / USER_SETTINGS_FILE


def save_provider_connection_with_key(
    record: ProviderConnectionRecord,
    api_key: str | None,
    *,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> Path:
    has_api_key = api_key is not None and bool(api_key.strip())
    if record.credential_source == "env" and has_api_key:
        raise ProviderProfileError("env credential_source does not allow saving api_key")
    path = save_provider_connection(record, config_dir=config_dir)
    if not has_api_key:
        return path
    if record.credential_source == "keyring":
        save_connection_keyring_api_key(
            record.id,
            api_key,
            credential_store=credential_store,
        )
        return path
    if record.credential_source == "insecure_file":
        save_connection_insecure_api_key(
            record.id,
            api_key,
            config_dir=config_dir or user_config_dir(),
        )
        return path
    raise ProviderProfileError(f"unsupported credential_source in connection: {record.credential_source}")


def save_provider_connection(record: ProviderConnectionRecord, *, config_dir: Path | None = None) -> Path:
    _validate_connection_record(record.to_dict())
    directory = config_dir or user_config_dir()
    path = directory / USER_PROVIDERS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    records = _load_connection_records(path) if path.exists() else []
    updated = False
    next_records = []
    for existing in records:
        if existing.id == record.id:
            next_records.append(record)
            updated = True
        else:
            next_records.append(existing)
    if not updated:
        next_records.append(record)
    _write_json(
        path,
        {
            "version": 2,
            "connections": [connection.to_dict() for connection in next_records],
            "custom_models": [],
        },
    )
    return path


def delete_provider_connection(connection_id: str, *, config_dir: Path | None = None) -> Path:
    if not connection_id.strip():
        raise ProviderProfileError("provider connection id is required")
    directory = config_dir or user_config_dir()
    path = directory / USER_PROVIDERS_FILE
    records = _load_connection_records(path)
    next_records = [record for record in records if record.id != connection_id]
    if len(next_records) == len(records):
        raise ProviderProfileError(f"provider connection not found: {connection_id}")
    _write_json(
        path,
        {
            "version": 2,
            "connections": [connection.to_dict() for connection in next_records],
            "custom_models": [],
        },
    )
    _refresh_active_model_after_connection_delete(directory, connection_id, next_records)
    return path


def load_provider_connection_record(
    connection_id: str,
    *,
    config_path: Path | None = None,
) -> ProviderConnectionRecord:
    records = list_provider_connection_records(config_path=config_path)
    for record in records:
        if record.id == connection_id:
            return record
    raise ProviderProfileError(f"provider connection not found: {connection_id}")


def list_provider_connection_records(
    *,
    config_path: Path | None = None,
) -> list[ProviderConnectionRecord]:
    path = config_path or user_provider_connections_path()
    if not path.exists():
        return []
    return _load_connection_records(path)


def provider_connection_credential_status(
    connection_id: str,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> CredentialStatus:
    config_path = (config_dir / USER_PROVIDERS_FILE) if config_dir is not None else None
    connection = load_provider_connection_record(connection_id, config_path=config_path)
    return credential_status(
        _credential_record(connection),
        environ=environ,
        credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
        config_dir=config_dir,
    )


def load_model_selection_profile(
    selection: ModelSelection,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> ProviderProfile:
    connection_config_path = config_path or (
        (config_dir / USER_PROVIDERS_FILE) if config_dir is not None else None
    )
    connection = load_provider_connection_record(selection.connection_id, config_path=connection_config_path)
    try:
        resolved = resolve_api_key(
            _credential_record(connection),
            environ=environ,
            credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
            config_dir=config_dir or _config_dir_for(config_path),
        )
    except CredentialError as error:
        raise ProviderProfileError(str(error)) from error
    if not resolved.api_key:
        detail = resolved.credential_store_error or f"configured source: {connection.credential_source}"
        raise ProviderProfileError(
            f"API key is not available for connection {selection.connection_id}: "
            f"{detail}; api_key_env={connection.api_key_env}",
        )
    return ProviderProfile(
        name=f"{connection.id}:{selection.model}",
        provider=connection.gateway_provider,
        base_url=connection.base_url,
        model=selection.model,
        api_key_env=connection.api_key_env,
        credential_source=connection.credential_source,
        credential_source_used=resolved.credential_source_used or "",
        api_key=resolved.api_key,
    )


def load_active_model_selection(*, settings_path: Path | None = None, config_dir: Path | None = None) -> ModelSelection:
    directory = config_dir or user_config_dir()
    path = settings_path or directory / USER_SETTINGS_FILE
    if not path.exists():
        raise ProviderProfileError(_setup_required_message())
    settings = _load_settings_record(path)
    active_model = settings.get("active_model")
    if isinstance(active_model, dict):
        connection_id = active_model.get("connection_id")
        model = active_model.get("model")
        if isinstance(connection_id, str) and connection_id.strip() and isinstance(model, str) and model.strip():
            return ModelSelection(connection_id=connection_id, model=model)
    raise ProviderProfileError("settings config must contain active_model")


def save_active_model_selection(selection: ModelSelection, *, config_dir: Path | None = None) -> Path:
    if not selection.connection_id.strip():
        raise ProviderProfileError("active model connection_id is required")
    if not selection.model.strip():
        raise ProviderProfileError("active model model is required")
    directory = config_dir or user_config_dir()
    path = directory / USER_SETTINGS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    settings = _load_settings_record(path) if path.exists() else {}
    settings["active_model"] = selection.to_dict()
    _write_json(path, settings)
    return path


def _credential_record(connection: ProviderConnectionRecord) -> CredentialRecord:
    return CredentialRecord(
        profile_name=connection.id,
        api_key_env=connection.api_key_env,
        credential_source=connection.credential_source,
        credential_username=f"connection:{connection.id}",
    )


def _refresh_active_model_after_connection_delete(
    directory: Path,
    connection_id: str,
    next_records: list[ProviderConnectionRecord],
) -> None:
    settings_path = directory / USER_SETTINGS_FILE
    if not settings_path.exists():
        return
    settings = _load_settings_record(settings_path)
    active_model = settings.get("active_model")
    active_connection_id = active_model.get("connection_id") if isinstance(active_model, dict) else None
    if active_connection_id != connection_id:
        return
    if next_records:
        settings["active_model"] = {
            "connection_id": next_records[0].id,
            "model": str(active_model.get("model", "")) if isinstance(active_model, dict) else "",
        }
    else:
        settings.pop("active_model", None)
    if settings:
        _write_json(settings_path, settings)
    else:
        settings_path.unlink()


def _load_connection_records(config_path: Path) -> list[ProviderConnectionRecord]:
    if not config_path.exists():
        raise ProviderProfileError(f"provider connection config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderProfileError(f"provider connection config is invalid JSON: {config_path}") from error
    if not isinstance(raw, dict):
        raise ProviderProfileError("provider connection config must be a JSON object")
    connections = raw.get("connections")
    if not isinstance(connections, list):
        if "connections" not in raw and isinstance(raw.get("profiles"), list):
            return []
        raise ProviderProfileError("provider connection config must contain connections")
    records = []
    for index, item in enumerate(connections):
        if not isinstance(item, dict):
            raise ProviderProfileError(f"provider connection at index {index} must be an object")
        _validate_connection_record(item)
        records.append(_connection_from_record(item))
    return records


def _connection_from_record(record: dict[str, object]) -> ProviderConnectionRecord:
    return ProviderConnectionRecord(
        id=_required_string(record, "id"),
        name=_required_string(record, "name"),
        provider_id=_required_string(record, "provider_id"),
        provider_name=_required_string(record, "provider_name"),
        gateway_provider=_required_gateway_provider(record),
        base_url=_required_string(record, "base_url"),
        api_key_env=_required_string(record, "api_key_env"),
        credential_source=_credential_source(record),
    )


def _validate_connection_record(record: dict[str, object]) -> None:
    _required_string(record, "id")
    _required_string(record, "name")
    _required_string(record, "provider_id")
    _required_string(record, "provider_name")
    _required_gateway_provider(record)
    _required_string(record, "base_url")
    _required_string(record, "api_key_env")
    _credential_source(record)
    if "api_key" in record:
        raise ProviderProfileError("provider connection must not contain api_key")


def _required_gateway_provider(record: dict[str, object]) -> str:
    provider = _required_string(record, "gateway_provider")
    if provider not in SUPPORTED_GATEWAY_PROVIDERS:
        raise ProviderProfileError(f"unsupported provider in connection: {provider}")
    return provider


def _required_string(record: dict[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError(f"provider connection field is required: {field_name}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_settings_record(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderProfileError(f"settings config is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ProviderProfileError("settings config must be a JSON object")
    return raw


def _setup_required_message() -> str:
    return "未找到默认模型配置，请运行 haagent 后在 TUI 内输入 /connect 配置供应商"


def _credential_source(record: dict[str, object]) -> str:
    value = record.get("credential_source", DEFAULT_CREDENTIAL_SOURCE)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError("provider connection field is required: credential_source")
    if value not in {"env", "keyring", "insecure_file"}:
        raise ProviderProfileError(f"unsupported credential_source in connection: {value}")
    return value


def _config_dir_for(config_path: Path | None) -> Path:
    if config_path is None:
        return user_config_dir()
    return config_path.parent
