"""
haagent/models/model_connections.py - 用户级模型连接与选择

读取和写入供应商连接、默认模型选择，并解析运行时模型 profile。
providers.json v4 支持 per-model options/variants；写入时保留未修改连接的完整 models。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

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
from haagent.models.model_options import (
    ModelOptionsError,
    ModelParameterConfig,
    ResolvedModelRequestConfig,
    parse_connection_models,
    resolve_model_request_config,
)


USER_CONFIG_DIR_NAME = ".haagent"
USER_PROVIDERS_FILE = "providers.json"
USER_SETTINGS_FILE = "settings.json"
SUPPORTED_GATEWAY_PROVIDERS = {"anthropic", "google", "openai", "openai-chat"}
SUPPORTED_RUNTIME_KINDS = {"remote", "ollama", "lm_studio"}
DEFAULT_CREDENTIAL_SOURCE = "keyring"
DEFAULT_CREDENTIAL_STORE: CredentialStore = KeyringCredentialStore()
PROVIDERS_CONFIG_VERSION = 4


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
    request_config: ResolvedModelRequestConfig
    runtime_kind: str = "remote"
    variant: str | None = None


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
        payload: dict[str, Any] = {
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
            payload["models"] = {
                model_id: _model_parameter_config_to_dict(config)
                for model_id, config in self.models.items()
            }
        return payload


@dataclass(frozen=True)
class ModelSelection:
    connection_id: str
    model: str
    variant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"connection_id": self.connection_id, "model": self.model}
        if self.variant is not None:
            payload["variant"] = self.variant
        return payload


@dataclass(frozen=True)
class ModelRoute:
    primary: ModelSelection
    fallback: ModelSelection | None
    cloud_fallback_consent: bool


@dataclass(frozen=True)
class ProvidersConfigSnapshot:
    """单个 AssistantService 生命周期内共享的 providers.json 不可变快照。"""

    path: Path
    records: tuple[ProviderConnectionRecord, ...]
    digest: str
    load_error: str | None = None
    invalid_model_configs: frozenset[tuple[str, str]] = frozenset()

    def require_valid(self) -> None:
        if self.load_error is not None:
            raise ProviderProfileError(self.load_error)

    def list_connections(self) -> list[ProviderConnectionRecord]:
        self.require_valid()
        return list(self.records)

    def connection(self, connection_id: str) -> ProviderConnectionRecord:
        self.require_valid()
        for record in self.records:
            if record.id == connection_id:
                return record
        raise ProviderProfileError(f"provider connection not found: {connection_id}")

    def diagnostics_for(self, connection_id: str) -> tuple[str, ...]:
        return tuple(
            f"connection={connection_id} model={model_id} is not available in catalog/discovery"
            for item_connection_id, model_id in sorted(self.invalid_model_configs)
            if item_connection_id == connection_id
        )

    def bind_available_models(
        self,
        available_models: Mapping[str, set[str]],
    ) -> "ProvidersConfigSnapshot":
        """目录/发现结果可用后原子绑定；没有结果的连接保持未判定。"""

        validated_connections = set(available_models)
        invalid: set[tuple[str, str]] = {
            item for item in self.invalid_model_configs if item[0] not in validated_connections
        }
        for record in self.records:
            available = available_models.get(record.id)
            if available is None:
                continue
            for model_id in record.models:
                if model_id in available:
                    continue
                invalid.add((record.id, model_id))
        return ProvidersConfigSnapshot(
            path=self.path,
            records=self.records,
            digest=self.digest,
            load_error=self.load_error,
            invalid_model_configs=frozenset(invalid),
        )


def load_providers_config_snapshot(path: Path | None = None) -> ProvidersConfigSnapshot:
    """读取一次 providers.json；错误也固化到快照，避免后续静默回退。"""

    config_path = path or user_provider_connections_path()
    if not config_path.exists():
        return ProvidersConfigSnapshot(path=config_path, records=(), digest=_snapshot_digest(b""))
    try:
        raw_bytes = config_path.read_bytes()
    except OSError as error:
        return ProvidersConfigSnapshot(
            path=config_path,
            records=(),
            digest=_snapshot_digest(b""),
            load_error=f"cannot read provider connection config: {config_path}: {error}",
        )
    try:
        records = tuple(_parse_connection_records(raw_bytes.decode("utf-8"), config_path))
    except UnicodeDecodeError:
        load_error = f"provider connection config must be UTF-8: {config_path}"
    except ProviderProfileError as error:
        load_error = str(error)
    else:
        return ProvidersConfigSnapshot(
            path=config_path,
            records=records,
            digest=_snapshot_digest(raw_bytes),
        )
    return ProvidersConfigSnapshot(
        path=config_path,
        records=(),
        digest=_snapshot_digest(raw_bytes),
        load_error=load_error,
    )


def _snapshot_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:16]


def ensure_providers_snapshot_current(snapshot: ProvidersConfigSnapshot) -> None:
    """写入前检测外部修改，避免旧快照覆盖用户刚编辑的配置。"""

    try:
        current = snapshot.path.read_bytes() if snapshot.path.exists() else b""
    except OSError as error:
        raise ProviderProfileError(
            f"cannot verify provider connection config before writing: {snapshot.path}: {error}",
        ) from error
    if _snapshot_digest(current) != snapshot.digest:
        raise ProviderProfileError(
            f"providers.json changed after HaAgent started: {snapshot.path}. "
            "Restart HaAgent before changing connections.",
        )


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
    snapshot: ProvidersConfigSnapshot,
    credential_store: CredentialStore | None = None,
) -> Path:
    has_api_key = api_key is not None and bool(api_key.strip())
    if record.credential_source in {"env", "none"} and has_api_key:
        raise ProviderProfileError(
            f"{record.credential_source} credential_source does not allow saving api_key",
        )
    path = save_provider_connection(
        record,
        snapshot=snapshot,
    )
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
            config_dir=snapshot.path.parent,
        )
        return path
    raise ProviderProfileError(f"unsupported credential_source in connection: {record.credential_source}")


def save_provider_connection(
    record: ProviderConnectionRecord,
    *,
    snapshot: ProvidersConfigSnapshot,
) -> Path:
    _validate_connection_record(record.to_dict())
    snapshot.require_valid()
    ensure_providers_snapshot_current(snapshot)
    path = snapshot.path
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    loaded_records = list(snapshot.records)
    # TUI 连接向导通常不携带 models；更新时保留该连接已有 models 配置。
    preserved_models = record.models
    if not preserved_models:
        for existing in loaded_records:
            if existing.id == record.id and existing.models:
                preserved_models = existing.models
                break
    write_record = ProviderConnectionRecord(
        id=record.id,
        name=record.name,
        provider_id=record.provider_id,
        provider_name=record.provider_name,
        gateway_provider=record.gateway_provider,
        base_url=record.base_url,
        api_key_env=record.api_key_env,
        credential_source=record.credential_source,
        runtime_kind=record.runtime_kind,
        models=preserved_models,
    )
    updated = False
    next_records: list[ProviderConnectionRecord] = []
    for existing in loaded_records:
        if existing.id == write_record.id:
            next_records.append(write_record)
            updated = True
        else:
            next_records.append(existing)
    if not updated:
        next_records.append(write_record)
    _write_providers_file(path, next_records)
    return path


def delete_provider_connection(
    connection_id: str,
    *,
    snapshot: ProvidersConfigSnapshot,
) -> Path:
    if not connection_id.strip():
        raise ProviderProfileError("provider connection id is required")
    snapshot.require_valid()
    ensure_providers_snapshot_current(snapshot)
    directory = snapshot.path.parent
    path = snapshot.path
    records = list(snapshot.records)
    next_records = [record for record in records if record.id != connection_id]
    if len(next_records) == len(records):
        raise ProviderProfileError(f"provider connection not found: {connection_id}")
    _write_providers_file(path, next_records)
    _refresh_active_model_after_connection_delete(directory, connection_id, next_records)
    return path


def provider_connection_credential_status(
    connection: ProviderConnectionRecord,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path,
) -> CredentialStatus:
    if connection.credential_source == "none":
        return CredentialStatus(
            api_key_available=True,
            credential_source_configured="none",
            credential_source_used="none",
            credential_store_available=None,
        )
    return credential_status(
        _credential_record(connection),
        environ=environ,
        credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
        config_dir=config_dir,
    )


def load_model_selection_profile(
    selection: ModelSelection,
    *,
    snapshot: ProvidersConfigSnapshot,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
) -> ProviderProfile:
    connection = snapshot.connection(selection.connection_id)
    if (selection.connection_id, selection.model) in snapshot.invalid_model_configs:
        raise ProviderProfileError(
            f"configured model is not available in catalog/discovery: "
            f"connection={selection.connection_id} model={selection.model}",
        )
    request_config = resolve_selection_request_config(selection, connection)
    if connection.credential_source == "none":
        return ProviderProfile(
            name=f"{connection.id}:{selection.model}",
            provider=connection.gateway_provider,
            base_url=connection.base_url,
            model=selection.model,
            api_key_env=connection.api_key_env,
            credential_source=connection.credential_source,
            credential_source_used="none",
            api_key="",
            runtime_kind=connection.runtime_kind,
            variant=selection.variant,
            request_config=request_config,
        )
    try:
        resolved = resolve_api_key(
            _credential_record(connection),
            environ=environ,
            credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
            config_dir=snapshot.path.parent,
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
        runtime_kind=connection.runtime_kind,
        variant=selection.variant,
        request_config=request_config,
    )


def resolve_selection_request_config(
    selection: ModelSelection,
    connection: ProviderConnectionRecord,
) -> ResolvedModelRequestConfig:
    model_config = connection.models.get(selection.model)
    try:
        return resolve_model_request_config(
            connection_id=selection.connection_id,
            model_id=selection.model,
            variant=selection.variant,
            model_config=model_config,
        )
    except ModelOptionsError as error:
        raise ProviderProfileError(str(error)) from error


def load_active_model_selection(*, settings_path: Path | None = None, config_dir: Path | None = None) -> ModelSelection:
    directory = config_dir or user_config_dir()
    path = settings_path or directory / USER_SETTINGS_FILE
    if not path.exists():
        raise ProviderProfileError(_setup_required_message())
    settings = _load_settings_record(path)
    active_model = settings.get("active_model")
    if isinstance(active_model, dict):
        selection = _model_selection_from_setting(active_model, field_name="active_model")
        if selection is not None:
            return selection
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


def load_model_route(*, settings_path: Path | None = None, config_dir: Path | None = None) -> ModelRoute:
    directory = config_dir or user_config_dir()
    path = settings_path or directory / USER_SETTINGS_FILE
    primary = load_active_model_selection(settings_path=path, config_dir=directory)
    settings = _load_settings_record(path)
    fallback = _model_selection_from_setting(settings.get("fallback_model"), field_name="fallback_model")
    return ModelRoute(
        primary=primary,
        fallback=fallback,
        cloud_fallback_consent=settings.get("cloud_fallback_consent") is True,
    )


def save_fallback_model_selection(
    selection: ModelSelection | None,
    *,
    cloud_fallback_consent: bool,
    config_dir: Path | None = None,
) -> Path:
    directory = config_dir or user_config_dir()
    path = directory / USER_SETTINGS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    settings = _load_settings_record(path) if path.exists() else {}
    if selection is None:
        settings.pop("fallback_model", None)
        settings["cloud_fallback_consent"] = False
    else:
        if not selection.connection_id.strip():
            raise ProviderProfileError("fallback model connection_id is required")
        if not selection.model.strip():
            raise ProviderProfileError("fallback model model is required")
        settings["fallback_model"] = selection.to_dict()
        settings["cloud_fallback_consent"] = cloud_fallback_consent
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
    fallback_model = settings.get("fallback_model")
    fallback_connection_id = (
        fallback_model.get("connection_id") if isinstance(fallback_model, dict) else None
    )
    if fallback_connection_id == connection_id:
        settings.pop("fallback_model", None)
        settings["cloud_fallback_consent"] = False
    active_model = settings.get("active_model")
    active_connection_id = active_model.get("connection_id") if isinstance(active_model, dict) else None
    if active_connection_id == connection_id:
        if next_records:
            next_active: dict[str, Any] = {
                "connection_id": next_records[0].id,
                "model": str(active_model.get("model", "")) if isinstance(active_model, dict) else "",
            }
            if isinstance(active_model, dict) and isinstance(active_model.get("variant"), str):
                next_active["variant"] = active_model["variant"]
            settings["active_model"] = next_active
        else:
            settings.pop("active_model", None)
    if settings:
        _write_json(settings_path, settings)
    else:
        settings_path.unlink()


def _write_providers_file(path: Path, records: list[ProviderConnectionRecord]) -> None:
    # 连接编辑必须完整保留未修改连接的 models。
    _write_json(
        path,
        {
            "version": PROVIDERS_CONFIG_VERSION,
            "connections": [connection.to_dict() for connection in records],
        },
    )


def _parse_connection_records(raw_text: str, config_path: Path) -> list[ProviderConnectionRecord]:
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ProviderProfileError(
            f"provider connection config is invalid JSON: {config_path}: {error.msg} "
            f"(line {error.lineno} column {error.colno}). Fix the file and restart HaAgent.",
        ) from error
    if not isinstance(raw, dict):
        raise ProviderProfileError(
            f"provider connection config must be a JSON object: {config_path}",
        )
    version = raw.get("version")
    if version != PROVIDERS_CONFIG_VERSION:
        raise ProviderProfileError(
            f"providers.json version must be {PROVIDERS_CONFIG_VERSION}: {config_path}",
        )
    _reject_unknown_fields(raw, {"$schema", "version", "connections"}, path="providers")
    if "$schema" in raw and not isinstance(raw["$schema"], str):
        raise ProviderProfileError("providers.$schema must be a string")
    connections = raw.get("connections")
    if not isinstance(connections, list):
        raise ProviderProfileError(
            f"provider connection config must contain connections: {config_path}",
        )
    records = []
    for index, item in enumerate(connections):
        if not isinstance(item, dict):
            raise ProviderProfileError(
                f"connections[{index}] must be an object ({config_path})",
            )
        try:
            _validate_connection_record(item, path=f"connections[{index}]")
            records.append(_connection_from_record(item, index=index))
        except (ProviderProfileError, ModelOptionsError) as error:
            raise ProviderProfileError(
                f"{error} (file={config_path})",
            ) from error
    return records


def _connection_from_record(
    record: dict[str, object],
    *,
    index: int,
) -> ProviderConnectionRecord:
    gateway_provider = _required_gateway_provider(record)
    models = parse_connection_models(
        record["models"] if "models" in record else {},
        path=f"connections[{index}].models",
    )
    return ProviderConnectionRecord(
        id=_required_string(record, "id"),
        name=_required_string(record, "name"),
        provider_id=_required_string(record, "provider_id"),
        provider_name=_required_string(record, "provider_name"),
        gateway_provider=gateway_provider,
        base_url=_required_string(record, "base_url"),
        api_key_env=_api_key_env(record),
        credential_source=_credential_source(record),
        runtime_kind=_runtime_kind(record),
        models=models,
    )


def _model_parameter_config_to_dict(config: ModelParameterConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if config.options:
        payload["options"] = config.options
    if config.variants:
        payload["variants"] = config.variants
    return payload


def _validate_connection_record(
    record: dict[str, object],
    *,
    path: str = "connection",
) -> None:
    _reject_unknown_fields(
        record,
        {
            "id",
            "name",
            "provider_id",
            "provider_name",
            "gateway_provider",
            "base_url",
            "api_key_env",
            "credential_source",
            "runtime_kind",
            "models",
        },
        path=path,
    )
    _required_string(record, "id")
    _required_string(record, "name")
    _required_string(record, "provider_id")
    _required_string(record, "provider_name")
    _required_gateway_provider(record)
    _required_string(record, "base_url")
    credential_source = _credential_source(record)
    if credential_source == "none":
        api_key_env = record.get("api_key_env")
        if not isinstance(api_key_env, str):
            raise ProviderProfileError("provider connection field must be a string: api_key_env")
    else:
        _required_string(record, "api_key_env")
    _runtime_kind(record)
    if "api_key" in record:
        raise ProviderProfileError("provider connection must not contain api_key")


def _reject_unknown_fields(
    value: Mapping[str, object],
    allowed: set[str],
    *,
    path: str,
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ProviderProfileError(f"{path} contains unknown field: {unknown[0]}")


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


def _api_key_env(record: dict[str, object]) -> str:
    value = record.get("api_key_env")
    if _credential_source(record) == "none" and isinstance(value, str):
        return value
    return _required_string(record, "api_key_env")


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
    if value not in {"env", "keyring", "insecure_file", "none"}:
        raise ProviderProfileError(f"unsupported credential_source in connection: {value}")
    return value


def _runtime_kind(record: dict[str, object]) -> str:
    value = record.get("runtime_kind", "remote")
    if not isinstance(value, str) or value not in SUPPORTED_RUNTIME_KINDS:
        raise ProviderProfileError(f"unsupported runtime_kind in connection: {value}")
    return value


def _model_selection_from_setting(value: object, *, field_name: str = "model") -> ModelSelection | None:
    if not isinstance(value, dict):
        return None
    connection_id = value.get("connection_id")
    model = value.get("model")
    if not isinstance(connection_id, str) or not connection_id.strip():
        raise ProviderProfileError(f"{field_name} connection_id is required")
    if not isinstance(model, str) or not model.strip():
        raise ProviderProfileError(f"{field_name} model is required")
    variant_raw = value.get("variant")
    variant: str | None
    if variant_raw is None:
        variant = None
    elif isinstance(variant_raw, str) and variant_raw.strip():
        variant = variant_raw
    else:
        raise ProviderProfileError(f"{field_name} variant must be a non-empty string when present")
    return ModelSelection(connection_id=connection_id, model=model, variant=variant)
