"""
haagent/models/config/config_store.py - providers v4 配置存储

独占 providers.json 的读取、严格解析、序列化、快照摘要和并发写入检查。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from haagent.models.config.connections import (
    DEFAULT_CREDENTIAL_SOURCE,
    PROVIDERS_CONFIG_VERSION,
    SUPPORTED_GATEWAY_PROVIDERS,
    SUPPORTED_RUNTIME_KINDS,
    ProviderConnectionRecord,
    ProviderProfileError,
    ProvidersConfigSnapshot,
    user_provider_connections_path,
)
from haagent.models.model_options import ModelOptionsError, parse_connection_models


ConfigSnapshot = ProvidersConfigSnapshot


class ModelConfigStore:
    """一个 runtime 生命周期内使用一个不可变 providers 快照。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or user_provider_connections_path()

    def load(self) -> ConfigSnapshot:
        if not self.path.exists():
            return ConfigSnapshot(path=self.path, records=(), digest=_digest(b""))
        try:
            raw = self.path.read_bytes()
        except OSError as error:
            return ConfigSnapshot(
                path=self.path,
                records=(),
                digest=_digest(b""),
                load_error=f"cannot read provider connection config: {self.path}: {error}",
            )
        try:
            # Windows 工具可能写入 UTF-8 BOM；读取兼容，HaAgent 自身仍写无 BOM 的 UTF-8。
            records = tuple(_parse_records(raw.decode("utf-8-sig"), self.path))
        except UnicodeDecodeError:
            error = f"provider connection config must be UTF-8: {self.path}"
        except ProviderProfileError as exc:
            error = str(exc)
        else:
            return ConfigSnapshot(path=self.path, records=records, digest=_digest(raw))
        return ConfigSnapshot(path=self.path, records=(), digest=_digest(raw), load_error=error)

    def ensure_current(self, snapshot: ConfigSnapshot) -> None:
        try:
            current = self.path.read_bytes() if self.path.exists() else b""
        except OSError as error:
            raise ProviderProfileError(
                f"cannot verify provider connection config before writing: {self.path}: {error}",
            ) from error
        if _digest(current) != snapshot.digest:
            raise ProviderProfileError(
                f"providers.json changed after HaAgent started: {self.path}. Restart HaAgent before changing connections.",
            )

    def save_connection(
        self,
        record: ProviderConnectionRecord,
        *,
        expected_digest: str,
    ) -> ConfigSnapshot:
        snapshot = self.load()
        if snapshot.digest != expected_digest:
            raise ProviderProfileError("providers.json changed after HaAgent started. Restart HaAgent before changing connections.")
        snapshot.require_valid()
        _validate_record(record.to_dict())
        existing = list(snapshot.records)
        if not record.models:
            previous = next((item for item in existing if item.id == record.id), None)
            if previous is not None and previous.models:
                record = ProviderConnectionRecord(
                    id=record.id,
                    name=record.name,
                    provider_id=record.provider_id,
                    provider_name=record.provider_name,
                    gateway_provider=record.gateway_provider,
                    base_url=record.base_url,
                    api_key_env=record.api_key_env,
                    credential_source=record.credential_source,
                    runtime_kind=record.runtime_kind,
                    models=previous.models,
                )
        records = [record if item.id == record.id else item for item in existing]
        if not any(item.id == record.id for item in existing):
            records.append(record)
        self._write(records)
        return self.load()

    def delete_connection(self, connection_id: str, *, expected_digest: str) -> ConfigSnapshot:
        if not connection_id.strip():
            raise ProviderProfileError("provider connection id is required")
        snapshot = self.load()
        if snapshot.digest != expected_digest:
            raise ProviderProfileError("providers.json changed after HaAgent started. Restart HaAgent before changing connections.")
        snapshot.require_valid()
        records = [item for item in snapshot.records if item.id != connection_id]
        if len(records) == len(snapshot.records):
            raise ProviderProfileError(f"provider connection not found: {connection_id}")
        self._write(records)
        return self.load()

    def _write(self, records: list[ProviderConnectionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": PROVIDERS_CONFIG_VERSION, "connections": [item.to_dict() for item in records]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:16]


def _parse_records(raw_text: str, path: Path) -> list[ProviderConnectionRecord]:
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ProviderProfileError(
            f"provider connection config is invalid JSON: {path}: {error.msg} "
            f"(line {error.lineno} column {error.colno}). Fix the file and restart HaAgent.",
        ) from error
    if not isinstance(raw, dict):
        raise ProviderProfileError(f"provider connection config must be a JSON object: {path}")
    if raw.get("version") != PROVIDERS_CONFIG_VERSION:
        raise ProviderProfileError(f"providers.json version must be {PROVIDERS_CONFIG_VERSION}: {path}")
    _reject_unknown(raw, {"$schema", "version", "connections"}, "providers")
    if "$schema" in raw and not isinstance(raw["$schema"], str):
        raise ProviderProfileError("providers.$schema must be a string")
    connections = raw.get("connections")
    if not isinstance(connections, list):
        raise ProviderProfileError(f"provider connection config must contain connections: {path}")
    records: list[ProviderConnectionRecord] = []
    for index, item in enumerate(connections):
        if not isinstance(item, dict):
            raise ProviderProfileError(f"connections[{index}] must be an object ({path})")
        try:
            _validate_record(item, path=f"connections[{index}]")
            records.append(_record_from_dict(item, index))
        except (ProviderProfileError, ModelOptionsError) as error:
            raise ProviderProfileError(f"{error} (file={path})") from error
    return records


def _record_from_dict(value: dict[str, object], index: int) -> ProviderConnectionRecord:
    return ProviderConnectionRecord(
        id=_required(value, "id"),
        name=_required(value, "name"),
        provider_id=_required(value, "provider_id"),
        provider_name=_required(value, "provider_name"),
        gateway_provider=_gateway_provider(value),
        base_url=_required(value, "base_url"),
        api_key_env=_api_key_env(value),
        credential_source=_credential_source(value),
        runtime_kind=_runtime_kind(value),
        models=parse_connection_models(value.get("models", {}), path=f"connections[{index}].models"),
    )


def _validate_record(value: dict[str, object], *, path: str = "connection") -> None:
    _reject_unknown(
        value,
        {"id", "name", "provider_id", "provider_name", "gateway_provider", "base_url", "api_key_env", "credential_source", "runtime_kind", "models"},
        path,
    )
    for field in ("id", "name", "provider_id", "provider_name", "base_url"):
        _required(value, field)
    _gateway_provider(value)
    source = _credential_source(value)
    if source == "none":
        if not isinstance(value.get("api_key_env", ""), str):
            raise ProviderProfileError("provider connection field must be a string: api_key_env")
    else:
        _required(value, "api_key_env")
    _runtime_kind(value)
    if "api_key" in value:
        raise ProviderProfileError("provider connection must not contain api_key")


def _reject_unknown(value: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ProviderProfileError(f"{path} contains unknown field: {unknown[0]}")


def _required(value: dict[str, object], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise ProviderProfileError(f"provider connection field is required: {field}")
    return item


def _gateway_provider(value: dict[str, object]) -> str:
    provider = _required(value, "gateway_provider")
    if provider not in SUPPORTED_GATEWAY_PROVIDERS:
        raise ProviderProfileError(f"unsupported provider in connection: {provider}")
    return provider


def _credential_source(value: dict[str, object]) -> str:
    source = value.get("credential_source", DEFAULT_CREDENTIAL_SOURCE)
    if not isinstance(source, str) or source not in {"env", "keyring", "insecure_file", "none"}:
        raise ProviderProfileError(f"unsupported credential_source in connection: {source}")
    return source


def _runtime_kind(value: dict[str, object]) -> str:
    kind = value.get("runtime_kind", "remote")
    if not isinstance(kind, str) or kind not in SUPPORTED_RUNTIME_KINDS:
        raise ProviderProfileError(f"unsupported runtime_kind in connection: {kind}")
    return kind


def _api_key_env(value: dict[str, object]) -> str:
    item = value.get("api_key_env")
    if _credential_source(value) == "none" and isinstance(item, str):
        return item
    return _required(value, "api_key_env")
