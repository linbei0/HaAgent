"""
haagent/models/provider_profile.py - 用户级模型连接配置

读取和写入 HaAgent provider profile，并通过凭据层解析真实 API key。
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
    save_insecure_api_key,
    save_keyring_api_key,
)


DEFAULT_PROVIDER_PROFILE_PATH = Path(".haagent") / "providers.json"
USER_CONFIG_DIR_NAME = ".haagent"
USER_PROVIDERS_FILE = "providers.json"
USER_SETTINGS_FILE = "settings.json"
SUPPORTED_PROFILE_PROVIDERS = {"openai", "openai-chat"}
DEFAULT_CREDENTIAL_SOURCE = "keyring"
DEFAULT_CREDENTIAL_STORE: CredentialStore = KeyringCredentialStore()


class ProviderProfileError(RuntimeError):
    """Raised when a provider profile cannot be loaded explicitly."""


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
class ProviderProfileRecord:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    credential_source: str = DEFAULT_CREDENTIAL_SOURCE

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "credential_source": self.credential_source,
        }


def user_config_dir() -> Path:
    return Path.home() / USER_CONFIG_DIR_NAME


def user_provider_profile_path() -> Path:
    return user_config_dir() / USER_PROVIDERS_FILE


def user_settings_path() -> Path:
    return user_config_dir() / USER_SETTINGS_FILE


def load_provider_profile(
    name: str,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> ProviderProfile:
    """按名称读取 provider profile，并按凭据优先级解析 API key。"""
    record = _load_profile_record(name, config_path=config_path)
    api_key_env = _required_string(record, "api_key_env")
    credential_source = _credential_source(record)
    try:
        resolved = resolve_api_key(
            CredentialRecord(
                profile_name=_required_string(record, "name"),
                api_key_env=api_key_env,
                credential_source=credential_source,
            ),
            environ=environ,
            credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
            config_dir=config_dir or _config_dir_for(config_path),
        )
    except CredentialError as error:
        raise ProviderProfileError(str(error)) from error
    if not resolved.api_key:
        detail = resolved.credential_store_error or f"configured source: {credential_source}"
        raise ProviderProfileError(
            f"API key is not available for profile {name}: {detail}; api_key_env={api_key_env}",
        )
    return ProviderProfile(
        name=_required_string(record, "name"),
        provider=_required_provider(record),
        base_url=_required_string(record, "base_url"),
        model=_required_string(record, "model"),
        api_key_env=api_key_env,
        credential_source=credential_source,
        credential_source_used=resolved.credential_source_used or "",
        api_key=resolved.api_key,
    )


def load_provider_profile_record(
    name: str,
    *,
    config_path: Path | None = None,
) -> ProviderProfileRecord:
    """读取不含真实 API key 的 provider profile 配置。"""
    record = _load_profile_record(name, config_path=config_path)
    return ProviderProfileRecord(
        name=_required_string(record, "name"),
        provider=_required_provider(record),
        base_url=_required_string(record, "base_url"),
        model=_required_string(record, "model"),
        api_key_env=_required_string(record, "api_key_env"),
        credential_source=_credential_source(record),
    )


def list_provider_profile_records(*, config_path: Path | None = None) -> list[ProviderProfileRecord]:
    path = config_path or user_provider_profile_path()
    records = _load_profile_records(path) if path.exists() else []
    return [
        ProviderProfileRecord(
            name=_required_string(record, "name"),
            provider=_required_provider(record),
            base_url=_required_string(record, "base_url"),
            model=_required_string(record, "model"),
            api_key_env=_required_string(record, "api_key_env"),
            credential_source=_credential_source(record),
        )
        for record in records
    ]


def provider_profile_credential_status(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> CredentialStatus:
    config_path = (config_dir / USER_PROVIDERS_FILE) if config_dir is not None else None
    record = load_provider_profile_record(name, config_path=config_path)
    return credential_status(
        CredentialRecord(
            profile_name=record.name,
            api_key_env=record.api_key_env,
            credential_source=record.credential_source,
        ),
        environ=environ,
        credential_store=credential_store,
        config_dir=config_dir,
    )


def save_provider_profile_with_key(
    record: ProviderProfileRecord,
    api_key: str | None,
    *,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> Path:
    has_api_key = api_key is not None and bool(api_key.strip())
    if record.credential_source == "env" and has_api_key:
        raise ProviderProfileError("env credential_source does not allow saving api_key")
    path = save_provider_profile(record, config_dir=config_dir)
    if not has_api_key:
        return path
    if record.credential_source == "keyring":
        save_keyring_api_key(
            record.name,
            api_key,
            credential_store=credential_store,
        )
        return path
    if record.credential_source == "insecure_file":
        save_insecure_api_key(
            record.name,
            api_key,
            config_dir=config_dir or user_config_dir(),
        )
        return path
    raise ProviderProfileError(f"unsupported credential_source in profile: {record.credential_source}")


def load_active_provider_profile(
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
) -> ProviderProfile:
    """读取用户级 active profile，并解析对应 API key。"""
    return load_provider_profile(
        load_active_profile_name(),
        config_path=user_provider_profile_path(),
        environ=environ,
        credential_store=credential_store,
        config_dir=user_config_dir(),
    )


def load_active_provider_profile_record() -> ProviderProfileRecord:
    """读取 active profile 的非敏感配置，用于状态展示。"""
    return load_provider_profile_record(
        load_active_profile_name(),
        config_path=user_provider_profile_path(),
    )


def active_provider_credential_status(
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
):
    """读取 active profile 的非敏感凭据状态。"""
    record = load_active_provider_profile_record()
    try:
        return credential_status(
            CredentialRecord(
                profile_name=record.name,
                api_key_env=record.api_key_env,
                credential_source=record.credential_source,
            ),
            environ=environ,
            credential_store=credential_store or DEFAULT_CREDENTIAL_STORE,
            config_dir=user_config_dir(),
        )
    except CredentialError as error:
        raise ProviderProfileError(str(error)) from error


def load_active_profile_name(*, settings_path: Path | None = None) -> str:
    path = settings_path or user_settings_path()
    if not path.exists():
        raise ProviderProfileError(_setup_required_message())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderProfileError(f"settings config is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ProviderProfileError("settings config must be a JSON object")
    active_profile = raw.get("active_profile")
    if not isinstance(active_profile, str) or not active_profile.strip():
        raise ProviderProfileError("settings config must contain active_profile")
    return active_profile


def save_provider_profile(record: ProviderProfileRecord, *, config_dir: Path | None = None) -> Path:
    """把模型连接配置写入用户级 providers.json；不会写入真实 API key。"""
    _validate_profile_record(record.to_dict())
    directory = config_dir or user_config_dir()
    path = directory / USER_PROVIDERS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    records = _load_profile_records(path) if path.exists() else []
    updated = False
    next_records = []
    for existing in records:
        if existing.get("name") == record.name:
            next_records.append(record.to_dict())
            updated = True
        else:
            next_records.append(existing)
    if not updated:
        next_records.append(record.to_dict())
    _write_json(path, {"profiles": next_records})
    return path


def delete_provider_profile(name: str, *, config_dir: Path | None = None) -> Path:
    if not name.strip():
        raise ProviderProfileError("provider profile name is required")
    directory = config_dir or user_config_dir()
    path = directory / USER_PROVIDERS_FILE
    records = _load_profile_records(path)
    next_records = [record for record in records if record.get("name") != name]
    if len(next_records) == len(records):
        raise ProviderProfileError(f"provider profile not found: {name}")
    _write_json(path, {"profiles": next_records})

    settings_path = directory / USER_SETTINGS_FILE
    try:
        active_profile = load_active_profile_name(settings_path=settings_path)
    except ProviderProfileError:
        active_profile = None
    if active_profile == name:
        if next_records:
            replacement = _required_string(next_records[0], "name")
            save_active_profile(replacement, config_dir=directory)
        elif settings_path.exists():
            settings = _load_settings_record(settings_path)
            settings.pop("active_profile", None)
            if settings:
                _write_json(settings_path, settings)
            else:
                settings_path.unlink()
    return path


def save_active_profile(name: str, *, config_dir: Path | None = None) -> Path:
    if not name.strip():
        raise ProviderProfileError("active profile name is required")
    directory = config_dir or user_config_dir()
    path = directory / USER_SETTINGS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    settings = _load_settings_record(path) if path.exists() else {}
    settings["active_profile"] = name
    _write_json(path, settings)
    return path


def _load_profile_record(name: str, *, config_path: Path | None) -> dict[str, object]:
    if config_path is not None:
        return _find_profile_record(_load_profile_records(config_path), name)
    searched = []
    for candidate in [user_provider_profile_path(), DEFAULT_PROVIDER_PROFILE_PATH]:
        searched.append(candidate)
        if not candidate.exists():
            continue
        records = _load_profile_records(candidate)
        try:
            return _find_profile_record(records, name)
        except ProviderProfileError:
            continue
    searched_paths = ", ".join(str(path) for path in searched)
    raise ProviderProfileError(f"provider profile not found: {name}; searched: {searched_paths}")


def _load_profile_records(config_path: Path) -> list[dict[str, object]]:
    if not config_path.exists():
        raise ProviderProfileError(f"provider profile config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProviderProfileError(f"provider profile config is invalid JSON: {config_path}") from error
    if not isinstance(raw, dict):
        raise ProviderProfileError("provider profile config must be a JSON object")
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        raise ProviderProfileError("provider profile config must contain a profiles list")
    records = []
    for index, item in enumerate(profiles):
        if not isinstance(item, dict):
            raise ProviderProfileError(f"provider profile at index {index} must be an object")
        _validate_profile_record(item)
        records.append(item)
    return records


def _find_profile_record(records: list[dict[str, object]], name: str) -> dict[str, object]:
    for record in records:
        if record.get("name") == name:
            return record
    raise ProviderProfileError(f"provider profile not found: {name}")


def _required_provider(record: dict[str, object]) -> str:
    provider = _required_string(record, "provider")
    if provider not in SUPPORTED_PROFILE_PROVIDERS:
        raise ProviderProfileError(f"unsupported provider in profile: {provider}")
    return provider


def _validate_profile_record(record: dict[str, object]) -> None:
    _required_string(record, "name")
    _required_provider(record)
    _required_string(record, "base_url")
    _required_string(record, "model")
    _required_string(record, "api_key_env")
    _credential_source(record)
    if "api_key" in record:
        raise ProviderProfileError("provider profile must not contain api_key")


def _required_string(record: dict[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError(f"provider profile field is required: {field_name}")
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
    return "未找到默认模型配置，请运行 haagent 后在 TUI 内输入 /model 完成配置"


def _credential_source(record: dict[str, object]) -> str:
    value = record.get("credential_source", DEFAULT_CREDENTIAL_SOURCE)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError("provider profile field is required: credential_source")
    if value not in {"env", "keyring", "insecure_file"}:
        raise ProviderProfileError(f"unsupported credential_source in profile: {value}")
    return value


def _config_dir_for(config_path: Path | None) -> Path:
    if config_path is None:
        return user_config_dir()
    return config_path.parent
