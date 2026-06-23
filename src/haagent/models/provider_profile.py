"""
haagent/models/provider_profile.py - 用户级模型连接配置

读取和写入 HaAgent provider profile，只通过 api_key_env 指定的环境变量解析密钥。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


DEFAULT_PROVIDER_PROFILE_PATH = Path(".haagent") / "providers.json"
USER_CONFIG_DIR_NAME = ".haagent"
USER_PROVIDERS_FILE = "providers.json"
USER_SETTINGS_FILE = "settings.json"
SUPPORTED_PROFILE_PROVIDERS = {"openai", "openai-chat"}


class ProviderProfileError(RuntimeError):
    """Raised when a provider profile cannot be loaded explicitly."""


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class ProviderProfileRecord:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
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
) -> ProviderProfile:
    """按名称读取 provider profile，并从指定环境变量解析 API key。"""
    record = _load_profile_record(name, config_path=config_path)
    api_key_env = _required_string(record, "api_key_env")
    environment = os.environ if environ is None else environ
    api_key = environment.get(api_key_env)
    if not api_key:
        raise ProviderProfileError(f"api key environment variable is not set: {api_key_env}")
    return ProviderProfile(
        name=_required_string(record, "name"),
        provider=_required_provider(record),
        base_url=_required_string(record, "base_url"),
        model=_required_string(record, "model"),
        api_key_env=api_key_env,
        api_key=api_key,
    )


def load_active_provider_profile(
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderProfile:
    """读取用户级 active profile，并解析对应 API key。"""
    return load_provider_profile(
        load_active_profile_name(),
        config_path=user_provider_profile_path(),
        environ=environ,
    )


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


def save_active_profile(name: str, *, config_dir: Path | None = None) -> Path:
    if not name.strip():
        raise ProviderProfileError("active profile name is required")
    directory = config_dir or user_config_dir()
    path = directory / USER_SETTINGS_FILE
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"active_profile": name})
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
    if "api_key" in record:
        raise ProviderProfileError("provider profile must not contain api_key")


def _required_string(record: dict[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError(f"provider profile field is required: {field_name}")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _setup_required_message() -> str:
    return "未找到默认模型配置，请先运行 haagent setup"
