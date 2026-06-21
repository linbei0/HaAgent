"""
haagent/models/provider_profile.py - 本地 provider profile 读取

读取 .haagent/providers.json，并只通过 api_key_env 指定的环境变量解析密钥。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


DEFAULT_PROVIDER_PROFILE_PATH = Path(".haagent") / "providers.json"
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


def load_provider_profile(
    name: str,
    *,
    config_path: Path = DEFAULT_PROVIDER_PROFILE_PATH,
    environ: Mapping[str, str] | None = None,
) -> ProviderProfile:
    """按名称读取 provider profile，并从指定环境变量解析 API key。"""
    records = _load_profile_records(config_path)
    record = _find_profile_record(records, name)
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


def _required_string(record: dict[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ProviderProfileError(f"provider profile field is required: {field_name}")
    return value
