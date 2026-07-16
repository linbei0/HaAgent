"""
src/haagent/models/catalog.py - 模型目录发现与缓存

负责从 Models.dev 读取公开 provider/model 元数据并维护本地缓存。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen

from haagent.models.config.connections import user_config_dir

MODELS_DEV_URL = "https://models.dev/api.json"
MODELS_DEV_USER_AGENT = "HaAgent/0.1 (+https://models.dev)"
MODEL_CATALOG_CACHE_FILE = "models_catalog_cache.json"
DEFAULT_MODEL_CATALOG_CACHE_MAX_AGE = timedelta(hours=24)
CatalogTransport = Callable[[], dict[str, object]]


class ModelCatalogError(Exception):
    """模型目录读取失败。"""


@dataclass(frozen=True)
class ModelCatalogModel:
    id: str
    name: str = ""
    family: str | None = None
    supports_tool_call: bool = False
    supports_reasoning: bool = False
    modalities: dict[str, object] = field(default_factory=dict)
    limits: dict[str, object] = field(default_factory=dict)
    cost: dict[str, object] = field(default_factory=dict)
    release_date: str | None = None
    last_updated: str | None = None


@dataclass(frozen=True)
class ModelCatalogProvider:
    id: str
    name: str = ""
    env_names: list[str] = field(default_factory=list)
    api_base_url: str | None = None
    provider_package: str | None = None
    documentation_url: str | None = None
    models: list[ModelCatalogModel] = field(default_factory=list)


@dataclass(frozen=True)
class CatalogFetchResult:
    providers: list[ModelCatalogProvider]
    source: str
    fetched_at: str
    used_cache: bool = False
    error: str | None = None


@dataclass(frozen=True)
class CachedCatalog:
    catalog: dict[str, object]
    fetched_at: datetime | None


def fetch_model_catalog(
    *,
    cache_path: Path | None = None,
    transport: CatalogTransport | None = None,
    max_cache_age: timedelta | None = None,
    force_refresh: bool = False,
) -> CatalogFetchResult:
    cache_file = cache_path or (user_config_dir() / MODEL_CATALOG_CACHE_FILE)
    fetch_transport = transport or _default_transport
    if max_cache_age is not None and not force_refresh:
        cached = _load_cached_catalog(cache_file)
        if cached is not None and _cache_is_fresh(cached.fetched_at, max_cache_age):
            return CatalogFetchResult(
                providers=_parse_providers(cached.catalog),
                source=str(cache_file),
                fetched_at=_now_iso(),
                used_cache=True,
                error=None,
            )

    try:
        raw = fetch_transport()
    except Exception as error:  # noqa: BLE001
        cached = _load_cached_catalog(cache_file)
        if cached is None:
            raise ModelCatalogError(str(error)) from error
        return CatalogFetchResult(
            providers=_parse_providers(cached.catalog),
            source=str(cache_file),
            fetched_at=_now_iso(),
            used_cache=True,
            error=str(error),
        )

    if not isinstance(raw, dict):
        raise ModelCatalogError("transport result must be a JSON object")
    providers = _parse_providers(raw)
    _write_cache(cache_file, raw)
    return CatalogFetchResult(
        providers=providers,
        source=MODELS_DEV_URL,
        fetched_at=_now_iso(),
        used_cache=False,
        error=None,
    )


def load_cached_model_catalog(*, cache_path: Path | None = None) -> CatalogFetchResult | None:
    """只读现有目录缓存，不联网；供 AssistantService 启动快照匹配模型。"""

    cache_file = cache_path or (user_config_dir() / MODEL_CATALOG_CACHE_FILE)
    cached = _load_cached_catalog(cache_file)
    if cached is None:
        return None
    return CatalogFetchResult(
        providers=_parse_providers(cached.catalog),
        source=str(cache_file),
        fetched_at=(cached.fetched_at.isoformat() if cached.fetched_at is not None else _now_iso()),
        used_cache=True,
        error=None,
    )


def _default_transport() -> dict[str, object]:
    request = Request(
        MODELS_DEV_URL,
        headers={
            "User-Agent": MODELS_DEV_USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ModelCatalogError("models.dev catalog must be a JSON object")
    return parsed


def _parse_providers(raw: dict[str, object]) -> list[ModelCatalogProvider]:
    providers: list[ModelCatalogProvider] = []
    for provider_id, payload in raw.items():
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ModelCatalogError("provider id must be a non-empty string")
        if not isinstance(payload, dict):
            raise ModelCatalogError(f"provider payload must be an object: {provider_id}")
        providers.append(_parse_provider(provider_id, payload))
    return providers


def _parse_provider(provider_id: str, payload: dict[str, object]) -> ModelCatalogProvider:
    provider_name = payload.get("name")
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ModelCatalogError(f"provider field must be a non-empty string: name ({provider_id})")
    models_raw = payload.get("models", {})
    if not isinstance(models_raw, dict):
        raise ModelCatalogError(f"provider models must be an object: {provider_id}")
    models: list[ModelCatalogModel] = []
    for model_id, model_payload in models_raw.items():
        if not isinstance(model_id, str):
            raise ModelCatalogError(f"model entry key must be a string: {provider_id}")
        if not isinstance(model_payload, dict):
            raise ModelCatalogError(f"model entry must be an object: {provider_id}/{model_id}")
        models.append(_parse_model(model_id, model_payload))
    env_names = payload.get("env", [])
    if not isinstance(env_names, list):
        raise ModelCatalogError(f"provider env must be a list: {provider_id}")
    parsed_env_names: list[str] = []
    for env_name in env_names:
        if not isinstance(env_name, str) or not env_name.strip():
            raise ModelCatalogError(f"provider env entry must be a string: {provider_id}")
        parsed_env_names.append(env_name)
    return ModelCatalogProvider(
        id=provider_id,
        name=provider_name,
        env_names=parsed_env_names,
        api_base_url=_optional_string(payload.get("api")),
        provider_package=_optional_string(payload.get("npm")),
        documentation_url=_optional_string(payload.get("doc")),
        models=models,
    )


def _parse_model(model_id: str, payload: dict[str, object]) -> ModelCatalogModel:
    tool_call = payload.get("tool_call", False)
    reasoning = payload.get("reasoning", False)
    modalities = payload.get("modalities", {})
    limits = payload.get("limit", {})
    cost = payload.get("cost", {})
    model_record_id = payload.get("id")
    model_name = payload.get("name")
    if not isinstance(model_record_id, str) or not model_record_id.strip():
        raise ModelCatalogError(f"model field must be a non-empty string: id ({model_id})")
    if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
        raise ModelCatalogError(f"model field must be a non-empty string: name ({model_id})")
    if not isinstance(tool_call, bool):
        raise ModelCatalogError(f"model field must be boolean: tool_call ({model_id})")
    if not isinstance(reasoning, bool):
        raise ModelCatalogError(f"model field must be boolean: reasoning ({model_id})")
    if not isinstance(modalities, dict):
        raise ModelCatalogError(f"model field must be an object: modalities ({model_id})")
    if not isinstance(limits, dict):
        raise ModelCatalogError(f"model field must be an object: limit ({model_id})")
    if not isinstance(cost, dict):
        raise ModelCatalogError(f"model field must be an object: cost ({model_id})")
    return ModelCatalogModel(
        id=model_record_id,
        name=model_name if isinstance(model_name, str) else "",
        family=_optional_string(payload.get("family")),
        supports_tool_call=tool_call,
        supports_reasoning=reasoning,
        modalities=modalities,
        limits=limits,
        cost=cost,
        release_date=_optional_string(payload.get("release_date")),
        last_updated=_optional_string(payload.get("last_updated")),
    )


def _write_cache(path: Path, raw: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": _now_iso(),
        "source": MODELS_DEV_URL,
        "catalog": raw,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_cached_catalog(path: Path) -> CachedCatalog | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelCatalogError(f"model catalog cache is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ModelCatalogError(f"model catalog cache must be a JSON object: {path}")
    catalog = raw.get("catalog")
    if not isinstance(catalog, dict):
        raise ModelCatalogError(f"model catalog cache missing catalog object: {path}")
    return CachedCatalog(catalog=catalog, fetched_at=_parse_cache_fetched_at(raw.get("fetched_at"), path))


def _parse_cache_fetched_at(value: object, path: Path) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelCatalogError(f"model catalog cache fetched_at must be a string: {path}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ModelCatalogError(f"model catalog cache fetched_at is invalid: {path}") from error
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _cache_is_fresh(fetched_at: datetime | None, max_age: timedelta) -> bool:
    if fetched_at is None:
        return False
    return datetime.now(tz=UTC) - fetched_at <= max_age


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()
