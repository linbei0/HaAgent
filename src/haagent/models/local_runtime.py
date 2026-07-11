"""
haagent/models/local_runtime.py - Ollama 与 LM Studio 本机发现

只探测固定 loopback 端点，归一化本地聊天模型及非敏感能力元数据。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import httpx

from haagent.models.capabilities import ModelCapabilities

DiscoveryStatus = Literal["available", "unreachable", "unauthorized", "invalid_response"]
RuntimeKind = Literal["ollama", "lm_studio"]

OLLAMA_API_URL = "http://127.0.0.1:11434"
OLLAMA_OPENAI_BASE_URL = f"{OLLAMA_API_URL}/v1"
LM_STUDIO_API_URL = "http://127.0.0.1:1234"
LM_STUDIO_OPENAI_BASE_URL = f"{LM_STUDIO_API_URL}/v1"
LOCAL_DISCOVERY_TIMEOUT_SECONDS = 1.5


@dataclass(frozen=True)
class LocalRuntimeModel:
    id: str
    name: str
    loaded: bool
    capabilities: ModelCapabilities
    family: str | None = None


@dataclass(frozen=True)
class LocalRuntimeDiscovery:
    runtime_kind: RuntimeKind
    base_url: str
    status: DiscoveryStatus
    models: tuple[LocalRuntimeModel, ...] = ()
    reason: str | None = None


def discover_ollama(
    *,
    transport: httpx.BaseTransport | None = None,
) -> LocalRuntimeDiscovery:
    try:
        with _client(transport=transport) as client:
            response = client.get(f"{OLLAMA_API_URL}/api/tags")
            failure = _http_failure(
                response,
                runtime_kind="ollama",
                base_url=OLLAMA_OPENAI_BASE_URL,
                display_name="Ollama",
            )
            if failure is not None:
                return failure
            payload = _json_object(response)
            model_records = payload.get("models")
            if not isinstance(model_records, list):
                raise ValueError("models must be a list")
            with ThreadPoolExecutor(max_workers=4) as executor:
                models = tuple(
                    executor.map(
                        lambda record: _ollama_model(client, record),
                        [record for record in model_records if isinstance(record, dict)],
                    ),
                )
        return LocalRuntimeDiscovery(
            runtime_kind="ollama",
            base_url=OLLAMA_OPENAI_BASE_URL,
            status="available",
            models=models,
        )
    except httpx.HTTPError:
        return _unreachable("ollama", OLLAMA_OPENAI_BASE_URL, "Ollama")
    except (TypeError, ValueError):
        return _invalid_response("ollama", OLLAMA_OPENAI_BASE_URL, "Ollama")


def discover_lm_studio(
    *,
    transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> LocalRuntimeDiscovery:
    token = (environ or os.environ).get("LM_STUDIO_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        with _client(transport=transport, headers=headers) as client:
            response = client.get(f"{LM_STUDIO_API_URL}/api/v1/models")
            failure = _http_failure(
                response,
                runtime_kind="lm_studio",
                base_url=LM_STUDIO_OPENAI_BASE_URL,
                display_name="LM Studio",
            )
            if failure is not None:
                return failure
            payload = _json_object(response)
            model_records = payload.get("models")
            if not isinstance(model_records, list):
                raise ValueError("models must be a list")
            models = tuple(
                _lm_studio_model(record)
                for record in model_records
                if isinstance(record, dict) and record.get("type") == "llm"
            )
        return LocalRuntimeDiscovery(
            runtime_kind="lm_studio",
            base_url=LM_STUDIO_OPENAI_BASE_URL,
            status="available",
            models=models,
        )
    except httpx.HTTPError:
        return _unreachable("lm_studio", LM_STUDIO_OPENAI_BASE_URL, "LM Studio")
    except (TypeError, ValueError):
        return _invalid_response("lm_studio", LM_STUDIO_OPENAI_BASE_URL, "LM Studio")


def discover_local_runtimes(
    *,
    ollama_transport: httpx.BaseTransport | None = None,
    lm_studio_transport: httpx.BaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[LocalRuntimeDiscovery, LocalRuntimeDiscovery]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        ollama_future = executor.submit(discover_ollama, transport=ollama_transport)
        lm_studio_future = executor.submit(
            discover_lm_studio,
            transport=lm_studio_transport,
            environ=environ,
        )
        return ollama_future.result(), lm_studio_future.result()


def _client(
    *,
    transport: httpx.BaseTransport | None,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    return httpx.Client(
        transport=transport,
        headers=headers,
        timeout=LOCAL_DISCOVERY_TIMEOUT_SECONDS,
        trust_env=False,
    )


def _ollama_model(client: httpx.Client, record: dict[str, object]) -> LocalRuntimeModel:
    model_id = _model_id(record)
    family = _nested_string(record, "details", "family")
    try:
        response = client.post(f"{OLLAMA_API_URL}/api/show", json={"model": model_id})
        response.raise_for_status()
        details = _json_object(response)
        raw_capabilities = details.get("capabilities")
        capability_names = {
            item for item in raw_capabilities if isinstance(item, str)
        } if isinstance(raw_capabilities, list) else set()
        capabilities = ModelCapabilities(
            tools="supported" if "tools" in capability_names else "unsupported",
            streaming="supported",
            vision="supported" if "vision" in capability_names else "unsupported",
            reasoning="supported" if "thinking" in capability_names else "unsupported",
            tools_mode="native" if "tools" in capability_names else "none",
            context_window_tokens=_ollama_context_window(details.get("model_info")),
            protocols=frozenset({"responses", "chat_completions"}),
        )
    except (httpx.HTTPError, TypeError, ValueError):
        # 单个模型详情失败不能隐藏已发现模型；未知能力留给协商层显式诊断。
        capabilities = ModelCapabilities(
            streaming="supported",
            protocols=frozenset({"responses", "chat_completions"}),
        )
    return LocalRuntimeModel(
        id=model_id,
        name=str(record.get("name") or model_id),
        loaded=False,
        capabilities=capabilities,
        family=family,
    )


def _lm_studio_model(record: dict[str, object]) -> LocalRuntimeModel:
    model_id = _model_id(record, key="key")
    raw_capabilities = record.get("capabilities")
    capability_data = raw_capabilities if isinstance(raw_capabilities, dict) else {}
    trained_for_tools = capability_data.get("trained_for_tool_use") is True
    loaded_instances = record.get("loaded_instances")
    instances = loaded_instances if isinstance(loaded_instances, list) else []
    context_window = _loaded_context_window(instances) or _positive_int(record.get("max_context_length"))
    reasoning = capability_data.get("reasoning")
    return LocalRuntimeModel(
        id=model_id,
        name=str(record.get("display_name") or model_id),
        loaded=bool(instances),
        family=str(record.get("architecture")) if record.get("architecture") else None,
        capabilities=ModelCapabilities(
            tools="supported",
            streaming="supported",
            vision="supported" if capability_data.get("vision") is True else "unsupported",
            reasoning="supported" if isinstance(reasoning, dict) else "unsupported",
            tools_mode="native" if trained_for_tools else "compat",
            context_window_tokens=context_window,
            protocols=frozenset({"responses", "chat_completions"}),
        ),
    )


def _http_failure(
    response: httpx.Response,
    *,
    runtime_kind: RuntimeKind,
    base_url: str,
    display_name: str,
) -> LocalRuntimeDiscovery | None:
    if response.status_code in {401, 403}:
        return LocalRuntimeDiscovery(
            runtime_kind=runtime_kind,
            base_url=base_url,
            status="unauthorized",
            reason=f"{display_name} discovery requires authentication",
        )
    if response.is_error:
        return _unreachable(runtime_kind, base_url, display_name)
    return None


def _unreachable(runtime_kind: RuntimeKind, base_url: str, display_name: str) -> LocalRuntimeDiscovery:
    return LocalRuntimeDiscovery(
        runtime_kind=runtime_kind,
        base_url=base_url,
        status="unreachable",
        reason=f"{display_name} is not reachable",
    )


def _invalid_response(runtime_kind: RuntimeKind, base_url: str, display_name: str) -> LocalRuntimeDiscovery:
    return LocalRuntimeDiscovery(
        runtime_kind=runtime_kind,
        base_url=base_url,
        status="invalid_response",
        reason=f"{display_name} returned an invalid discovery response",
    )


def _json_object(response: httpx.Response) -> dict[str, object]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("response must be an object")
    return payload


def _model_id(record: dict[str, object], *, key: str = "model") -> str:
    value = record.get(key) or record.get("name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model identifier is required")
    return value.strip()


def _ollama_context_window(value: object) -> int | None:
    if not isinstance(value, dict):
        return None
    windows = [
        item
        for key, raw in value.items()
        if isinstance(key, str) and key.endswith(".context_length")
        if (item := _positive_int(raw)) is not None
    ]
    return max(windows) if windows else None


def _loaded_context_window(instances: list[object]) -> int | None:
    for instance in instances:
        if not isinstance(instance, dict):
            continue
        config = instance.get("config")
        if isinstance(config, dict):
            value = _positive_int(config.get("context_length"))
            if value is not None:
                return value
    return None


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nested_string(record: dict[str, object], parent: str, key: str) -> str | None:
    value = record.get(parent)
    if not isinstance(value, dict):
        return None
    nested = value.get(key)
    return nested if isinstance(nested, str) and nested else None
