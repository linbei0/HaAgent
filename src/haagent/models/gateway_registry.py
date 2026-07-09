"""
src/haagent/models/gateway_registry.py - 模型网关能力映射

负责把 profile 和公开模型目录映射为 HaAgent 当前可运行的 ModelGateway 能力。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.models.anthropic import AnthropicMessagesGateway
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.google import GoogleGeminiGateway
from haagent.models.model_connections import ProviderProfile, ProviderProfileError
from haagent.models.openai_chat import OpenAIChatCompletionsGateway
from haagent.models.openai_responses import OpenAIResponsesGateway
from haagent.models.types import ModelGateway


@dataclass(frozen=True)
class GatewayCapability:
    status: str
    gateway_provider: str | None
    reason: str | None = None


class GatewayRegistryError(ProviderProfileError):
    """网关能力映射失败。"""


_OPENAI_COMPATIBLE_PACKAGES = {
    "@ai-sdk/openai-compatible",
    "@openrouter/ai-sdk-provider",
}
_OPENAI_COMPATIBLE_PROVIDER_IDS = {
    "openrouter",
}


def catalog_provider_capability(provider: ModelCatalogProvider) -> GatewayCapability:
    if provider.id == "openai":
        return GatewayCapability(status="runnable", gateway_provider="openai")
    if provider.id == "openai-chat":
        return GatewayCapability(status="runnable", gateway_provider="openai-chat")
    provider_package = getattr(provider, "provider_package", None)
    if provider_package == "@ai-sdk/anthropic":
        return GatewayCapability(status="runnable", gateway_provider="anthropic")
    if provider_package == "@ai-sdk/google":
        return GatewayCapability(status="runnable", gateway_provider="google")
    if _is_openai_compatible_catalog_provider(provider):
        return GatewayCapability(status="runnable", gateway_provider="openai-chat")
    return GatewayCapability(
        status="adapter_required",
        gateway_provider=None,
        reason="native provider adapter is not available",
    )


def gateway_from_profile(profile: ProviderProfile) -> ModelGateway:
    # 空字符串与未配置等价，便于 CLI 临时 profile 回落到环境变量默认值。
    gateway_kwargs = {
        "api_key": profile.api_key or None,
        "model": profile.model,
        "base_url": profile.base_url or None,
    }
    if profile.provider == "openai":
        return OpenAIResponsesGateway(**gateway_kwargs)
    if profile.provider == "openai-chat":
        return OpenAIChatCompletionsGateway(**gateway_kwargs)
    if profile.provider == "anthropic":
        return AnthropicMessagesGateway(**gateway_kwargs)
    if profile.provider == "google":
        return GoogleGeminiGateway(**gateway_kwargs)
    raise GatewayRegistryError(f"unsupported provider in profile: {profile.provider}")


def _is_openai_compatible_catalog_provider(provider: ModelCatalogProvider) -> bool:
    provider_package = (getattr(provider, "provider_package", None) or "").strip().lower()
    if provider_package in _OPENAI_COMPATIBLE_PACKAGES:
        return True
    if provider.id in _OPENAI_COMPATIBLE_PROVIDER_IDS and _has_openai_chat_base_url(provider):
        return True
    return False


def _has_openai_chat_base_url(provider: ModelCatalogProvider) -> bool:
    api_base_url = (provider.api_base_url or "").strip().lower().rstrip("/")
    return api_base_url.endswith("/v1") or api_base_url.endswith("/api/v1")
