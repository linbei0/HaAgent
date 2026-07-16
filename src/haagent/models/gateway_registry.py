"""
src/haagent/models/gateway_registry.py - 模型网关能力映射

负责把 profile 和公开模型目录映射为 HaAgent 当前可运行的 ModelGateway 能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from haagent.models.anthropic import AnthropicMessagesGateway
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.google import GoogleGeminiGateway
from haagent.models.http_transport import ModelHttpTransport, close_model_gateway
from haagent.models.model_connections import ProviderProfile, ProviderProfileError
from haagent.models.model_options import empty_resolved_config
from haagent.models.openai_chat import OpenAIChatCompletionsGateway
from haagent.models.openai_responses import OpenAIResponsesGateway
from haagent.models.types import ModelGateway
from haagent.models.negotiating_gateway import NegotiatingModelGateway
from haagent.models.local_runtime import discover_lm_studio, discover_ollama
from haagent.models.capabilities import ModelCapabilities
from haagent.runtime.execution.retry import RetryController


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


def gateway_from_profile(
    profile: ProviderProfile,
    *,
    retry_controller: RetryController | None = None,
    http_transport: ModelHttpTransport | None = None,
) -> ModelGateway:
    # 空字符串与未配置等价，便于 CLI 临时 profile 回落到环境变量默认值。
    # request_config 在构造时绑定；切换 variant 时走新 gateway，不热改 payload。
    gateway_kwargs: dict[str, object] = {
        "api_key": profile.api_key or None,
        "model": profile.model,
        "base_url": profile.base_url or None,
        "request_config": profile.request_config,
    }
    if retry_controller is not None:
        gateway_kwargs["retry_controller"] = retry_controller
    if http_transport is not None:
        gateway_kwargs["http_transport"] = http_transport
    if profile.provider == "openai":
        return OpenAIResponsesGateway(
            **gateway_kwargs,  # type: ignore[arg-type]
            require_api_key=profile.runtime_kind == "remote",
        )
    if profile.provider == "openai-chat":
        return OpenAIChatCompletionsGateway(
            **gateway_kwargs,  # type: ignore[arg-type]
            require_api_key=profile.runtime_kind == "remote",
        )
    if profile.provider == "anthropic":
        return AnthropicMessagesGateway(**gateway_kwargs)  # type: ignore[arg-type]
    if profile.provider == "google":
        return GoogleGeminiGateway(**gateway_kwargs)  # type: ignore[arg-type]
    raise GatewayRegistryError(f"unsupported provider in profile: {profile.provider}")


def gateway_from_route(
    primary_profile: ProviderProfile,
    *,
    fallback_profile: ProviderProfile | None = None,
    cloud_fallback_consent: bool = False,
    retry_controller: RetryController | None = None,
    route_event_sink=None,
    http_transport: ModelHttpTransport | None = None,
) -> ModelGateway:
    """从 settings route 构造协商网关；同一 route 共享一个 ModelHttpTransport。"""
    # route 级共享 transport：primary / chat fallback / model fallback 复用连接池。
    shared_transport = http_transport or ModelHttpTransport()
    owns_shared = http_transport is None
    primary = _with_discovered_capabilities(
        gateway_from_profile(
            primary_profile,
            retry_controller=retry_controller,
            http_transport=shared_transport,
        ),
        primary_profile,
        protocol="responses" if primary_profile.provider == "openai" else None,
    )
    fallback = (
        _with_discovered_capabilities(
            gateway_from_profile(
                fallback_profile,
                retry_controller=retry_controller,
                http_transport=shared_transport,
            ),
            fallback_profile,
            protocol="responses" if fallback_profile.provider == "openai" else None,
        )
        if fallback_profile is not None
        else None
    )
    primary_chat = None
    if primary_profile.provider == "openai":
        # Responses 原生参数不能透传到 Chat Completions；协议 fallback 使用原有默认 payload。
        primary_chat = _with_discovered_capabilities(
            OpenAIChatCompletionsGateway(
                api_key=primary_profile.api_key or None,
                model=primary_profile.model,
                base_url=primary_profile.base_url,
                retry_controller=retry_controller,
                require_api_key=primary_profile.runtime_kind == "remote",
                http_transport=shared_transport,
                request_config=empty_resolved_config(
                    connection_id=primary_profile.name.split(":", 1)[0],
                    model_id=primary_profile.model,
                    variant=primary_profile.variant,
                ),
            ),
            primary_profile,
            protocol="chat_completions",
        )
    return NegotiatingModelGateway(
        primary=primary,
        primary_chat=primary_chat,
        fallback=fallback,
        primary_runtime_kind=primary_profile.runtime_kind,
        fallback_runtime_kind=fallback_profile.runtime_kind if fallback_profile else "remote",
        cloud_fallback_consent=cloud_fallback_consent,
        route_event_sink=route_event_sink,
        primary_connection=primary_profile.name.split(":", 1)[0],
        fallback_connection=(fallback_profile.name.split(":", 1)[0] if fallback_profile else None),
        http_transport=shared_transport if owns_shared else None,
    )


class _CapabilityOverrideGateway:
    def __init__(self, gateway: ModelGateway, capabilities: ModelCapabilities) -> None:
        self._gateway = gateway
        self._capabilities = capabilities
        self.provider_name = gateway.provider_name

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def metadata(self):
        return self._gateway.metadata()

    def generate(self, *args, **kwargs):
        return self._gateway.generate(*args, **kwargs)

    def close(self) -> None:
        # 包装层转发 close，确保 route 关闭时底层 provider gateway 也能释放资源。
        close_model_gateway(self._gateway)


def _with_discovered_capabilities(
    gateway: ModelGateway,
    profile: ProviderProfile,
    *,
    protocol: str | None,
) -> ModelGateway:
    if profile.runtime_kind == "remote":
        return gateway
    discovery = discover_ollama() if profile.runtime_kind == "ollama" else discover_lm_studio()
    model = next((item for item in discovery.models if item.id == profile.model), None)
    if model is None:
        return gateway
    capabilities = model.capabilities
    if protocol in {"responses", "chat_completions"}:
        capabilities = replace(capabilities, protocols=frozenset({protocol}))
    return _CapabilityOverrideGateway(gateway, capabilities)


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
