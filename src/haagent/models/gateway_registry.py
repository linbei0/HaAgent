"""
src/haagent/models/gateway_registry.py - 模型网关能力映射

负责把 profile 和公开模型目录映射为 HaAgent 当前可运行的 ModelGateway 能力。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from haagent.models.adapters.anthropic import AnthropicMessagesGateway
from haagent.models.adapters.google import GoogleGeminiGateway
from haagent.models.adapters.openai_chat import OpenAIChatCompletionsGateway
from haagent.models.adapters.openai_responses import OpenAIResponsesGateway
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.http_transport import ModelHttpTransport, close_model_gateway
from haagent.models.config.connections import ProviderProfileError
from haagent.models.model_ref import ModelInvocation, ResolvedModel
from haagent.models.model_settings import ModelSettings
from haagent.models.types import ModelGateway, ModelGatewayMetadata
from haagent.models.negotiating_gateway import NegotiatingModelGateway
from haagent.models.local_runtime import discover_lm_studio, discover_ollama
from haagent.models.capabilities import (
    ModelCapabilities,
    effective_input_window_tokens,
)
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


def gateway_from_resolved(
    model: ResolvedModel,
    *,
    retry_controller: RetryController | None = None,
    http_transport: ModelHttpTransport | None = None,
    capabilities: ModelCapabilities | None = None,
) -> ModelGateway:
    """从不可变运行时绑定创建 gateway；provider 细节留在 adapter 注册表。"""
    kwargs: dict[str, object] = {
        "api_key": model.credential.api_key or None,
        "model": model.ref.model,
        "base_url": model.base_url or None,
        "request_config": model.settings,
    }
    if retry_controller is not None:
        kwargs["retry_controller"] = retry_controller
    if http_transport is not None:
        kwargs["http_transport"] = http_transport
    require_key = model.runtime_kind == "remote"
    if model.provider == "openai":
        gateway: ModelGateway = OpenAIResponsesGateway(**kwargs, require_api_key=require_key)  # type: ignore[arg-type]
    elif model.provider == "openai-chat":
        gateway = OpenAIChatCompletionsGateway(**kwargs, require_api_key=require_key)  # type: ignore[arg-type]
    elif model.provider == "anthropic":
        gateway = AnthropicMessagesGateway(**kwargs)  # type: ignore[arg-type]
    elif model.provider == "google":
        gateway = GoogleGeminiGateway(**kwargs)  # type: ignore[arg-type]
    else:
        raise GatewayRegistryError(f"unsupported provider in resolved model: {model.provider}")
    # 审计字段挂在 ResolvedModel 上，不泄漏到 adapter 构造参数。
    return _with_window_capabilities(_AuditMetadataGateway(gateway, model), capabilities)


def gateway_from_route(
    primary_model: ResolvedModel,
    *,
    fallback_model: ResolvedModel | None = None,
    cloud_fallback_consent: bool = False,
    retry_controller: RetryController | None = None,
    route_event_sink=None,
    http_transport: ModelHttpTransport | None = None,
    primary_capabilities: ModelCapabilities | None = None,
    fallback_capabilities: ModelCapabilities | None = None,
) -> ModelGateway:
    """从 settings route 构造协商网关；同一 route 共享一个 ModelHttpTransport。"""
    # route 级共享 transport：primary / chat fallback / model fallback 复用连接池。
    shared_transport = http_transport or ModelHttpTransport()
    owns_shared = http_transport is None
    primary = _with_window_capabilities(
        _with_discovered_capabilities(
            gateway_from_resolved(
                primary_model,
                retry_controller=retry_controller,
                http_transport=shared_transport,
            ),
            primary_model,
            protocol="responses" if primary_model.provider == "openai" else None,
        ),
        primary_capabilities,
    )
    fallback = (
        _with_window_capabilities(
            _with_discovered_capabilities(
                gateway_from_resolved(
                    fallback_model,
                    retry_controller=retry_controller,
                    http_transport=shared_transport,
                ),
                fallback_model,
                protocol="responses" if fallback_model.provider == "openai" else None,
            ),
            fallback_capabilities,
        )
        if fallback_model is not None
        else None
    )
    primary_chat = None
    if primary_model.provider == "openai":
        # Responses 原生参数不能透传到 Chat Completions；协议 fallback 使用原有默认 payload。
        chat_binding = replace(primary_model, settings=ModelSettings.empty())
        primary_chat = _with_window_capabilities(
            _with_discovered_capabilities(
                _AuditMetadataGateway(
                    OpenAIChatCompletionsGateway(
                        api_key=primary_model.credential.api_key or None,
                        model=primary_model.ref.model,
                        base_url=primary_model.base_url,
                        retry_controller=retry_controller,
                        require_api_key=primary_model.runtime_kind == "remote",
                        http_transport=shared_transport,
                        request_config=ModelSettings.empty(),
                    ),
                    chat_binding,
                ),
                chat_binding,
                protocol="chat_completions",
            ),
            primary_capabilities,
        )
    return NegotiatingModelGateway(
        primary=primary,
        primary_chat=primary_chat,
        fallback=fallback,
        primary_runtime_kind=primary_model.runtime_kind,
        fallback_runtime_kind=fallback_model.runtime_kind if fallback_model else "remote",
        cloud_fallback_consent=cloud_fallback_consent,
        route_event_sink=route_event_sink,
        primary_connection=primary_model.ref.connection_id,
        fallback_connection=(fallback_model.ref.connection_id if fallback_model else None),
        http_transport=shared_transport if owns_shared else None,
    )


class _AuditMetadataGateway:
    """为 episode 附加脱敏 connection/model/variant/settings digest，不含 secret。"""

    def __init__(self, gateway: ModelGateway, model: ResolvedModel) -> None:
        self._gateway = gateway
        self._model = model
        self.provider_name = gateway.provider_name
        # turns 审计仍读私有 _retry_controller；包装层必须转发，避免 max_attempts 变 None。
        self._retry_controller = getattr(gateway, "_retry_controller", None)

    def capabilities(self):
        return self._gateway.capabilities()

    def metadata(self) -> ModelGatewayMetadata:
        base = self._gateway.metadata()
        request_config = dict(base.request_config or {})
        request_config["connection_id"] = self._model.ref.connection_id
        request_config["settings_digest"] = self._model.settings.digest
        if self._model.ref.variant is not None:
            request_config["variant"] = self._model.ref.variant
        return ModelGatewayMetadata(
            provider=base.provider,
            model=base.model or self._model.ref.model,
            endpoint=base.endpoint,
            base_url=base.base_url,
            profile_name=base.profile_name or self._model.ref.connection_id,
            context_window_tokens=base.context_window_tokens,
            request_config=request_config,
        )

    @property
    def model_settings(self):
        return self._gateway.model_settings

    def generate(self, invocation: ModelInvocation, **kwargs: object):
        return self._gateway.generate(invocation, **kwargs)

    def close(self) -> None:
        close_model_gateway(self._gateway)


class _CapabilityOverrideGateway:
    def __init__(self, gateway: ModelGateway, capabilities: ModelCapabilities) -> None:
        self._gateway = gateway
        self._capabilities = capabilities
        self.provider_name = gateway.provider_name
        self._retry_controller = getattr(gateway, "_retry_controller", None)

    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def metadata(self):
        base = self._gateway.metadata()
        return replace(
            base,
            context_window_tokens=effective_input_window_tokens(self._capabilities),
        )

    @property
    def model_settings(self):
        return self._gateway.model_settings

    def generate(self, invocation: ModelInvocation, **kwargs: object):
        return self._gateway.generate(invocation, **kwargs)

    def close(self) -> None:
        # 包装层转发 close，确保 route 关闭时底层 provider gateway 也能释放资源。
        close_model_gateway(self._gateway)


def _with_discovered_capabilities(
    gateway: ModelGateway,
    model_binding: ResolvedModel,
    *,
    protocol: str | None,
) -> ModelGateway:
    if model_binding.runtime_kind == "remote":
        return gateway
    discovery = discover_ollama() if model_binding.runtime_kind == "ollama" else discover_lm_studio()
    model = next((item for item in discovery.models if item.id == model_binding.ref.model), None)
    if model is None:
        return gateway
    capabilities = model.capabilities
    if protocol in {"responses", "chat_completions"}:
        capabilities = replace(capabilities, protocols=frozenset({protocol}))
    return _CapabilityOverrideGateway(gateway, capabilities)


def _with_window_capabilities(
    gateway: ModelGateway,
    window_capabilities: ModelCapabilities | None,
) -> ModelGateway:
    """只叠加目录或已发现的窗口事实，不覆盖 adapter 的其他能力。"""

    if window_capabilities is None:
        return gateway
    current = gateway.capabilities()
    context_window = _merge_known_window(
        current.context_window_tokens,
        window_capabilities.context_window_tokens,
    )
    input_window = _merge_known_window(
        current.input_window_tokens,
        window_capabilities.input_window_tokens,
    )
    if context_window == current.context_window_tokens and input_window == current.input_window_tokens:
        return gateway
    return _CapabilityOverrideGateway(
        gateway,
        replace(
            current,
            context_window_tokens=context_window,
            input_window_tokens=input_window,
        ),
    )


def _merge_known_window(current: int | None, incoming: int | None) -> int | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return min(current, incoming)


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
