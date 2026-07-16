"""
tests/unit/models/test_negotiating_gateway.py - 模型协议与备用路由协商测试

验证能力、协议和失败类型只在明确允许的边界触发可审计回退。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from haagent.models.capabilities import ModelCapabilities
from haagent.models.negotiating_gateway import NegotiatingModelGateway, RouteFallbackEvent
from haagent.models.model_ref import ModelInvocation
from haagent.models.model_settings import ModelSettings
from haagent.models.types import (
    ModelCallError,
    ModelFailureDetails,
    ModelGatewayMetadata,
    ModelResponse,
)
from haagent.runtime.execution.cancellation import RunCancelled


@dataclass
class StubGateway:
    name: str
    model_capabilities: ModelCapabilities
    result: ModelResponse | Exception
    delta: str | None = None
    calls: int = 0

    provider_name = "stub"
    model_settings = ModelSettings.empty()

    def capabilities(self) -> ModelCapabilities:
        return self.model_capabilities

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(provider=self.name, model=self.name, endpoint=None)

    def generate(
        self,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None = None,
        **_: object,
    ) -> ModelResponse:
        self.calls += 1
        if self.delta is not None and event_sink is not None:
            event_sink(self.delta)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _caps(*, tools: str = "supported", protocols: set[str] | None = None) -> ModelCapabilities:
    return ModelCapabilities(
        tools=tools,  # type: ignore[arg-type]
        streaming="supported",
        vision="supported",
        tools_mode="native" if tools == "supported" else "none",
        protocols=frozenset(protocols or {"responses"}),  # type: ignore[arg-type]
    )


def _ok(name: str, *, caps: ModelCapabilities | None = None) -> StubGateway:
    return StubGateway(name, caps or _caps(), ModelResponse(content=name))


def _invoke(gateway, tools=None, **kwargs):
    return gateway.generate(ModelInvocation([], tools or [], ModelSettings.empty()), **kwargs)


def test_explicit_missing_capability_uses_fallback() -> None:
    primary = _ok("primary", caps=_caps(tools="unsupported"))
    fallback = _ok("fallback")
    events: list[RouteFallbackEvent] = []
    gateway = NegotiatingModelGateway(
        primary=primary,
        fallback=fallback,
        primary_runtime_kind="ollama",
        fallback_runtime_kind="lm_studio",
        route_event_sink=events.append,
    )

    response = _invoke(gateway, [{"name": "file_read"}])

    assert response.content == "fallback"
    assert primary.calls == 0
    assert events[0].kind == "model_fallback"
    assert events[0].missing_capabilities == ("tools",)


def test_unknown_capability_does_not_trigger_fallback() -> None:
    primary = _ok("primary", caps=_caps(tools="unknown"))
    fallback = _ok("fallback")
    gateway = NegotiatingModelGateway(primary=primary, fallback=fallback)

    assert _invoke(gateway, [{"name": "file_read"}]).content == "primary"
    assert fallback.calls == 0


def test_responses_unsupported_status_falls_back_to_chat_before_output() -> None:
    primary = StubGateway(
        "responses",
        _caps(protocols={"responses", "chat_completions"}),
        ModelCallError(
            "not implemented",
            details=ModelFailureDetails(category="client", status_code=501),
        ),
    )
    chat = _ok("chat", caps=_caps(protocols={"chat_completions"}))
    events: list[RouteFallbackEvent] = []
    gateway = NegotiatingModelGateway(
        primary=primary,
        primary_chat=chat,
        route_event_sink=events.append,
    )

    assert _invoke(gateway).content == "chat"
    assert events[0].kind == "model_protocol_fallback"


@pytest.mark.parametrize(
    "details",
    [
        ModelFailureDetails(category="network", retryable=True),
        ModelFailureDetails(category="timeout", retryable=True),
        ModelFailureDetails(category="rate_limited", status_code=429, retryable=True),
        ModelFailureDetails(category="server", status_code=503, retryable=True),
    ],
)
def test_retry_exhausted_transient_failure_uses_fallback(details: ModelFailureDetails) -> None:
    primary = StubGateway("primary", _caps(), ModelCallError("failed", details=details))
    fallback = _ok("fallback")
    gateway = NegotiatingModelGateway(primary=primary, fallback=fallback)

    assert _invoke(gateway).content == "fallback"


def test_partial_output_auth_and_cancellation_never_use_fallback() -> None:
    fallback = _ok("fallback")
    partial = StubGateway(
        "partial",
        _caps(),
        ModelCallError(
            "interrupted",
            details=ModelFailureDetails(category="stream_interrupted"),
        ),
        delta="started",
    )
    with pytest.raises(ModelCallError, match="interrupted"):
        _invoke(NegotiatingModelGateway(primary=partial, fallback=fallback), event_sink=lambda _: None)
    auth = StubGateway(
        "auth",
        _caps(),
        ModelCallError("unauthorized", details=ModelFailureDetails(category="auth", status_code=401)),
    )
    with pytest.raises(ModelCallError, match="unauthorized"):
        _invoke(NegotiatingModelGateway(primary=auth, fallback=fallback))
    cancelled = StubGateway("cancelled", _caps(), RunCancelled("cancelled"))
    with pytest.raises(RunCancelled):
        _invoke(NegotiatingModelGateway(primary=cancelled, fallback=fallback))
    assert fallback.calls == 0


def test_local_to_remote_fallback_requires_consent_and_fallback_capabilities() -> None:
    primary = _ok("primary", caps=_caps(tools="unsupported"))
    remote = _ok("remote")
    with pytest.raises(ModelCallError, match="cloud fallback consent"):
        _invoke(NegotiatingModelGateway(
            primary=primary,
            fallback=remote,
            primary_runtime_kind="ollama",
            fallback_runtime_kind="remote",
        ), [{"name": "shell"}])

    incapable = _ok("incapable", caps=_caps(tools="unsupported"))
    with pytest.raises(ModelCallError, match="primary missing.*fallback missing"):
        _invoke(NegotiatingModelGateway(primary=primary, fallback=incapable), [{"name": "shell"}])


@dataclass
class CapturingGateway(StubGateway):
    last_settings: ModelSettings | None = None

    def generate(
        self,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None = None,
        **kwargs: object,
    ) -> ModelResponse:
        self.last_settings = invocation.settings
        return super().generate(invocation, event_sink=event_sink, **kwargs)


def test_protocol_and_model_fallback_rebind_target_settings() -> None:
    primary_settings = ModelSettings.from_options({"reasoning": {"effort": "high"}})
    chat_settings = ModelSettings.empty()
    fallback_settings = ModelSettings.from_options({"temperature": 0.1})

    primary = CapturingGateway(
        "responses",
        _caps(protocols={"responses", "chat_completions"}),
        ModelCallError(
            "not implemented",
            details=ModelFailureDetails(category="client", status_code=501),
        ),
    )
    primary.model_settings = primary_settings
    chat = CapturingGateway("chat", _caps(protocols={"chat_completions"}), ModelResponse(content="chat"))
    chat.model_settings = chat_settings
    fallback = CapturingGateway("fallback", _caps(), ModelResponse(content="fallback"))
    fallback.model_settings = fallback_settings

    protocol_gateway = NegotiatingModelGateway(primary=primary, primary_chat=chat)
    protocol_gateway.generate(
        ModelInvocation([], [], primary_settings),
    )
    assert chat.last_settings is chat_settings
    assert chat.last_settings.options == {}

    failing_primary = CapturingGateway(
        "primary",
        _caps(),
        ModelCallError("down", details=ModelFailureDetails(category="network", retryable=True)),
    )
    failing_primary.model_settings = primary_settings
    model_gateway = NegotiatingModelGateway(primary=failing_primary, fallback=fallback)
    model_gateway.generate(ModelInvocation([], [], primary_settings))
    assert fallback.last_settings is fallback_settings
    assert fallback.last_settings.options == {"temperature": 0.1}
