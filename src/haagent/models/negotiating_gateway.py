"""
haagent/models/negotiating_gateway.py - 模型能力、协议与备用路由协商

在首个有效输出前执行显式协议降级或单一备用模型切换，并暴露脱敏事件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from haagent.models.capabilities import (
    ModelCapabilities,
    build_model_requirements,
    missing_capabilities,
)
from haagent.models.http_transport import ModelHttpTransport, close_model_gateway
from haagent.models.model_ref import ModelInvocation
from haagent.models.types import (
    ModelCallError,
    ModelGateway,
    ModelGatewayMetadata,
    ModelResponse,
)
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryEvent, RetryFailure

RouteEventKind = Literal["model_protocol_fallback", "model_fallback"]


@dataclass(frozen=True)
class RouteFallbackEvent:
    kind: RouteEventKind
    reason: str
    from_connection: str | None
    from_model: str | None
    from_protocol: str | None
    to_connection: str | None
    to_model: str | None
    to_protocol: str | None
    required_capabilities: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()


class NegotiatingModelGateway:
    """只在明确能力或可重试失败边界切换协议/模型。"""

    provider_name = "negotiating"

    def __init__(
        self,
        *,
        primary: ModelGateway,
        primary_chat: ModelGateway | None = None,
        fallback: ModelGateway | None = None,
        primary_connection: str | None = None,
        fallback_connection: str | None = None,
        primary_runtime_kind: str = "remote",
        fallback_runtime_kind: str = "remote",
        cloud_fallback_consent: bool = False,
        route_event_sink: Callable[[RouteFallbackEvent], None] | None = None,
        http_transport: ModelHttpTransport | None = None,
    ) -> None:
        self._primary = primary
        self._primary_chat = primary_chat
        self._fallback = fallback
        self._primary_connection = primary_connection
        self._fallback_connection = fallback_connection
        self._primary_runtime_kind = primary_runtime_kind
        self._fallback_runtime_kind = fallback_runtime_kind
        self._cloud_fallback_consent = cloud_fallback_consent
        self._route_event_sink = route_event_sink
        self._active = primary
        # 仅当 route factory 把共享 transport 所有权交给本 gateway 时非 None。
        self._http_transport = http_transport
        self._closed = False

    def set_route_event_sink(self, sink: Callable[[RouteFallbackEvent], None] | None) -> None:
        """由每轮 orchestrator 绑定事件 sink，避免 gateway 绕过 runtime bus。"""
        self._route_event_sink = sink

    def close(self) -> None:
        """幂等关闭：先关 route 共享 transport，再关有独立资源的子 gateway。"""

        if self._closed:
            return
        self._closed = True
        if self._http_transport is not None:
            self._http_transport.close()
            self._http_transport = None
        for gateway in (self._primary, self._primary_chat, self._fallback):
            if gateway is not None:
                close_model_gateway(gateway)

    def capabilities(self) -> ModelCapabilities:
        return _capabilities(self._active)

    def metadata(self) -> ModelGatewayMetadata:
        return self._active.metadata()

    @property
    def model_settings(self):
        return self._active.model_settings

    def generate(
        self,
        invocation: ModelInvocation,
        *,
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
        telemetry_sink=None,
    ) -> ModelResponse:
        if self._closed:
            # 关闭后拒绝新请求，避免复用已释放连接池。
            raise ModelCallError(
                "model gateway is closed",
                details=None,
            )
        requirements = build_model_requirements(
            messages=invocation.messages,
            tool_schemas=invocation.tool_schemas,
            streaming=event_sink is not None,
        )
        required = _required_names(requirements)
        primary_missing = missing_capabilities(requirements, _capabilities(self._primary))
        if primary_missing:
            return self._use_fallback(
                reason="primary_missing_capabilities",
                required=required,
                primary_missing=primary_missing,
                invocation=invocation,
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
                requirements=requirements,
                telemetry_sink=telemetry_sink,
            )

        emitted = False

        def tracked_sink(delta: str) -> None:
            nonlocal emitted
            if delta:
                emitted = True
            if event_sink is not None:
                event_sink(delta)

        sink = tracked_sink if event_sink is not None else None
        gateway = self._protocol_gateway_for_capabilities(required)
        if gateway is self._primary_chat:
            self._emit_protocol_event("responses_not_supported", required)
        try:
            return self._call(
                gateway,
                invocation,
                sink,
                cancellation_token,
                retry_event_sink,
                retry_exhausted_sink,
                telemetry_sink,
            )
        except RunCancelled:
            raise
        except ModelCallError as error:
            if (
                gateway is self._primary
                and self._primary_chat is not None
                and not emitted
                and _is_protocol_fallback_error(error)
            ):
                self._emit_protocol_event("responses_endpoint_unsupported", required)
                try:
                    return self._call(
                        self._primary_chat,
                        invocation,
                        sink,
                        cancellation_token,
                        retry_event_sink,
                        retry_exhausted_sink,
                        telemetry_sink,
                    )
                except RunCancelled:
                    raise
                except ModelCallError as chat_error:
                    error = chat_error
            if emitted or not _is_model_fallback_error(error):
                raise error
            return self._use_fallback(
                reason=_failure_reason(error),
                required=required,
                primary_missing=(),
                invocation=invocation,
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
                requirements=requirements,
                telemetry_sink=telemetry_sink,
            )

    def _protocol_gateway_for_capabilities(self, required: tuple[str, ...]) -> ModelGateway:
        protocols = _capabilities(self._primary).protocols
        if protocols and "responses" not in protocols and self._primary_chat is not None:
            return self._primary_chat
        return self._primary

    def _use_fallback(
        self,
        *,
        reason: str,
        required: tuple[str, ...],
        primary_missing: tuple[str, ...],
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None,
        cancellation_token: CancellationToken | None,
        retry_event_sink: Callable[[RetryEvent], None] | None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None,
        requirements: object,
        telemetry_sink=None,
    ) -> ModelResponse:
        if self._fallback is None:
            if primary_missing:
                raise ModelCallError(f"primary missing capabilities: {', '.join(primary_missing)}")
            raise ModelCallError(f"primary model failed without configured fallback: {reason}")
        if (
            self._primary_runtime_kind != "remote"
            and self._fallback_runtime_kind == "remote"
            and not self._cloud_fallback_consent
        ):
            # 本地内容进入云端必须由配置时的明确同意授权，运行时不静默放宽。
            raise ModelCallError("cloud fallback consent is required for local-to-remote routing")
        fallback_missing = missing_capabilities(requirements, _capabilities(self._fallback))  # type: ignore[arg-type]
        if fallback_missing:
            raise ModelCallError(
                "primary missing capabilities: "
                f"{', '.join(primary_missing) or 'none'}; fallback missing capabilities: "
                f"{', '.join(fallback_missing)}",
            )
        self._emit(
            RouteFallbackEvent(
                kind="model_fallback",
                reason=reason,
                from_connection=self._primary_connection,
                from_model=self._primary.metadata().model,
                from_protocol=_protocol(self._active),
                to_connection=self._fallback_connection,
                to_model=self._fallback.metadata().model,
                to_protocol=_protocol(self._fallback),
                required_capabilities=required,
                missing_capabilities=primary_missing,
            ),
        )
        self._active = self._fallback
        return self._call(
            self._fallback,
            invocation,
            event_sink,
            cancellation_token,
            retry_event_sink,
            retry_exhausted_sink,
            telemetry_sink,
        )

    def _emit_protocol_event(self, reason: str, required: tuple[str, ...]) -> None:
        assert self._primary_chat is not None
        self._emit(
            RouteFallbackEvent(
                kind="model_protocol_fallback",
                reason=reason,
                from_connection=self._primary_connection,
                from_model=self._primary.metadata().model,
                from_protocol="responses",
                to_connection=self._primary_connection,
                to_model=self._primary_chat.metadata().model,
                to_protocol="chat_completions",
                required_capabilities=required,
            ),
        )
        self._active = self._primary_chat

    def _emit(self, event: RouteFallbackEvent) -> None:
        if self._route_event_sink is not None:
            self._route_event_sink(event)

    @staticmethod
    def _call(
        gateway: ModelGateway,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None,
        cancellation_token: CancellationToken | None,
        retry_event_sink: Callable[[RetryEvent], None] | None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None,
        telemetry_sink=None,
    ) -> ModelResponse:
        kwargs = {
            "event_sink": event_sink,
            "cancellation_token": cancellation_token,
            "retry_event_sink": retry_event_sink,
            "retry_exhausted_sink": retry_exhausted_sink,
            "telemetry_sink": telemetry_sink,
        }
        return gateway.generate(invocation, **kwargs)


def _capabilities(gateway: ModelGateway) -> ModelCapabilities:
    method = getattr(gateway, "capabilities", None)
    return method() if callable(method) else ModelCapabilities()


def _required_names(requirements: object) -> tuple[str, ...]:
    names = []
    for name in ("tools", "streaming", "vision"):
        if getattr(requirements, name):
            names.append(name)
    names.append("context_window")
    return tuple(names)


def _is_protocol_fallback_error(error: ModelCallError) -> bool:
    return error.details is not None and error.details.status_code in {404, 405, 501}


def _is_model_fallback_error(error: ModelCallError) -> bool:
    details = error.details
    if details is None:
        return False
    if details.category in {"network", "timeout"}:
        return True
    if details.category == "rate_limited":
        return details.status_code == 429 and details.retryable
    return details.category == "server" and (details.status_code or 0) >= 500


def _failure_reason(error: ModelCallError) -> str:
    details = error.details
    return details.category if details is not None else "model_error"


def _protocol(gateway: ModelGateway) -> str | None:
    protocols = _capabilities(gateway).protocols
    if "responses" in protocols:
        return "responses"
    if "chat_completions" in protocols:
        return "chat_completions"
    return None
