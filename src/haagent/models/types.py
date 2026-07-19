"""
src/haagent/models/types.py - 模型网关协议与公共 DTO

上层只依赖 ModelGateway 协议；真实 provider 失败必须显式暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol

from haagent.models.capabilities import ModelCapabilities
from haagent.models.model_ref import ModelInvocation
from haagent.models.telemetry import ModelTransportEvent
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.retry import RetryEvent, RetryFailure


ModelFailureCategory = Literal[
    "network",
    "timeout",
    "rate_limited",
    "server",
    "client",
    "auth",
    "quota_exhausted",
    "protocol",
    "response_parse",
    "stream_interrupted",
]

ModelTermination = Literal["completed", "tool_calls", "length", "content_filter", "unknown"]


@dataclass(frozen=True)
class ModelFailureDetails:
    """模型 transport 提供给网关的脱敏失败事实。"""

    category: ModelFailureCategory
    status_code: int | None = None
    provider_code: str | None = None
    retry_after_seconds: float | None = None
    request_id: str | None = None
    retryable: bool = False

    def to_retry_failure(self) -> RetryFailure:
        """将模型失败事实无损适配到统一重试内核。"""

        return RetryFailure(
            category=self.category,
            status_code=self.status_code,
            provider_code=self.provider_code,
            retry_after_seconds=self.retry_after_seconds,
            request_id=self.request_id,
            retryable=self.retryable,
        )

class ModelCallError(RuntimeError):
    """Raised when a model provider fails explicitly."""

    def __init__(self, message: str, *, details: ModelFailureDetails | None = None) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    raw_source: str = "unknown"
    # provider 的“当前输入上下文”口径；Anthropic 需要包含 cache 创建与读取 token。
    context_input_tokens: int | None = None


@dataclass(frozen=True)
class ModelGatewayMetadata:
    provider: str
    model: str | None
    endpoint: str | None
    base_url: str | None = None
    profile_name: str | None = None
    # HaAgent 本地有效输入窗口；不是 provider 请求参数。
    context_window_tokens: int | None = None
    # episode 审计用；不含 secret，只放选择状态与脱敏参数摘要。
    request_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: ModelUsage | None = None
    # provider 的停止原因经 adapter 归一化；runtime 据此拒绝截断或内容过滤的伪完成回答。
    termination: ModelTermination = "unknown"
    # runtime 只保存和透传；provider adapter 独占 payload 语义。
    provider_turn_state: "ProviderTurnState | None" = None


@dataclass(frozen=True)
class ProviderTurnState:
    provider: str
    payload: Mapping[str, Any]


class ModelGateway(Protocol):
    provider_name: str

    def generate(
        self,
        invocation: ModelInvocation,
        *,
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
        telemetry_sink: Callable[[ModelTransportEvent], None] | None = None,
    ) -> ModelResponse:
        """Generate a model response given a conversation messages list."""

    def metadata(self) -> ModelGatewayMetadata:
        """Return non-sensitive metadata for episode audit records."""

    def capabilities(self) -> ModelCapabilities:
        """Return tri-state capabilities used by route negotiation."""



Transport = Callable[[dict[str, object], str], dict[str, object]]
StreamTransport = Callable[[dict[str, object], str, Callable[[str], None]], dict[str, object]]
AnthropicTransport = Callable[[dict[str, object], str, str], dict[str, object]]
AnthropicStreamTransport = Callable[[dict[str, object], str, str, Callable[[str], None]], dict[str, object]]
GoogleGeminiTransport = Callable[[dict[str, object], str, str], dict[str, object]]
GoogleGeminiStreamTransport = Callable[[dict[str, object], str, str, Callable[[str], None]], dict[str, object]]
