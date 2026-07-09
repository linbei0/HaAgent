"""
src/haagent/models/types.py - 模型网关协议与公共 DTO

上层只依赖 ModelGateway 协议；真实 provider 失败必须显式暴露。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

class ModelCallError(RuntimeError):
    """Raised when a model provider fails explicitly."""


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


@dataclass(frozen=True)


class ModelGatewayMetadata:
    provider: str
    model: str | None
    endpoint: str | None
    base_url: str | None = None
    profile_name: str | None = None


@dataclass(frozen=True)


class ModelResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: ModelUsage | None = None


class ModelGateway(Protocol):
    provider_name: str

    def generate(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        event_sink: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Generate a model response given a conversation messages list."""

    def metadata(self) -> ModelGatewayMetadata:
        """Return non-sensitive metadata for episode audit records."""



Transport = Callable[[dict[str, object], str], dict[str, object]]
StreamTransport = Callable[[dict[str, object], str, Callable[[str], None]], dict[str, object]]
AnthropicTransport = Callable[[dict[str, object], str, str], dict[str, object]]
AnthropicStreamTransport = Callable[[dict[str, object], str, str, Callable[[str], None]], dict[str, object]]
GoogleGeminiTransport = Callable[[dict[str, object], str, str], dict[str, object]]
GoogleGeminiStreamTransport = Callable[[dict[str, object], str, str, Callable[[str], None]], dict[str, object]]
