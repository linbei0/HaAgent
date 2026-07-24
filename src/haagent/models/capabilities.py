"""
haagent/models/capabilities.py - 模型能力与本轮需求合同

提供确定性的能力三态、调用需求提取和缺失能力比较。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from haagent.context.compression.budget import estimate_message_tokens

CapabilityState = Literal["supported", "unsupported", "unknown"]
ToolsMode = Literal["native", "compat", "none"]
ModelProtocol = Literal["responses", "chat_completions"]


@dataclass(frozen=True)
class ModelCapabilities:
    tools: CapabilityState = "unknown"
    streaming: CapabilityState = "unknown"
    vision: CapabilityState = "unknown"
    reasoning: CapabilityState = "unknown"
    tools_mode: ToolsMode = "none"
    context_window_tokens: int | None = None
    input_window_tokens: int | None = None
    protocols: frozenset[ModelProtocol] = frozenset()
    # 仅官方 Responses + 显式 background 时 supported；兼容端点不得猜测。
    background_response_retrieval: CapabilityState = "unknown"


@dataclass(frozen=True)
class ModelRequirements:
    tools: bool
    streaming: bool
    vision: bool
    estimated_input_tokens: int


def build_model_requirements(
    *,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    streaming: bool,
) -> ModelRequirements:
    token_messages = [*messages]
    if tool_schemas:
        token_messages.append({"role": "system", "tools": tool_schemas})
    return ModelRequirements(
        tools=bool(tool_schemas),
        streaming=streaming,
        vision=any(_contains_image(message.get("content")) for message in messages),
        estimated_input_tokens=estimate_message_tokens(token_messages),
    )


def missing_capabilities(
    requirements: ModelRequirements,
    capabilities: ModelCapabilities,
) -> tuple[str, ...]:
    missing: list[str] = []
    if requirements.tools and capabilities.tools == "unsupported":
        missing.append("tools")
    if requirements.streaming and capabilities.streaming == "unsupported":
        missing.append("streaming")
    if requirements.vision and capabilities.vision == "unsupported":
        missing.append("vision")
    context_window = effective_input_window_tokens(capabilities)
    if context_window is not None and requirements.estimated_input_tokens > context_window:
        missing.append("context_window")
    return tuple(missing)


def effective_input_window_tokens(capabilities: ModelCapabilities) -> int | None:
    """返回模型实际输入上限；独立 input limit 优先于总 context window。"""

    if _positive_int(capabilities.input_window_tokens):
        return capabilities.input_window_tokens
    if _positive_int(capabilities.context_window_tokens):
        return capabilities.context_window_tokens
    return None


def apply_context_window_limit(
    capabilities: ModelCapabilities | None,
    max_context_tokens: int | None,
) -> ModelCapabilities | None:
    """将用户本地上限与 provider/discovery 窗口取最小值。"""

    if max_context_tokens is None:
        return capabilities
    if not isinstance(max_context_tokens, int) or isinstance(max_context_tokens, bool) or max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be a positive integer")
    current = capabilities or ModelCapabilities()
    # 目录只提供一个窗口字段时，另一个字段必须继承该约束，不能凭空放宽输入上限。
    input_window = min(
        effective_input_window_tokens(current) or max_context_tokens,
        max_context_tokens,
    )
    context_window = min(
        current.context_window_tokens
        if _positive_int(current.context_window_tokens)
        else input_window,
        max_context_tokens,
    )
    return replace(
        current,
        context_window_tokens=context_window,
        input_window_tokens=input_window,
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _contains_image(value: object) -> bool:
    if isinstance(value, list):
        return any(_contains_image(item) for item in value)
    if not isinstance(value, dict):
        return False
    content_type = value.get("type")
    if content_type in {"image_attachment", "input_image", "image_url"}:
        return True
    return any(_contains_image(item) for item in value.values())
