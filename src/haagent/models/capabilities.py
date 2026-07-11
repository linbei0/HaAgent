"""
haagent/models/capabilities.py - 模型能力与本轮需求合同

提供确定性的能力三态、调用需求提取和缺失能力比较。
"""

from __future__ import annotations

from dataclasses import dataclass
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
    protocols: frozenset[ModelProtocol] = frozenset()


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
    context_window = capabilities.context_window_tokens
    if context_window is not None and requirements.estimated_input_tokens > context_window:
        missing.append("context_window")
    return tuple(missing)


def _contains_image(value: object) -> bool:
    if isinstance(value, list):
        return any(_contains_image(item) for item in value)
    if not isinstance(value, dict):
        return False
    content_type = value.get("type")
    if content_type in {"image_attachment", "input_image", "image_url"}:
        return True
    return any(_contains_image(item) for item in value.values())
