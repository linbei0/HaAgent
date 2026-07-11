"""
haagent/models/fake.py - 测试用模型网关

提供确定性 fake model，方便测试 orchestrator 和工具链路。
"""

from __future__ import annotations

from typing import Any

from haagent.models.types import ModelGatewayMetadata, ModelResponse, ToolCall
from haagent.models.capabilities import ModelCapabilities


class FakeModelGateway:
    provider_name = "fake"

    def __init__(self, response: ModelResponse | None = None) -> None:
        self._response = response or ModelResponse(
            content="Use the fake tool for the MVP execution step.",
            tool_calls=[ToolCall(name="fake_tool", args={}, id="call_fake0001")],
        )
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tool_schemas": list(tool_schemas),
                # legacy key kept for tests that read model_input
                "model_input": _extract_model_input(messages),
            },
        )
        if self._response.tool_calls and not _tool_schema_available(tool_schemas, "fake_tool"):
            return ModelResponse(
                content="Fake model has no fake_tool available; relying on verification.",
                tool_calls=[],
            )
        tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
        if tool_result_count > 0 and self._response.tool_calls:
            return ModelResponse(content="Fake model observed tool results.", tool_calls=[])
        return self._response

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model="fake-model",
            endpoint=None,
            base_url=None,
            profile_name=None,
        )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tools="supported",
            streaming="supported",
            vision="unknown",
            reasoning="unknown",
            tools_mode="native",
            protocols=frozenset({"responses", "chat_completions"}),
        )


def _tool_schema_available(tool_schemas: list[dict[str, Any]], tool_name: str) -> bool:
    if not tool_schemas:
        return True
    return any(schema.get("name") == tool_name for schema in tool_schemas)


def _extract_model_input(messages: list[dict[str, Any]]) -> str:
    """Return concatenated text content of all messages for backward-compat test inspection."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
    return "\n".join(parts)
