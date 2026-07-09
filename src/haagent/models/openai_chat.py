"""
src/haagent/models/openai_chat.py - OpenAI Chat Completions 兼容网关

把 Chat Completions 兼容 API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.transport import (
    _chat_completions_stream_transport,
    _chat_completions_transport,
    _endpoint_base_url,
    _image_data_url,
    _normalize_chat_completions_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _usage_from_fields,
)
from haagent.models.types import (
    ModelCallError,
    ModelGatewayMetadata,
    ModelResponse,
    ModelUsage,
    StreamTransport,
    ToolCall,
    Transport,
)

def _chat_tool_schemas(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把内部工具 schema 转成 Chat Completions 的 function tool 格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            },
        }
        for schema in tool_schemas
    ]


def _parse_chat_completion_response(response: dict[str, object]) -> ModelResponse:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelCallError("OpenAI chat response did not include choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ModelCallError("OpenAI chat choice must be an object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ModelCallError("OpenAI chat choice did not include message")
    content = message.get("content", "")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise ModelCallError("OpenAI chat message content must be a string")
    return ModelResponse(
        content=content,
        tool_calls=_parse_chat_tool_calls(message.get("tool_calls")),
        usage=_parse_openai_chat_usage(response),
    )


def _parse_chat_tool_calls(raw_tool_calls: object) -> list[ToolCall]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise ModelCallError("OpenAI chat tool_calls must be a list when present")
    tool_calls: list[ToolCall] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ModelCallError("unsupported OpenAI chat tool_call item")
        if item.get("type") != "function":
            raise ModelCallError(
                f"unsupported OpenAI chat tool_call type: {item.get('type')}",
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise ModelCallError("OpenAI chat tool_call missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        tool_call_id = str(item.get("id") or "")
        tool_calls.append(ToolCall(name=name, args=_parse_tool_arguments(arguments), id=tool_call_id))
    return tool_calls


def _openai_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = _openai_chat_content_parts(content)
        converted.append(item)
    return converted


def _openai_chat_content_parts(content: list[Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ModelCallError("message content parts must be objects")
        if part.get("type") == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_attachment":
            parts.append({"type": "image_url", "image_url": {"url": _image_data_url(part)}})
        else:
            raise ModelCallError(f"unsupported message content part: {part.get('type')}")
    return parts


def _openai_chat_error_message(message: str) -> str:
    normalized = message.lower()
    if "image_url" in normalized and "expected" in normalized and "text" in normalized:
        return "当前模型或接口不支持图片输入，请切换到支持视觉的模型后重试。"
    return message
def _parse_openai_chat_usage(response: dict[str, object]) -> ModelUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return _usage_from_fields(
        usage,
        input_field="prompt_tokens",
        output_field="completion_tokens",
        total_field="total_tokens",
        raw_source="openai.chat_completions.usage",
    )

class OpenAIChatCompletionsGateway:
    provider_name = "openai-chat"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._chat_completions_endpoint = _normalize_chat_completions_endpoint(base_url)
        self._transport = transport or (
            lambda payload, api_key: _chat_completions_transport(
                payload,
                api_key,
                self._chat_completions_endpoint,
            )
        )
        self._stream_transport = stream_transport or (
            lambda payload, api_key, on_delta: _chat_completions_stream_transport(
                payload,
                api_key,
                self._chat_completions_endpoint,
                on_delta,
            )
        )

    @property
    def chat_completions_endpoint(self) -> str:
        """返回本次 gateway 会请求的 Chat Completions endpoint，便于审计和测试。"""
        return self._chat_completions_endpoint

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._chat_completions_endpoint),
            base_url=_endpoint_base_url(self._chat_completions_endpoint),
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        event_sink: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """调用 OpenAI Chat Completions 兼容 API，并归一化为 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError(
                "OPENAI_API_KEY is required for OpenAIChatCompletionsGateway",
            )

        payload: dict[str, object] = {
            "model": self._model,
            "messages": _openai_chat_messages(messages),
        }
        if tool_schemas:
            payload["tools"] = _chat_tool_schemas(tool_schemas)
            payload["parallel_tool_calls"] = True
        if event_sink is not None:
            payload["stream"] = True
        try:
            response = (
                self._stream_transport(payload, self._api_key, event_sink)
                if event_sink is not None
                else self._transport(payload, self._api_key)
            )
        except Exception as error:
            raise ModelCallError(_openai_chat_error_message(str(error))) from error
        return _parse_chat_completion_response(response)

