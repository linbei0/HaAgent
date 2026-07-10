"""
src/haagent/models/anthropic.py - Anthropic Messages 网关

把 Anthropic Messages API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.transport import (
    _anthropic_stream_transport,
    _anthropic_transport,
    _endpoint_base_url,
    _image_base64,
    _image_mime_type,
    _normalize_anthropic_messages_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _usage_from_fields,
)
from haagent.models.types import (
    AnthropicStreamTransport,
    AnthropicTransport,
    ModelCallError,
    ModelGatewayMetadata,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryController, RetryEvent, RetryFailure

def _anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            if not isinstance(content, str):
                raise ModelCallError("Anthropic system message content must be a string")
            if content:
                system_parts.append(content)
            continue
        if role == "user":
            normalized.append({"role": "user", "content": _anthropic_user_content(content)})
            continue
        if role == "assistant":
            if not isinstance(content, str):
                raise ModelCallError("Anthropic assistant message content must be a string")
            normalized.append(_anthropic_assistant_message(content, message.get("tool_calls")))
            continue
        if role == "tool":
            tool_result_block = _anthropic_tool_result_block(message)
            if _is_anthropic_tool_result_message(normalized[-1] if normalized else None):
                normalized[-1]["content"].append(tool_result_block)
            else:
                normalized.append({"role": "user", "content": [tool_result_block]})
            continue
        raise ModelCallError(f"unsupported Anthropic message role: {role}")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, normalized


def _anthropic_user_content(content: object) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ModelCallError("Anthropic user message content must be a string or content parts")
    images: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ModelCallError("Anthropic content parts must be objects")
        if part.get("type") == "image_attachment":
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_mime_type(part),
                    "data": _image_base64(part),
                },
            })
        elif part.get("type") == "text":
            text = str(part.get("text", ""))
            if text:
                texts.append({"type": "text", "text": text})
        else:
            raise ModelCallError(f"unsupported Anthropic content part: {part.get('type')}")
    return [*images, *texts]


def _anthropic_assistant_message(content: str, raw_tool_calls: object) -> dict[str, Any]:
    if raw_tool_calls is None:
        return {"role": "assistant", "content": content}
    if not isinstance(raw_tool_calls, list):
        raise ModelCallError("Anthropic assistant tool_calls must be a list when present")
    blocks: list[dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ModelCallError("unsupported Anthropic assistant tool_call item")
        if item.get("type") != "function":
            raise ModelCallError(
                f"unsupported Anthropic assistant tool_call type: {item.get('type')}",
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise ModelCallError("Anthropic assistant tool_call missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        blocks.append({
            "type": "tool_use",
            "id": str(item.get("id") or ""),
            "name": name,
            "input": _parse_tool_arguments(arguments),
        })
    return {"role": "assistant", "content": blocks}


def _anthropic_tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = message.get("tool_call_id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ModelCallError("Anthropic tool result missing tool_call_id")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ModelCallError("Anthropic tool result content must be a string")
    return {
        "type": "tool_result",
        "tool_use_id": tool_call_id,
        "content": content,
    }


def _is_anthropic_tool_result_message(message: object) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(item, dict) and item.get("type") == "tool_result" for item in content)


def _anthropic_tool_schemas(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "input_schema": schema.get("parameters", {}),
        }
        for schema in tool_schemas
    ]


def _parse_anthropic_response(response: dict[str, object]) -> ModelResponse:
    content_blocks = response.get("content")
    if not isinstance(content_blocks, list):
        raise ModelCallError("Anthropic response did not include content blocks")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            raise ModelCallError("Anthropic content block must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ModelCallError("Anthropic text block must include text")
            text_parts.append(text)
            continue
        if block_type == "tool_use":
            name = block.get("name")
            if not isinstance(name, str) or not name:
                raise ModelCallError("Anthropic tool_use block missing name")
            raw_input = block.get("input")
            if not isinstance(raw_input, dict):
                raise ModelCallError("Anthropic tool_use block input must be an object")
            tool_id = str(block.get("id") or "")
            tool_calls.append(ToolCall(name=name, args=raw_input, id=tool_id))
            continue
        raise ModelCallError(f"unsupported Anthropic content block type: {block_type}")
    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=_parse_anthropic_usage(response),
    )


def _parse_anthropic_usage(response: dict[str, object]) -> ModelUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return _usage_from_fields(
        usage,
        input_field="input_tokens",
        output_field="output_tokens",
        total_field=None,
        raw_source="anthropic.messages.usage",
    )

class AnthropicMessagesGateway:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
        transport: AnthropicTransport | None = None,
        stream_transport: AnthropicStreamTransport | None = None,
        retry_controller: RetryController | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._messages_endpoint = _normalize_anthropic_messages_endpoint(base_url)
        self._transport = transport or _anthropic_transport
        self._stream_transport = stream_transport or _anthropic_stream_transport
        self._retry_controller = default_retry_controller(retry_controller)

    @property
    def messages_endpoint(self) -> str:
        """返回本次 gateway 会请求的 Anthropic Messages endpoint，便于审计和测试。"""
        return self._messages_endpoint

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._messages_endpoint),
            base_url=_endpoint_base_url(self._messages_endpoint),
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
    ) -> ModelResponse:
        """调用 Anthropic Messages API，并归一化为统一 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError("ANTHROPIC_API_KEY is required for AnthropicMessagesGateway")

        system, anthropic_messages = _anthropic_messages(messages)
        payload: dict[str, object] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if tool_schemas:
            payload["tools"] = _anthropic_tool_schemas(tool_schemas)
        if event_sink is not None:
            payload["stream"] = True
        try:
            response = execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=lambda on_delta: (
                    self._stream_transport(payload, self._api_key, self._messages_endpoint, on_delta)
                    if on_delta is not None
                    else self._transport(payload, self._api_key, self._messages_endpoint)
                ),
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
            )
        except ModelCallError:
            raise
        except RunCancelled:
            raise
        except Exception as error:
            raise unexpected_model_error(error) from error
        return _parse_anthropic_response(response)
