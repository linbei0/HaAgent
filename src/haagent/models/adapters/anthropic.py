"""
src/haagent/models/adapters/anthropic.py - Anthropic Messages 网关

把 Anthropic Messages API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.http_transport import ModelHttpTransport
from haagent.models.model_options import merge_provider_payload
from haagent.models.model_ref import ModelInvocation
from haagent.models.model_settings import ModelSettings
from haagent.models.adapters.transport import (
    _anthropic_stream_transport,
    _anthropic_transport,
    _endpoint_base_url,
    _image_base64,
    _image_mime_type,
    _invoke_transport,
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
    ProviderTurnState,
    ToolCall,
)
from haagent.models.capabilities import ModelCapabilities
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
            normalized.append(
                _anthropic_assistant_message(
                    content,
                    message.get("tool_calls"),
                    message.get("provider_turn_state"),
                ),
            )
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


def _anthropic_assistant_message(
    content: str,
    raw_tool_calls: object,
    provider_turn_state: object = None,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    # continuation 保存完整原始 block 序列；存在时不能再从规范化字段重组。
    if isinstance(provider_turn_state, dict) and provider_turn_state.get("provider") == "anthropic":
        payload = provider_turn_state.get("payload")
        continuation_blocks = payload.get("blocks") if isinstance(payload, dict) else None
        if isinstance(continuation_blocks, list):
            return {
                "role": "assistant",
                "content": [dict(block) for block in continuation_blocks if isinstance(block, dict)],
            }
    if raw_tool_calls is None:
        if content:
            blocks.append({"type": "text", "text": content})
        if blocks:
            return {"role": "assistant", "content": blocks}
        return {"role": "assistant", "content": content}
    if not isinstance(raw_tool_calls, list):
        raise ModelCallError("Anthropic assistant tool_calls must be a list when present")
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
    continuation_blocks: list[dict[str, Any]] = []
    has_thinking = False
    for block in content_blocks:
        if not isinstance(block, dict):
            raise ModelCallError("Anthropic content block must be an object")
        block_type = block.get("type")
        continuation_blocks.append(dict(block))
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ModelCallError("Anthropic text block must include text")
            text_parts.append(text)
            continue
        if block_type == "thinking":
            # 不展示 CoT；仅作为 opaque continuation 保留。
            has_thinking = True
            continue
        if block_type == "redacted_thinking":
            has_thinking = True
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
    continuation = (
        {"kind": "anthropic_messages", "blocks": continuation_blocks}
        if has_thinking
        else None
    )
    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=_parse_anthropic_usage(response),
        provider_turn_state=(ProviderTurnState("anthropic", continuation) if continuation else None),
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

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tools="supported",
            streaming="supported",
            vision="supported",
            reasoning="unknown",
            tools_mode="native",
        )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
        transport: AnthropicTransport | None = None,
        stream_transport: AnthropicStreamTransport | None = None,
        retry_controller: RetryController | None = None,
        http_transport: ModelHttpTransport | None = None,
        request_config: ModelSettings | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._request_config = request_config or ModelSettings.empty()
        self.model_settings = self._request_config
        self._messages_endpoint = _normalize_anthropic_messages_endpoint(base_url)
        # 仅在需要默认 HTTP path 时创建/绑定 transport；外部注入不越权关闭。
        self._owns_http_transport = False
        self._http_transport: ModelHttpTransport | None = http_transport
        needs_default_http = transport is None or stream_transport is None
        if needs_default_http and self._http_transport is None:
            self._http_transport = ModelHttpTransport()
            self._owns_http_transport = True
        bound = self._http_transport
        if transport is None:
            self._transport = (
                lambda payload, api_key, endpoint, *, attempt=1, telemetry_sink=None: _anthropic_transport(
                    payload,
                    api_key,
                    endpoint,
                    http_transport=bound,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                )
            )
        else:
            self._transport = transport
        if stream_transport is None:
            self._stream_transport = (
                lambda payload, api_key, endpoint, on_delta, *, attempt=1, telemetry_sink=None: (
                    _anthropic_stream_transport(
                        payload,
                        api_key,
                        endpoint,
                        on_delta,
                        http_transport=bound,
                        attempt=attempt,
                        telemetry_sink=telemetry_sink,
                    )
                )
            )
        else:
            self._stream_transport = stream_transport
        self._retry_controller = default_retry_controller(retry_controller)

    def close(self) -> None:
        """仅关闭自身创建的共享 transport；注入资源保持外部所有权。"""

        if self._owns_http_transport and self._http_transport is not None:
            self._http_transport.close()
            self._owns_http_transport = False

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._messages_endpoint),
            base_url=_endpoint_base_url(self._messages_endpoint),
            request_config=self._request_config.to_traceable_dict(),
        )

    def generate(
        self,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
        telemetry_sink=None,
    ) -> ModelResponse:
        """调用 Anthropic Messages API，并归一化为统一 ModelResponse。"""
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        if not self._api_key:
            raise ModelCallError("ANTHROPIC_API_KEY is required for AnthropicMessagesGateway")

        system, anthropic_messages = _anthropic_messages(messages)
        # 未配置时保持历史默认 max_tokens=4096；用户 options 可覆盖。
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
        # settings 以 gateway 绑定为准；fallback 目标不得继承其他模型的 options。
        payload = merge_provider_payload(payload, self._request_config.options)
        try:
            response = execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=lambda on_delta, attempt: _invoke_transport(
                    self._transport,
                    self._stream_transport,
                    payload,
                    self._api_key,
                    self._messages_endpoint,
                    on_delta=on_delta,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                ),
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
                telemetry_sink=telemetry_sink,
            )
        except ModelCallError:
            raise
        except RunCancelled:
            raise
        except Exception as error:
            raise unexpected_model_error(error) from error
        return _parse_anthropic_response(response)
