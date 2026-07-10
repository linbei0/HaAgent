"""
src/haagent/models/openai_responses.py - OpenAI Responses API 网关

把 Responses API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.transport import (
    _endpoint_base_url,
    _image_data_url,
    _normalize_responses_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _responses_stream_transport,
    _responses_transport,
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
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryController, RetryEvent, RetryFailure

def _parse_tool_calls(response: dict[str, object]) -> list[ToolCall]:
    output = response.get("output")
    if output is None:
        return []
    if not isinstance(output, list):
        raise ModelCallError("OpenAI output must be a list when present")

    tool_calls: list[ToolCall] = []
    for item in output:
        # 当前只支持 Responses API 的最小 function_call 结构，避免误吞 provider 新格式。
        if not isinstance(item, dict):
            raise ModelCallError("unsupported OpenAI output item")
        output_type = item.get("type")
        if output_type in {"message", "output_text", "text"}:
            continue
        if output_type != "function_call":
            raise ModelCallError(f"unsupported OpenAI output type: {output_type}")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = item.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        call_id = str(item.get("call_id") or item.get("id") or "")
        tool_calls.append(ToolCall(name=name, args=_parse_tool_arguments(arguments), id=call_id))
    return tool_calls


def _parse_openai_responses_usage(response: dict[str, object]) -> ModelUsage | None:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return _usage_from_fields(
        usage,
        input_field="input_tokens",
        output_field="output_tokens",
        total_field="total_tokens",
        raw_source="openai.responses.usage",
    )


def _messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions messages to Responses API input format."""
    result = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            result.append({"role": "system", "content": msg.get("content", "")})
        elif role == "user":
            result.append({"role": "user", "content": _responses_user_content(msg.get("content", ""))})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": msg.get("content", "")}
            for tc in msg.get("tool_calls", []):
                result.append({
                    "type": "function_call",
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "call_id": tc.get("id", ""),
                })
            if item["content"]:
                result.append(item)
        elif role == "tool":
            result.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": msg.get("content", ""),
            })
    return result


def _responses_user_content(content: object) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ModelCallError("Responses user message content must be a string or content parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ModelCallError("Responses content parts must be objects")
        if part.get("type") == "text":
            parts.append({"type": "input_text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_attachment":
            parts.append({"type": "input_image", "image_url": _image_data_url(part)})
        else:
            raise ModelCallError(f"unsupported Responses content part: {part.get('type')}")
    return parts

class OpenAIResponsesGateway:
    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        retry_controller: RetryController | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        configured_base_url = (
            base_url
            if base_url is not None
            else os.environ.get("OPENAI_BASE_URL")
        )
        self._responses_endpoint = _normalize_responses_endpoint(configured_base_url)
        self._transport = transport or (
            lambda payload, api_key: _responses_transport(
                payload,
                api_key,
                self._responses_endpoint,
            )
        )
        self._stream_transport = stream_transport or (
            lambda payload, api_key, on_delta: _responses_stream_transport(
                payload,
                api_key,
                self._responses_endpoint,
                on_delta,
            )
        )
        self._retry_controller = default_retry_controller(retry_controller)

    @property
    def responses_endpoint(self) -> str:
        """返回本次 gateway 会请求的 Responses API endpoint，便于审计和测试。"""
        return self._responses_endpoint

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._responses_endpoint),
            base_url=_endpoint_base_url(self._responses_endpoint),
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
        """调用 OpenAI Responses API，并把 provider 输出收敛成统一 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError("OPENAI_API_KEY is required for OpenAIResponsesGateway")

        # Responses API uses "input" — convert messages to input format
        payload: dict[str, object] = {
            "model": self._model,
            "input": _messages_to_responses_input(messages),
        }
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["parallel_tool_calls"] = True
        if event_sink is not None:
            payload["stream"] = True
        try:
            response = execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=lambda on_delta: (
                    self._stream_transport(payload, self._api_key, on_delta)
                    if on_delta is not None
                    else self._transport(payload, self._api_key)
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

        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ModelCallError("OpenAI response did not include output_text")
        return ModelResponse(
            content=output_text,
            tool_calls=_parse_tool_calls(response),
            usage=_parse_openai_responses_usage(response),
        )
