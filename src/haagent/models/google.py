"""
src/haagent/models/google.py - Google Gemini 网关

把 Gemini generateContent API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.transport import (
    _endpoint_base_url,
    _google_gemini_stream_transport,
    _google_gemini_transport,
    _image_base64,
    _image_mime_type,
    _normalize_gemini_generate_content_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _usage_from_fields,
)
from haagent.models.types import (
    GoogleGeminiStreamTransport,
    GoogleGeminiTransport,
    ModelCallError,
    ModelGatewayMetadata,
    ModelResponse,
    ModelUsage,
    ToolCall,
)
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryController, RetryEvent, RetryFailure

def _gemini_contents(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        raw_role = message.get("role")
        content = message.get("content", "")
        if raw_role == "system":
            if not isinstance(content, str):
                raise ModelCallError("Gemini system message content must be a string")
            if content:
                system_parts.append(content)
            continue
        if raw_role == "user":
            contents.append({"role": "user", "parts": _gemini_user_parts(content)})
            continue
        if raw_role == "assistant":
            if not isinstance(content, str):
                raise ModelCallError("Gemini assistant message content must be a string")
            contents.append(_gemini_assistant_content(content, message.get("tool_calls")))
            continue
        if raw_role == "model":
            if not isinstance(content, str):
                raise ModelCallError("Gemini model message content must be a string")
            contents.append({"role": "model", "parts": [{"text": content}]})
            continue
        if raw_role == "tool":
            contents.append(_gemini_tool_result_content(message))
            continue
        raise ModelCallError(f"unsupported Gemini message role: {raw_role}")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, contents


def _gemini_user_parts(content: object) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        raise ModelCallError("Gemini user message content must be a string or content parts")
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ModelCallError("Gemini content parts must be objects")
        if part.get("type") == "text":
            parts.append({"text": str(part.get("text", ""))})
        elif part.get("type") == "image_attachment":
            parts.append({
                "inline_data": {
                    "mime_type": _image_mime_type(part),
                    "data": _image_base64(part),
                },
            })
        else:
            raise ModelCallError(f"unsupported Gemini content part: {part.get('type')}")
    return parts


def _gemini_assistant_content(content: str, raw_tool_calls: object) -> dict[str, Any]:
    if raw_tool_calls is None:
        return {"role": "model", "parts": [{"text": content}]}
    if not isinstance(raw_tool_calls, list):
        raise ModelCallError("Gemini assistant tool_calls must be a list when present")
    parts: list[dict[str, Any]] = []
    if content:
        parts.append({"text": content})
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            raise ModelCallError("unsupported Gemini assistant tool_call item")
        if item.get("type") != "function":
            raise ModelCallError(
                f"unsupported Gemini assistant tool_call type: {item.get('type')}",
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise ModelCallError("Gemini assistant tool_call missing function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ModelCallError("missing tool name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ModelCallError("missing tool arguments")
        parts.append({
            "functionCall": {
                "name": name,
                "args": _parse_tool_arguments(arguments),
            }
        })
    return {"role": "model", "parts": parts}


def _gemini_tool_result_content(message: dict[str, Any]) -> dict[str, Any]:
    name = message.get("name")
    if not isinstance(name, str) or not name:
        raise ModelCallError("Gemini tool result missing name")
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ModelCallError("Gemini tool result content must be a string")
    return {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": name,
                    "response": {"content": content},
                }
            }
        ],
    }


def _gemini_tool_schemas(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "functionDeclarations": [
                {
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {}),
                }
            ]
        }
        for schema in tool_schemas
    ]


def _parse_gemini_response(response: dict[str, object]) -> ModelResponse:
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ModelCallError("Gemini response did not include candidates")
    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise ModelCallError("Gemini candidate must be an object")
    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise ModelCallError("Gemini candidate did not include content")
    parts = content.get("parts")
    if not isinstance(parts, list):
        raise ModelCallError("Gemini content did not include parts")

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for part in parts:
        if not isinstance(part, dict):
            raise ModelCallError("Gemini part must be an object")
        if "text" in part:
            text = part["text"]
            if not isinstance(text, str):
                raise ModelCallError("Gemini text part must be a string")
            text_parts.append(text)
            continue
        if "functionCall" in part:
            function_call = part["functionCall"]
            if not isinstance(function_call, dict):
                raise ModelCallError("Gemini functionCall must be an object")
            name = function_call.get("name")
            if not isinstance(name, str) or not name:
                raise ModelCallError("Gemini functionCall missing name")
            args = function_call.get("args", {})
            if not isinstance(args, dict):
                raise ModelCallError("Gemini functionCall args must be an object")
            tool_calls.append(ToolCall(name=name, args=args))
            continue
        raise ModelCallError("unsupported Gemini part")
    return ModelResponse(
        content="".join(text_parts),
        tool_calls=tool_calls,
        usage=_parse_gemini_usage(response),
    )


def _parse_gemini_usage(response: dict[str, object]) -> ModelUsage | None:
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        return None
    return _usage_from_fields(
        usage,
        input_field="promptTokenCount",
        output_field="candidatesTokenCount",
        total_field="totalTokenCount",
        raw_source="google.gemini.usageMetadata",
    )

class GoogleGeminiGateway:
    provider_name = "google"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
        base_url: str | None = None,
        transport: GoogleGeminiTransport | None = None,
        stream_transport: GoogleGeminiStreamTransport | None = None,
        retry_controller: RetryController | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._model = model
        self._endpoint = _normalize_gemini_generate_content_endpoint(base_url, model)
        self._transport = transport or _google_gemini_transport
        self._stream_transport = stream_transport or _google_gemini_stream_transport
        self._retry_controller = default_retry_controller(retry_controller)

    @property
    def generate_content_endpoint(self) -> str:
        """返回本次 gateway 会请求的 Gemini generateContent endpoint，便于审计和测试。"""
        return self._endpoint

    def metadata(self) -> ModelGatewayMetadata:
        return ModelGatewayMetadata(
            provider=self.provider_name,
            model=self._model,
            endpoint=_redact_url(self._endpoint),
            base_url=_endpoint_base_url(self._endpoint),
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
        """调用 Gemini generateContent API，并归一化为统一 ModelResponse。"""
        if not self._api_key:
            raise ModelCallError("GEMINI_API_KEY is required for GoogleGeminiGateway")

        system_instruction, contents = _gemini_contents(messages)
        payload: dict[str, object] = {
            "contents": contents,
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tool_schemas:
            payload["tools"] = _gemini_tool_schemas(tool_schemas)
        if event_sink is not None:
            payload["stream"] = True
        try:
            response = execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=lambda on_delta: (
                    self._stream_transport(payload, self._api_key, self._endpoint, on_delta)
                    if on_delta is not None
                    else self._transport(payload, self._api_key, self._endpoint)
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
        return _parse_gemini_response(response)
