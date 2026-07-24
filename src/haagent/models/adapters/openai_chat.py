"""
src/haagent/models/adapters/openai_chat.py - OpenAI Chat Completions 兼容网关

把 Chat Completions 兼容 API 请求与输出归一化为统一 ModelResponse。
"""

from __future__ import annotations

import os
from typing import Any, Callable

from haagent.models.gateway_retry import default_retry_controller, execute_model_request, unexpected_model_error
from haagent.models.capabilities import ModelCapabilities
from haagent.models.http_transport import ModelHttpTransport
from haagent.models.model_options import merge_provider_payload
from haagent.models.model_ref import ModelInvocation
from haagent.models.model_settings import ModelSettings
from haagent.models.adapters.transport import (
    _chat_completions_stream_transport,
    _chat_completions_transport,
    _endpoint_base_url,
    _image_data_url,
    _invoke_transport,
    _normalize_chat_completions_endpoint,
    _parse_tool_arguments,
    _redact_url,
    _usage_from_fields,
)
from haagent.models.types import (
    ModelCallError,
    ModelFailureDetails,
    ModelGatewayMetadata,
    ModelResponse,
    ModelUsage,
    StreamTransport,
    ToolCall,
    Transport,
)
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryController, RetryEvent, RetryFailure


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
    tool_calls = _parse_chat_tool_calls(message.get("tool_calls"))
    if not tool_calls and _contains_embedded_tool_markup(content):
        # 某些 OpenAI-compatible 服务把内部 DSML/XML 工具协议塞进 content；
        # 这不是结构化工具调用，禁止把协议文本误当成最终回答或直接执行。
        raise ModelCallError(
            "OpenAI chat response contained embedded tool markup without structured tool_calls",
            details=ModelFailureDetails(category="protocol", retryable=False),
        )
    return ModelResponse(
        content=content,
        tool_calls=tool_calls,
        usage=_parse_openai_chat_usage(response),
        termination=_openai_chat_termination(first_choice.get("finish_reason"), bool(tool_calls)),
    )


def _openai_chat_termination(raw: object, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return {
        "stop": "completed",
        "length": "length",
        "content_filter": "content_filter",
    }.get(raw, "unknown")


def _contains_embedded_tool_markup(content: str) -> bool:
    """识别已知 provider 工具协议标记；不对普通 XML/Markdown 做宽泛猜测。"""

    return any(
        marker in content
        for marker in (
            "<｜｜DSML｜｜tool_calls>",
            "<｜｜DSML｜｜invoke",
            "<tool_call>",
            "</tool_call>",
        )
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
        retry_controller: RetryController | None = None,
        require_api_key: bool = True,
        http_transport: ModelHttpTransport | None = None,
        request_config: ModelSettings | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._require_api_key = require_api_key
        self._model = model
        self._request_config = request_config or ModelSettings.empty()
        self.model_settings = self._request_config
        self._chat_completions_endpoint = _normalize_chat_completions_endpoint(base_url)
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
                lambda payload, api_key, *, attempt=1, telemetry_sink=None: _chat_completions_transport(
                    payload,
                    api_key,
                    self._chat_completions_endpoint,
                    http_transport=bound,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                )
            )
        else:
            self._transport = transport
        if stream_transport is None:
            self._stream_transport = (
                lambda payload, api_key, on_delta, *, attempt=1, telemetry_sink=None: (
                    _chat_completions_stream_transport(
                        payload,
                        api_key,
                        self._chat_completions_endpoint,
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
            endpoint=_redact_url(self._chat_completions_endpoint),
            base_url=_endpoint_base_url(self._chat_completions_endpoint),
            request_config=self._request_config.to_traceable_dict(),
        )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tools="supported",
            streaming="supported",
            vision="supported",
            reasoning="unknown",
            tools_mode="native",
            protocols=frozenset({"chat_completions"}),
        )

    def generate(
        self,
        invocation: ModelInvocation,
        event_sink: Callable[[str], None] | None = None,
        cancellation_token: CancellationToken | None = None,
        retry_event_sink: Callable[[RetryEvent], None] | None = None,
        retry_exhausted_sink: Callable[[RetryFailure, int], None] | None = None,
        telemetry_sink=None,
        stream_reset_sink=None,
    ) -> ModelResponse:
        """调用 OpenAI Chat Completions 兼容 API，并归一化为 ModelResponse。"""
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        if self._require_api_key and not self._api_key:
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
        # settings 以 gateway 绑定为准；协议/模型 fallback 不得沿用调用方带入的 primary options。
        payload = merge_provider_payload(payload, self._request_config.options)
        try:
            response = execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=lambda on_delta, attempt: _invoke_transport(
                    self._transport,
                    self._stream_transport,
                    payload,
                    self._api_key or "",
                    on_delta=on_delta,
                    attempt=attempt,
                    telemetry_sink=telemetry_sink,
                ),
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
                telemetry_sink=telemetry_sink,
                stream_reset_sink=stream_reset_sink,
            )
        except ModelCallError as error:
            raise ModelCallError(
                _openai_chat_error_message(str(error)),
                details=error.details,
            ) from error
        except RunCancelled:
            raise
        except Exception as error:
            raise unexpected_model_error(
                error,
                message=_openai_chat_error_message(str(error)),
            ) from error
        return _parse_chat_completion_response(response)

