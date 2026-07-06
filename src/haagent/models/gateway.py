"""
haagent/models/gateway.py - 统一模型网关接口

上层只依赖 ModelGateway 协议；真实 provider 失败必须显式暴露。
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import urllib.error
import urllib.request
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
DEFAULT_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class OpenAIResponsesGateway:
    provider_name = "openai"

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
        if event_sink is not None:
            payload["stream"] = True
        try:
            response = (
                self._stream_transport(payload, self._api_key, event_sink)
                if event_sink is not None
                else self._transport(payload, self._api_key)
            )
        except Exception as error:
            raise ModelCallError(str(error)) from error

        output_text = response.get("output_text")
        if not isinstance(output_text, str):
            raise ModelCallError("OpenAI response did not include output_text")
        return ModelResponse(
            content=output_text,
            tool_calls=_parse_tool_calls(response),
            usage=_parse_openai_responses_usage(response),
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


class AnthropicMessagesGateway:
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        base_url: str | None = None,
        transport: AnthropicTransport | None = None,
        stream_transport: AnthropicStreamTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model
        self._messages_endpoint = _normalize_anthropic_messages_endpoint(base_url)
        self._transport = transport or _anthropic_transport
        self._stream_transport = stream_transport or _anthropic_stream_transport

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
            response = (
                self._stream_transport(payload, self._api_key, self._messages_endpoint, event_sink)
                if event_sink is not None
                else self._transport(payload, self._api_key, self._messages_endpoint)
            )
        except Exception as error:
            raise ModelCallError(str(error)) from error
        return _parse_anthropic_response(response)


class GoogleGeminiGateway:
    provider_name = "google"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
        base_url: str | None = None,
        transport: GoogleGeminiTransport | None = None,
        stream_transport: GoogleGeminiStreamTransport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._model = model
        self._endpoint = _normalize_gemini_generate_content_endpoint(base_url, model)
        self._transport = transport or _google_gemini_transport
        self._stream_transport = stream_transport or _google_gemini_stream_transport

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
            response = (
                self._stream_transport(payload, self._api_key, self._endpoint, event_sink)
                if event_sink is not None
                else self._transport(payload, self._api_key, self._endpoint)
            )
        except Exception as error:
            raise ModelCallError(str(error)) from error
        return _parse_gemini_response(response)


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ModelCallError("invalid tool arguments JSON") from error
    if not isinstance(parsed, dict):
        raise ModelCallError("tool arguments must be a JSON object")
    return parsed


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


def _usage_from_fields(
    usage: dict[str, object],
    *,
    input_field: str,
    output_field: str,
    total_field: str | None,
    raw_source: str,
) -> ModelUsage | None:
    input_tokens = _optional_int(usage.get(input_field))
    output_tokens = _optional_int(usage.get(output_field))
    total_tokens = _optional_int(usage.get(total_field)) if total_field is not None else None
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw_source=raw_source,
    )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _image_data_url(part: dict[str, Any]) -> str:
    return f"data:{_image_mime_type(part)};base64,{_image_base64(part)}"


def _image_base64(part: dict[str, Any]) -> str:
    path = part.get("path")
    if not isinstance(path, str) or not path:
        raise ModelCallError("image attachment missing path")
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError as error:
        raise ModelCallError(f"failed to read image attachment: {error}") from error


def _image_mime_type(part: dict[str, Any]) -> str:
    mime_type = part.get("mime_type")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise ModelCallError("image attachment missing mime_type")
    return mime_type


def _normalize_responses_endpoint(base_url: str | None) -> str:
    """把裸域名或 /v1 base URL 规范化为 Responses API endpoint。"""
    if base_url is None or not base_url.strip():
        return DEFAULT_RESPONSES_ENDPOINT
    endpoint = (_redact_url(base_url) or "").rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    if endpoint.endswith("/v1/responses"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/responses"
    return f"{endpoint}/v1/responses"


def _normalize_chat_completions_endpoint(base_url: str | None) -> str:
    """把裸域名或 /v1 base URL 规范化为 Chat Completions endpoint。"""
    if base_url is None or not base_url.strip():
        return DEFAULT_CHAT_COMPLETIONS_ENDPOINT
    endpoint = (_redact_url(base_url) or "").rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    if endpoint.endswith("/v1/chat/completions"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/chat/completions"
    return f"{endpoint}/v1/chat/completions"


def _normalize_anthropic_messages_endpoint(base_url: str | None) -> str:
    """把裸域名或 /v1 base URL 规范化为 Anthropic Messages endpoint。"""
    if base_url is None or not base_url.strip():
        return DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT
    endpoint = (_redact_url(base_url) or "").rstrip("/")
    if "://" not in endpoint:
        endpoint = f"https://{endpoint}"
    if endpoint.endswith("/v1/messages"):
        return endpoint
    if endpoint.endswith("/v1"):
        return f"{endpoint}/messages"
    return f"{endpoint}/v1/messages"


def _normalize_gemini_generate_content_endpoint(base_url: str | None, model: str) -> str:
    """把 Gemini base URL 规范化为指定 model 的 generateContent endpoint。"""
    model_path = model if model.startswith("models/") else f"models/{model}"
    if base_url is None or not base_url.strip():
        base = DEFAULT_GEMINI_API_BASE_URL
    else:
        base = (_redact_url(base_url) or "").rstrip("/")
        if "://" not in base:
            base = f"https://{base}"
    if base.endswith(":generateContent"):
        return base
    return f"{base}/{model_path}:generateContent"


def _redact_url(url: str | None) -> str | None:
    if url is None or not url.strip():
        return None
    parsed = urlsplit(url.strip())
    if not parsed.scheme or not parsed.hostname:
        return url.strip().split("?", 1)[0]
    netloc = parsed.hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _endpoint_base_url(endpoint: str | None) -> str | None:
    redacted = _redact_url(endpoint)
    if redacted is None:
        return None
    parsed = urlsplit(redacted)
    path = parsed.path.rstrip("/")
    for suffix in (
        "/chat/completions",
        "/responses",
        "/messages",
    ):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    if "/models/" in path:
        path = path.split("/models/", 1)[0]
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _responses_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_RESPONSES_ENDPOINT,
) -> dict[str, object]:
    """执行真实 HTTP 请求；保持为函数便于测试注入替身 transport。"""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(f"OpenAI request failed with HTTP {error.code}: {detail}") from error
    return json.loads(body)


def _chat_completions_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_CHAT_COMPLETIONS_ENDPOINT,
) -> dict[str, object]:
    """执行真实 Chat Completions HTTP 请求；测试中通过 transport 注入替身。"""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"OpenAI chat request failed with HTTP {error.code}: {detail}",
        ) from error
    return json.loads(body)


def _anthropic_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT,
) -> dict[str, object]:
    """执行真实 Anthropic Messages HTTP 请求；测试中通过 transport 注入替身。"""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"Anthropic request failed with HTTP {error.code}: {detail}",
        ) from error
    return json.loads(body)


def _google_gemini_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
) -> dict[str, object]:
    """执行真实 Gemini generateContent HTTP 请求；测试中通过 transport 注入替身。"""
    separator = "&" if "?" in endpoint else "?"
    request = urllib.request.Request(
        f"{endpoint}{separator}key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"Gemini request failed with HTTP {error.code}: {detail}",
        ) from error
    return json.loads(body)


def _responses_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _parse_openai_responses_stream(response, on_delta)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(f"OpenAI request failed with HTTP {error.code}: {detail}") from error


def _chat_completions_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _parse_openai_chat_stream(response, on_delta)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"OpenAI chat request failed with HTTP {error.code}: {detail}",
        ) from error


def _anthropic_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _parse_anthropic_stream(response, on_delta)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"Anthropic request failed with HTTP {error.code}: {detail}",
        ) from error


def _google_gemini_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    stream_endpoint = endpoint.replace(":generateContent", ":streamGenerateContent")
    separator = "&" if "?" in stream_endpoint else "?"
    request = urllib.request.Request(
        f"{stream_endpoint}{separator}alt=sse&key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return _parse_gemini_stream(response, on_delta)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ModelCallError(
            f"Gemini request failed with HTTP {error.code}: {detail}",
        ) from error


def _iter_sse_events(response) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    event_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            if event_lines:
                data_chunks = [part[5:].strip() for part in event_lines if part.startswith("data:")]
                if data_chunks:
                    data = "\n".join(data_chunks)
                    if data != "[DONE]":
                        events.append(json.loads(data))
                event_lines = []
            continue
        event_lines.append(line)
    if event_lines:
        data_chunks = [part[5:].strip() for part in event_lines if part.startswith("data:")]
        if data_chunks:
            data = "\n".join(data_chunks)
            if data != "[DONE]":
                events.append(json.loads(data))
    return events


def _parse_openai_chat_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, object]] = {}
    usage: dict[str, object] | None = None
    for event in _iter_sse_events(response):
        raw_usage = event.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                on_delta(content)
            raw_tool_calls = delta.get("tool_calls")
            if not isinstance(raw_tool_calls, list):
                continue
            for item in raw_tool_calls:
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index", 0))
                aggregated = tool_calls.setdefault(
                    index,
                    {"id": str(item.get("id") or ""), "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if item.get("id"):
                    aggregated["id"] = str(item["id"])
                function = item.get("function")
                if not isinstance(function, dict):
                    continue
                aggregated_function = aggregated["function"]
                if isinstance(function.get("name"), str):
                    aggregated_function["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    aggregated_function["arguments"] += function["arguments"]
    parsed: dict[str, object] = {
        "choices": [
            {
                "message": {
                    "content": "".join(content_parts),
                    "tool_calls": [tool_calls[index] for index in sorted(tool_calls)],
                },
            },
        ],
    }
    if usage is not None:
        parsed["usage"] = usage
    return parsed


def _parse_openai_responses_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    output_text_parts: list[str] = []
    final_response: dict[str, object] | None = None
    for event in _iter_sse_events(response):
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str) and delta:
                output_text_parts.append(delta)
                on_delta(delta)
        elif event_type == "response.completed":
            response_payload = event.get("response")
            if isinstance(response_payload, dict):
                final_response = response_payload
    if final_response is None:
        final_response = {"output_text": "".join(output_text_parts), "output": []}
    elif not isinstance(final_response.get("output_text"), str):
        final_response["output_text"] = "".join(output_text_parts)
    return final_response


def _parse_anthropic_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    text_parts: list[str] = []
    content_blocks: list[dict[str, object]] = []
    current_tool_block: dict[str, object] | None = None
    usage: dict[str, object] | None = None
    for event in _iter_sse_events(response):
        event_type = event.get("type")
        if event_type == "content_block_start":
            block = event.get("content_block")
            if isinstance(block, dict) and block.get("type") == "tool_use":
                current_tool_block = {
                    "type": "tool_use",
                    "id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "input": {},
                }
        elif event_type == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                continue
            if delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    text_parts.append(text)
                    on_delta(text)
            elif delta.get("type") == "input_json_delta" and current_tool_block is not None:
                partial_json = delta.get("partial_json")
                if isinstance(partial_json, str) and partial_json.strip():
                    current_tool_block["_partial_json"] = str(current_tool_block.get("_partial_json", "")) + partial_json
        elif event_type == "content_block_stop" and current_tool_block is not None:
            partial_json = current_tool_block.pop("_partial_json", "")
            if isinstance(partial_json, str) and partial_json.strip():
                current_tool_block["input"] = _parse_tool_arguments(partial_json)
            content_blocks.append(current_tool_block)
            current_tool_block = None
        elif event_type == "message_stop":
            break
        raw_usage = event.get("usage")
        if isinstance(raw_usage, dict):
            usage = raw_usage
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("usage"), dict):
            usage = delta["usage"]
    if text_parts:
        content_blocks.insert(0, {"type": "text", "text": "".join(text_parts)})
    parsed: dict[str, object] = {"content": content_blocks}
    if usage is not None:
        parsed["usage"] = usage
    return parsed


def _parse_gemini_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    text_parts: list[str] = []
    function_calls: list[dict[str, object]] = []
    usage_metadata: dict[str, object] | None = None
    for event in _iter_sse_events(response):
        raw_usage = event.get("usageMetadata")
        if isinstance(raw_usage, dict):
            usage_metadata = raw_usage
        candidates = event.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            candidate_parts = content.get("parts")
            if not isinstance(candidate_parts, list):
                continue
            for part in candidate_parts:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str) and part["text"]:
                    text_parts.append(part["text"])
                    on_delta(part["text"])
                elif isinstance(part.get("functionCall"), dict):
                    function_calls.append(part["functionCall"])
    parts.extend({"text": item} for item in text_parts)
    parts.extend({"functionCall": item} for item in function_calls)
    parsed: dict[str, object] = {"candidates": [{"content": {"parts": parts}}]}
    if usage_metadata is not None:
        parsed["usageMetadata"] = usage_metadata
    return parsed
