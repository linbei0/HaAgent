"""
src/haagent/models/transport.py - 模型网关 HTTP/SSE 传输与共享解析

提供 endpoint 规范化、HTTP 请求、SSE 解析和跨厂商共用的小工具。
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import http.client
import json
from pathlib import Path
import socket
from collections.abc import Iterator
from typing import Any, Callable, Mapping, TypeVar
from urllib.parse import urlsplit, urlunsplit
import urllib.error
import urllib.request

from haagent.models.types import ModelCallError, ModelFailureDetails, ModelUsage

DEFAULT_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_RESPONSE_BODY_EXCERPT_LENGTH = 4096

ResponseT = TypeVar("ResponseT")


def _model_error_from_http_error(provider: str, error: urllib.error.HTTPError) -> ModelCallError:
    """将 HTTP 失败转为脱敏的模型错误，不暴露响应正文和认证头。"""

    body = error.read(_RESPONSE_BODY_EXCERPT_LENGTH).decode("utf-8", errors="replace")
    details = _http_details(error.code, error.headers or {}, body)
    return ModelCallError(
        f"{provider} request failed with HTTP {error.code}",
        details=details,
    )


def _model_error_from_network_error(error: Exception) -> ModelCallError:
    """保留网络失败类别，避免把连接和超时错误压扁成字符串。"""

    category = "timeout" if isinstance(error, (socket.timeout, TimeoutError)) else "network"
    return ModelCallError(
        f"model request {category} failure",
        details=ModelFailureDetails(category=category, retryable=True),
    )


def _http_details(
    status_code: int,
    headers: Mapping[str, str],
    body: str,
) -> ModelFailureDetails:
    provider_code = _safe_provider_code(body)
    if status_code in {401, 403}:
        category = "auth"
    elif status_code == 429:
        category = "rate_limited"
    elif status_code >= 500:
        category = "server"
    elif status_code == 408:
        category = "timeout"
    else:
        category = "client"
    if status_code == 429 and provider_code == "insufficient_quota":
        category = "quota_exhausted"
    retryable = status_code in RETRYABLE_STATUS_CODES and category not in {
        "auth",
        "quota_exhausted",
    }
    return ModelFailureDetails(
        category=category,
        status_code=status_code,
        provider_code=provider_code,
        retry_after_seconds=_retry_after_seconds(headers),
        request_id=_request_id(headers),
        retryable=retryable,
    )


def _safe_provider_code(body: str) -> str | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return parsed.get("code") if isinstance(parsed.get("code"), str) else None


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    value = _header_value(headers, "retry-after")
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError, OverflowError):
                retry_at = None
            if retry_at is not None:
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
            else:
                seconds = None
        if seconds is not None:
            return seconds if seconds > 0 else None
    rate_limit_reset = _header_value(headers, "x-ratelimit-reset-requests")
    if rate_limit_reset and rate_limit_reset.endswith("s"):
        try:
            seconds = float(rate_limit_reset[:-1])
        except ValueError:
            return None
        return seconds if seconds > 0 else None
    return None


def _request_id(headers: Mapping[str, str]) -> str | None:
    return _header_value(headers, "x-request-id") or _header_value(headers, "request-id")


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    for header_name, value in headers.items():
        if header_name.lower() == name and isinstance(value, str):
            return value
    return None


def _with_urlopen(request: urllib.request.Request, provider: str, parse: Callable[[object], ResponseT]) -> ResponseT:
    """所有 provider transport 共用失败分类边界，禁止在此处重试。"""

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return parse(response)
    except urllib.error.HTTPError as error:
        raise _model_error_from_http_error(provider, error) from error
    except (
        urllib.error.URLError,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ) as error:
        raise _model_error_from_network_error(error) from error
    except json.JSONDecodeError as error:
        raise ModelCallError(
            f"{provider} response is not valid JSON",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        ) from error


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ModelCallError("invalid tool arguments JSON") from error
    if not isinstance(parsed, dict):
        raise ModelCallError("tool arguments must be a JSON object")
    return parsed


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
    return _with_urlopen(
        request,
        "OpenAI",
        lambda response: json.loads(response.read().decode("utf-8")),
    )


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
    return _with_urlopen(
        request,
        "OpenAI chat",
        lambda response: json.loads(response.read().decode("utf-8")),
    )


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
    return _with_urlopen(
        request,
        "Anthropic",
        lambda response: json.loads(response.read().decode("utf-8")),
    )


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
    return _with_urlopen(
        request,
        "Gemini",
        lambda response: json.loads(response.read().decode("utf-8")),
    )


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
    return _with_urlopen(
        request,
        "OpenAI",
        lambda response: _parse_openai_responses_stream(response, on_delta),
    )


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
    return _with_urlopen(
        request,
        "OpenAI chat",
        lambda response: _parse_openai_chat_stream(response, on_delta),
    )


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
    return _with_urlopen(
        request,
        "Anthropic",
        lambda response: _parse_anthropic_stream(response, on_delta),
    )


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
    return _with_urlopen(
        request,
        "Gemini",
        lambda response: _parse_gemini_stream(response, on_delta),
    )


def _iter_sse_events(response) -> Iterator[dict[str, object]]:
    """逐条 yield 完整 SSE event；禁止先收集成 list，否则首字延迟=整次生成时间。

    HTTP 常把 `data: ...\\n\\n` 作为一整块交付；必须按行拆分并保留空行事件边界，
    不能对整块 `.strip()`，否则会吞掉分隔空行并把多个 event 拼进同一次 json.loads。
    """

    event_lines: list[str] = []
    pending = ""
    for raw_chunk in response:
        if isinstance(raw_chunk, bytes):
            pending += raw_chunk.decode("utf-8")
        else:
            pending += str(raw_chunk)
        while True:
            newline_at = pending.find("\n")
            if newline_at < 0:
                break
            line = pending[:newline_at]
            pending = pending[newline_at + 1 :]
            if line.endswith("\r"):
                line = line[:-1]
            if not line:
                event = _sse_event_from_lines(event_lines)
                event_lines = []
                if event is not None:
                    yield event
                continue
            event_lines.append(line)
    if pending:
        line = pending[:-1] if pending.endswith("\r") else pending
        if line:
            event_lines.append(line)
    event = _sse_event_from_lines(event_lines)
    if event is not None:
        yield event


def _sse_event_from_lines(event_lines: list[str]) -> dict[str, object] | None:
    if not event_lines:
        return None
    # SSE: 字段名后最多去掉一个前导空格；多 data 行用 \n 拼接。
    data_chunks: list[str] = []
    for part in event_lines:
        if not part.startswith("data:"):
            continue
        value = part[5:]
        if value.startswith(" "):
            value = value[1:]
        data_chunks.append(value)
    if not data_chunks:
        return None
    data = "\n".join(data_chunks)
    if data == "[DONE]":
        return None
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ModelCallError(
            "SSE event data must be a JSON object",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        )
    return parsed


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
