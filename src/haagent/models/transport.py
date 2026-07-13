"""
src/haagent/models/transport.py - 模型网关 HTTP/SSE 传输与共享解析

提供 endpoint 规范化、HTTP 请求、SSE 解析和跨厂商共用的小工具。
"""

from __future__ import annotations

import base64
import inspect
import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from haagent.models.types import ModelCallError, ModelFailureDetails, ModelUsage


def _supports_transport_observability(callback: Callable[..., object]) -> bool:
    """调用前判断 transport 是否接受 attempt/telemetry，避免捕获执行期 TypeError 后重复请求。"""

    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {parameter.name for parameter in parameters}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    return {"attempt", "telemetry_sink"}.issubset(names)


def _invoke_transport(
    transport: Callable[..., dict[str, object]],
    stream_transport: Callable[..., dict[str, object]],
    *args: object,
    on_delta: Callable[[str], None] | None,
    attempt: int,
    telemetry_sink: object,
) -> dict[str, object]:
    """兼容旧注入签名，并确保执行期 TypeError 不会触发第二次模型请求。"""

    callback = stream_transport if on_delta is not None else transport
    call_args = (*args, on_delta) if on_delta is not None else args
    if _supports_transport_observability(callback):
        return callback(
            *call_args,
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    return callback(*call_args)

DEFAULT_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
DEFAULT_GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


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
def _bearer_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _json_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def _http_transport_or_default(http_transport):
    """生产默认使用共享 ModelHttpTransport；调用方负责生命周期时注入实例。"""

    from haagent.models.http_transport import ModelHttpTransport

    if http_transport is not None:
        return http_transport, False
    return ModelHttpTransport(), True


def _responses_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_RESPONSES_ENDPOINT,
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    """执行真实 HTTP 请求；保持为函数便于测试注入替身 transport。"""

    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.request_json(
            "OpenAI",
            endpoint,
            payload,
            _bearer_headers(api_key),
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _chat_completions_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_CHAT_COMPLETIONS_ENDPOINT,
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    """执行真实 Chat Completions HTTP 请求；测试中通过 transport 注入替身。"""

    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.request_json(
            "OpenAI chat",
            endpoint,
            payload,
            _bearer_headers(api_key),
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _anthropic_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str = DEFAULT_ANTHROPIC_MESSAGES_ENDPOINT,
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    """执行真实 Anthropic Messages HTTP 请求；测试中通过 transport 注入替身。"""

    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.request_json(
            "Anthropic",
            endpoint,
            payload,
            _anthropic_headers(api_key),
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _google_gemini_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    """执行真实 Gemini generateContent HTTP 请求；测试中通过 transport 注入替身。"""

    separator = "&" if "?" in endpoint else "?"
    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.request_json(
            "Gemini",
            f"{endpoint}{separator}key={api_key}",
            payload,
            _json_headers(),
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _responses_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.stream_json(
            "OpenAI",
            endpoint,
            payload,
            _bearer_headers(api_key),
            parser=_parse_openai_responses_sse,
            on_delta=on_delta,
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _chat_completions_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.stream_json(
            "OpenAI chat",
            endpoint,
            payload,
            _bearer_headers(api_key),
            parser=_parse_openai_chat_sse,
            on_delta=on_delta,
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _anthropic_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.stream_json(
            "Anthropic",
            endpoint,
            payload,
            _anthropic_headers(api_key),
            parser=_parse_anthropic_sse,
            on_delta=on_delta,
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _google_gemini_stream_transport(
    payload: dict[str, object],
    api_key: str,
    endpoint: str,
    on_delta: Callable[[str], None],
    *,
    http_transport=None,
    attempt: int = 1,
    telemetry_sink=None,
) -> dict[str, object]:
    stream_endpoint = endpoint.replace(":generateContent", ":streamGenerateContent")
    separator = "&" if "?" in stream_endpoint else "?"
    transport, owns = _http_transport_or_default(http_transport)
    try:
        return transport.stream_json(
            "Gemini",
            f"{stream_endpoint}{separator}alt=sse&key={api_key}",
            payload,
            _json_headers(),
            parser=_parse_gemini_sse,
            on_delta=on_delta,
            attempt=attempt,
            telemetry_sink=telemetry_sink,
        )
    finally:
        if owns:
            transport.close()


def _iter_sse_events(response) -> Iterator[dict[str, object]]:
    """逐条 yield 完整 SSE JSON object；禁止先收集成 list，否则首字延迟=整次生成时间。

    使用 IncrementalSseDecoder 处理 \\n\\n / \\r\\r / \\r\\n\\r\\n、跨 chunk UTF-8 与 [DONE]。
    """

    from haagent.models.http_transport import IncrementalSseDecoder

    decoder = IncrementalSseDecoder()
    for raw_chunk in response:
        if isinstance(raw_chunk, bytes):
            chunk = raw_chunk
        else:
            chunk = str(raw_chunk).encode("utf-8")
        for event in decoder.feed(chunk):
            yield _sse_event_json(event.data)
    for event in decoder.finish():
        yield _sse_event_json(event.data)


def _sse_event_json(data: str) -> dict[str, object]:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as error:
        raise ModelCallError(
            "provider response is not valid SSE JSON",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        ) from error
    if not isinstance(parsed, dict):
        raise ModelCallError(
            "SSE event data must be a JSON object",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        )
    return parsed


def _sse_event_dicts(events) -> Iterator[dict[str, object]]:
    """把已完成 SseEvent 转成 provider 归并使用的 JSON object 流。"""

    for event in events:
        data = event.data if hasattr(event, "data") else event
        if not isinstance(data, str):
            continue
        yield _sse_event_json(data)


def _parse_openai_chat_sse(events, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_openai_chat_events(_sse_event_dicts(events), on_delta)


def _parse_openai_responses_sse(events, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_openai_responses_events(_sse_event_dicts(events), on_delta)


def _parse_anthropic_sse(events, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_anthropic_events(_sse_event_dicts(events), on_delta)


def _parse_gemini_sse(events, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_gemini_events(_sse_event_dicts(events), on_delta)


def _parse_openai_chat_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_openai_chat_events(_iter_sse_events(response), on_delta)


def _parse_openai_responses_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_openai_responses_events(_iter_sse_events(response), on_delta)


def _parse_anthropic_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_anthropic_events(_iter_sse_events(response), on_delta)


def _parse_gemini_stream(response, on_delta: Callable[[str], None]) -> dict[str, object]:
    return _merge_gemini_events(_iter_sse_events(response), on_delta)


def _merge_openai_chat_events(
    events: Iterator[dict[str, object]],
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, object]] = {}
    usage: dict[str, object] | None = None
    for event in events:
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


def _merge_openai_responses_events(
    events: Iterator[dict[str, object]],
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    output_text_parts: list[str] = []
    final_response: dict[str, object] | None = None
    for event in events:
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


def _merge_anthropic_events(
    events: Iterator[dict[str, object]],
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    text_parts: list[str] = []
    content_blocks: list[dict[str, object]] = []
    current_tool_block: dict[str, object] | None = None
    usage: dict[str, object] | None = None
    for event in events:
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


def _merge_gemini_events(
    events: Iterator[dict[str, object]],
    on_delta: Callable[[str], None],
) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    text_parts: list[str] = []
    function_calls: list[dict[str, object]] = []
    usage_metadata: dict[str, object] | None = None
    for event in events:
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
