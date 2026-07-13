"""
src/haagent/models/http_transport.py - 共享 httpx 模型传输与增量 SSE 解码

session-owned Client 生命周期、JSON/stream 请求、transport telemetry 与 SSE 边界解析。
"""

from __future__ import annotations

import codecs
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelCallError, ModelFailureDetails

TelemetrySink = Callable[[ModelTransportEvent], None]
StreamParser = Callable[[Iterator["SseEvent"], Callable[[str], None]], dict[str, object]]

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
_RESPONSE_BODY_EXCERPT_LENGTH = 4096
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class SseEvent:
    """已完成的一条 SSE 事件；data 保持原始字符串，由 provider parser 决定如何解析。"""

    data: str
    event: str | None = None
    id: str | None = None
    retry: int | None = None


class IncrementalSseDecoder:
    """增量 SSE 解码器：跨 chunk 保留未完成 UTF-8 与未完成 event。"""

    def __init__(self) -> None:
        self._text_decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._pending = ""
        self._event_lines: list[str] = []
        self._done = False

    def feed(self, chunk: bytes) -> list[SseEvent]:
        if self._done or not chunk:
            return []
        try:
            text = self._text_decoder.decode(chunk, final=False)
        except UnicodeDecodeError as error:
            raise ModelCallError(
                "provider response is not valid SSE JSON",
                details=ModelFailureDetails(category="response_parse", retryable=False),
            ) from error
        return self._consume_text(text)

    def finish(self) -> list[SseEvent]:
        if self._done:
            return []
        try:
            text = self._text_decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            raise ModelCallError(
                "provider response is not valid SSE JSON",
                details=ModelFailureDetails(category="response_parse", retryable=False),
            ) from error
        events = self._consume_text(text)
        if self._done:
            return events
        # 流结束时冲刷无尾部分隔符的最后一个 event。
        if self._pending:
            self._event_lines.append(self._pending)
            self._pending = ""
        final = self._event_from_lines(self._event_lines)
        self._event_lines = []
        if final is not None:
            if final.data == "[DONE]":
                self._done = True
            else:
                events.append(final)
        return events

    def _consume_text(self, text: str) -> list[SseEvent]:
        if not text:
            return []
        self._pending += text
        events: list[SseEvent] = []
        while not self._done:
            separator_at, separator_len = self._find_separator(self._pending)
            if separator_at < 0:
                break
            block = self._pending[:separator_at]
            self._pending = self._pending[separator_at + separator_len :]
            lines = block.splitlines()
            # 注释与字段都在完整 block 上解析；空 block 忽略。
            event = self._event_from_lines(lines)
            if event is None:
                continue
            if event.data == "[DONE]":
                self._done = True
                break
            events.append(event)
        return events

    @staticmethod
    def _find_separator(text: str) -> tuple[int, int]:
        """返回最早出现的 SSE 事件分隔符位置与长度。"""

        candidates = [
            (text.find("\r\n\r\n"), 4),
            (text.find("\n\n"), 2),
            (text.find("\r\r"), 2),
        ]
        found = [(index, length) for index, length in candidates if index >= 0]
        if not found:
            return -1, 0
        return min(found, key=lambda item: item[0])

    @staticmethod
    def _event_from_lines(lines: list[str]) -> SseEvent | None:
        if not lines:
            return None
        data_chunks: list[str] = []
        event_name: str | None = None
        event_id: str | None = None
        retry: int | None = None
        for line in lines:
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_chunks.append(value)
                continue
            if line.startswith("event:"):
                value = line[6:]
                event_name = value[1:] if value.startswith(" ") else value
                continue
            if line.startswith("id:"):
                value = line[3:]
                event_id = value[1:] if value.startswith(" ") else value
                continue
            if line.startswith("retry:"):
                value = line[6:]
                raw = value[1:] if value.startswith(" ") else value
                try:
                    retry = int(raw)
                except ValueError:
                    retry = None
                continue
            # 未知字段按 SSE 规范忽略。
        if not data_chunks:
            return None
        return SseEvent(
            data="\n".join(data_chunks),
            event=event_name,
            id=event_id,
            retry=retry,
        )


class ModelHttpTransport:
    """共享 httpx.Client 的同步模型 HTTP transport。"""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._owns_client = client is None
        # trust_env=False：忽略系统 HTTP(S)_PROXY，避免本地/内网 endpoint 被代理劫持返回 502。
        self._client = client or httpx.Client(
            timeout=timeout or DEFAULT_TIMEOUT,
            trust_env=False,
        )
        self._closed = False

    def close(self) -> None:
        """幂等关闭；仅关闭自身创建的 client，不越权关闭注入 client。"""

        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def request_json(
        self,
        provider: str,
        endpoint: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        attempt: int,
        telemetry_sink: TelemetrySink | None,
    ) -> dict[str, object]:
        self._ensure_open()
        started_at = time.perf_counter()
        body = json.dumps(payload).encode("utf-8")
        self._publish(
            telemetry_sink,
            kind="request_prepared",
            attempt=attempt,
            started_at=started_at,
            request_payload_bytes=len(body),
        )
        try:
            response = self._client.post(endpoint, content=body, headers=headers)
        except httpx.TimeoutException as error:
            raise _network_error("timeout", error) from error
        except httpx.HTTPError as error:
            raise _network_error("network", error) from error
        try:
            self._publish(
                telemetry_sink,
                kind="headers_received",
                attempt=attempt,
                started_at=started_at,
            )
            if response.status_code >= 400:
                raise _http_error(provider, response)
            return _json_object(provider, response)
        finally:
            response.close()

    def stream_json(
        self,
        provider: str,
        endpoint: str,
        payload: dict[str, object],
        headers: dict[str, str],
        *,
        parser: StreamParser,
        on_delta: Callable[[str], None],
        attempt: int,
        telemetry_sink: TelemetrySink | None,
    ) -> dict[str, object]:
        self._ensure_open()
        started_at = time.perf_counter()
        body = json.dumps(payload).encode("utf-8")
        self._publish(
            telemetry_sink,
            kind="request_prepared",
            attempt=attempt,
            started_at=started_at,
            request_payload_bytes=len(body),
        )
        try:
            with self._client.stream("POST", endpoint, content=body, headers=headers) as response:
                self._publish(
                    telemetry_sink,
                    kind="headers_received",
                    attempt=attempt,
                    started_at=started_at,
                )
                if response.status_code >= 400:
                    # 错误响应读一小段 body 做分类，消息本身不包含正文。
                    response.read()
                    raise _http_error(provider, response)
                first_sse_published = False

                def event_iter() -> Iterator[SseEvent]:
                    nonlocal first_sse_published
                    decoder = IncrementalSseDecoder()
                    for chunk in response.iter_bytes():
                        for event in decoder.feed(chunk):
                            if not first_sse_published:
                                self._publish(
                                    telemetry_sink,
                                    kind="first_sse",
                                    attempt=attempt,
                                    started_at=started_at,
                                )
                                first_sse_published = True
                            yield event
                    for event in decoder.finish():
                        if not first_sse_published:
                            self._publish(
                                telemetry_sink,
                                kind="first_sse",
                                attempt=attempt,
                                started_at=started_at,
                            )
                            first_sse_published = True
                        yield event

                return parser(event_iter(), on_delta)
        except httpx.TimeoutException as error:
            raise _network_error("timeout", error) from error
        except httpx.HTTPError as error:
            raise _network_error("network", error) from error

    def _ensure_open(self) -> None:
        if self._closed:
            raise ModelCallError(
                "model HTTP transport is closed",
                details=ModelFailureDetails(category="client", retryable=False),
            )

    def _publish(
        self,
        telemetry_sink: TelemetrySink | None,
        *,
        kind: str,
        attempt: int,
        started_at: float,
        request_payload_bytes: int | None = None,
    ) -> None:
        if telemetry_sink is None:
            return
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        telemetry_sink(
            ModelTransportEvent(
                kind=kind,  # type: ignore[arg-type]
                attempt=attempt,
                elapsed_ms=elapsed_ms,
                request_payload_bytes=request_payload_bytes,
            )
        )


def _json_object(provider: str, response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as error:
        raise ModelCallError(
            f"{provider} response is not valid JSON",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        ) from error
    if not isinstance(payload, dict):
        raise ModelCallError(
            f"{provider} response must be a JSON object",
            details=ModelFailureDetails(category="response_parse", retryable=False),
        )
    return payload


def _network_error(category: str, error: Exception) -> ModelCallError:
    return ModelCallError(
        f"model request {category} failure",
        details=ModelFailureDetails(category=category, retryable=True),  # type: ignore[arg-type]
    )


def _http_error(provider: str, response: httpx.Response) -> ModelCallError:
    body = response.text[:_RESPONSE_BODY_EXCERPT_LENGTH]
    details = _http_details(response.status_code, response.headers, body)
    return ModelCallError(
        f"{provider} request failed with HTTP {response.status_code}",
        details=details,
    )


def _http_details(
    status_code: int,
    headers: httpx.Headers,
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
        category=category,  # type: ignore[arg-type]
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


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _request_id(headers: httpx.Headers) -> str | None:
    return headers.get("x-request-id") or headers.get("request-id")


def close_model_gateway(gateway: object) -> None:
    """幂等关闭 gateway 拥有的 HTTP 资源；无 close 的对象直接忽略。"""

    close = getattr(gateway, "close", None)
    if callable(close):
        close()
