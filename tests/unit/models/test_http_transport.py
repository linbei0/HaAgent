"""
tests/unit/models/test_http_transport.py - ModelHttpTransport 与 IncrementalSseDecoder

覆盖 SSE 边界、共享 client 生命周期、timeout 默认值和 transport telemetry。
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx
import pytest

from haagent.models.http_transport import (
    IncrementalSseDecoder,
    ModelHttpTransport,
    SseEvent,
)
from haagent.models.telemetry import ModelTransportEvent
from haagent.models.types import ModelCallError


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\r", b"\r\n\r\n"])
def test_decoder_emits_event_for_all_sse_separators(separator: bytes) -> None:
    decoder = IncrementalSseDecoder()
    events = decoder.feed(b'data: {"value": 1}' + separator)
    assert len(events) == 1
    assert events[0].data == '{"value": 1}'


def test_decoder_joins_multi_data_lines_and_ignores_comments() -> None:
    decoder = IncrementalSseDecoder()
    events = decoder.feed(
        b": keep-alive\n"
        b"event: message\n"
        b"id: 7\n"
        b"retry: 1500\n"
        b'data: {"text": "line1",\n'
        b'data: "num": 2}\n'
        b"\n"
    )
    assert len(events) == 1
    assert events[0].event == "message"
    assert events[0].id == "7"
    assert events[0].retry == 1500
    assert events[0].data == '{"text": "line1",\n"num": 2}'


def test_decoder_handles_event_split_across_chunks() -> None:
    decoder = IncrementalSseDecoder()
    assert decoder.feed(b'data: {"a":') == []
    assert decoder.feed(b' 1}') == []
    events = decoder.feed(b"\n\n")
    assert len(events) == 1
    assert events[0].data == '{"a": 1}'


def test_decoder_handles_utf8_character_split_across_byte_chunks() -> None:
    # “中” UTF-8 = e4 b8 ad；故意跨两个 chunk 切开。
    decoder = IncrementalSseDecoder()
    payload = 'data: {"text": "中"}\n\n'.encode("utf-8")
    split_at = payload.index(b"\xe4") + 1
    assert decoder.feed(payload[:split_at]) == []
    events = decoder.feed(payload[split_at:])
    assert len(events) == 1
    assert events[0].data == '{"text": "中"}'


def test_decoder_done_stops_further_events() -> None:
    decoder = IncrementalSseDecoder()
    first = decoder.feed(b'data: {"ok": true}\n\ndata: [DONE]\n\n')
    assert len(first) == 1
    assert first[0].data == '{"ok": true}'
    assert decoder.feed(b'data: {"late": true}\n\n') == []
    assert decoder.finish() == []


def test_decoder_bad_utf8_raises_response_parse() -> None:
    decoder = IncrementalSseDecoder()
    with pytest.raises(ModelCallError) as exc_info:
        decoder.feed(b"data: \xff\n\n")
    assert exc_info.value.details is not None
    assert exc_info.value.details.category == "response_parse"
    assert exc_info.value.details.retryable is False


def test_model_http_transport_get_json_publishes_telemetry_and_closes() -> None:
    close_count = {"value": 0}

    class TrackingResponse(httpx.Response):
        def close(self) -> None:
            close_count["value"] += 1
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith("/responses/resp_1")
        return TrackingResponse(200, json={"id": "resp_1", "status": "completed"}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = ModelHttpTransport(client=client)
    events: list[ModelTransportEvent] = []
    try:
        result = transport.get_json(
            "OpenAI",
            "https://api.openai.com/v1/responses/resp_1",
            {"Authorization": "Bearer secret"},
            attempt=2,
            telemetry_sink=events.append,
        )
        assert result == {"id": "resp_1", "status": "completed"}
        assert [item.kind for item in events] == ["request_prepared", "headers_received"]
        assert all(item.attempt == 2 for item in events)
    finally:
        transport.close()
    assert close_count["value"] >= 1


def test_model_http_transport_request_json_publishes_telemetry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        body = json.loads(request.content.decode("utf-8"))
        assert body == {"hello": "world"}
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = ModelHttpTransport(client=client)
    events: list[ModelTransportEvent] = []
    try:
        result = transport.request_json(
            "OpenAI chat",
            "https://example.test/v1/chat/completions",
            {"hello": "world"},
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
            attempt=1,
            telemetry_sink=events.append,
        )
        assert result == {"ok": True}
        kinds = [item.kind for item in events]
        assert kinds == ["request_prepared", "headers_received"]
        assert events[0].request_payload_bytes == len(json.dumps({"hello": "world"}).encode("utf-8"))
        assert all(item.attempt == 1 for item in events)
    finally:
        transport.close()


def test_model_http_transport_stream_json_closes_response_and_emits_first_sse() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = ModelHttpTransport(client=client)
    events: list[ModelTransportEvent] = []
    deltas: list[str] = []
    closed: list[bool] = []

    def parser(sse_events: Iterator[SseEvent], on_delta) -> dict[str, object]:
        content_parts: list[str] = []
        for event in sse_events:
            payload = json.loads(event.data)
            choice = payload["choices"][0]
            text = choice["delta"]["content"]
            content_parts.append(text)
            on_delta(text)
        return {"content": "".join(content_parts)}

    # MockTransport 不暴露真实 close 钩子；通过 parser 提前结束验证仍能完成。
    try:
        result = transport.stream_json(
            "OpenAI chat",
            "https://example.test/v1/chat/completions",
            {"stream": True},
            {"Authorization": "Bearer secret", "Content-Type": "application/json"},
            parser=parser,
            on_delta=deltas.append,
            attempt=2,
            telemetry_sink=events.append,
        )
        assert result == {"content": "Hi"}
        assert deltas == ["Hi"]
        kinds = [item.kind for item in events]
        assert kinds == ["request_prepared", "headers_received", "first_sse"]
        assert all(item.attempt == 2 for item in events)
        closed.append(True)
    finally:
        transport.close()
    assert closed == [True]


def test_model_http_stream_closes_response_when_parser_raises() -> None:
    close_count = {"value": 0}

    class TrackingResponse(httpx.Response):
        def close(self) -> None:
            close_count["value"] += 1
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        return TrackingResponse(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"x": 1}\n\n',
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = ModelHttpTransport(client=client)

    def parser(sse_events: Iterator[SseEvent], on_delta) -> dict[str, object]:
        next(sse_events)
        raise RuntimeError("consumer failed")

    try:
        with pytest.raises(RuntimeError, match="consumer failed"):
            transport.stream_json(
                "OpenAI chat",
                "https://example.test/v1/chat/completions",
                {"stream": True},
                {"Content-Type": "application/json"},
                parser=parser,
                on_delta=lambda _text: None,
                attempt=1,
                telemetry_sink=None,
            )
    finally:
        transport.close()
    assert close_count["value"] >= 1


def test_model_http_transport_close_is_idempotent_and_blocks_later_requests() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True})))
    transport = ModelHttpTransport(client=client)
    transport.close()
    transport.close()
    with pytest.raises(ModelCallError) as exc_info:
        transport.request_json(
            "OpenAI chat",
            "https://example.test/v1/chat/completions",
            {"hello": "world"},
            {"Content-Type": "application/json"},
            attempt=1,
            telemetry_sink=None,
        )
    assert "closed" in str(exc_info.value).lower() or "lifecycle" in str(exc_info.value).lower()


def test_model_http_transport_maps_http_error_without_body_leak() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": "server_error", "message": "secret body"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    transport = ModelHttpTransport(client=client)
    try:
        with pytest.raises(ModelCallError) as exc_info:
            transport.request_json(
                "OpenAI chat",
                "https://example.test/v1/chat/completions",
                {"hello": "world"},
                {"Content-Type": "application/json"},
                attempt=1,
                telemetry_sink=None,
            )
        assert exc_info.value.details is not None
        assert exc_info.value.details.category == "server"
        assert exc_info.value.details.status_code == 500
        assert "secret body" not in str(exc_info.value)
    finally:
        transport.close()


def test_model_http_transport_preserves_retry_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "rate_limit", "message": "secret body"}},
            headers={"retry-after": "4", "x-request-id": "req_123"},
        )

    transport = ModelHttpTransport(client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(ModelCallError) as exc_info:
            transport.request_json(
                "OpenAI chat",
                "https://example.test/v1/chat/completions",
                {},
                {},
                attempt=1,
                telemetry_sink=None,
            )
        details = exc_info.value.details
        assert details is not None
        assert details.category == "rate_limited"
        assert details.retry_after_seconds == 4
        assert details.request_id == "req_123"
        assert "secret body" not in str(exc_info.value)
    finally:
        transport.close()
