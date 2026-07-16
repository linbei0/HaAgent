"""
tests/integration/models/test_http_connection_reuse.py - session 级 HTTP 连接复用

使用本地 HTTP/1.1 server 验证同一 route 连续请求复用连接，以及 close 后显式失败。
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any

import pytest

from haagent.models.gateway_registry import gateway_from_route
from haagent.models.http_transport import ModelHttpTransport, close_model_gateway
from haagent.models.model_connections import ProviderProfile
from haagent.models.model_options import empty_resolved_config
from haagent.models.types import ModelCallError


class _ConnectionCountingServer(ThreadingHTTPServer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.accepted_connection_count = 0
        self._lock = threading.Lock()

    def get_request(self):  # type: ignore[override]
        request, client_address = super().get_request()
        with self._lock:
            self.accepted_connection_count += 1
        return request, client_address


class _ChatHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "ok",
                            "tool_calls": [],
                        }
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture()
def local_chat_server():
    server = _ConnectionCountingServer(("127.0.0.1", 0), _ChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/v1", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _local_chat_profile(base_url: str) -> ProviderProfile:
    return ProviderProfile(
        name="local:chat",
        provider="openai-chat",
        model="local-model",
        base_url=base_url,
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
        credential_source_used="direct",
        api_key="test-key",
        request_config=empty_resolved_config(
            connection_id="local",
            model_id="local-model",
        ),
        runtime_kind="remote",
    )


def test_shared_route_reuses_http11_connection(local_chat_server) -> None:
    base_url, server = local_chat_server
    profile = _local_chat_profile(base_url)
    gateway = gateway_from_route(profile)
    try:
        first = gateway.generate([{"role": "user", "content": "one"}], [])
        second = gateway.generate([{"role": "user", "content": "two"}], [])
        assert first.content == "ok"
        assert second.content == "ok"
        assert server.accepted_connection_count == 1
    finally:
        close_model_gateway(gateway)


def test_closed_gateway_rejects_later_requests(local_chat_server) -> None:
    base_url, server = local_chat_server
    profile = _local_chat_profile(base_url)
    gateway = gateway_from_route(profile)
    close_model_gateway(gateway)
    close_model_gateway(gateway)
    with pytest.raises(ModelCallError):
        gateway.generate([{"role": "user", "content": "after close"}], [])
    assert server.accepted_connection_count == 0


def test_model_http_transport_close_blocks_request_json() -> None:
    transport = ModelHttpTransport()
    transport.close()
    with pytest.raises(ModelCallError):
        transport.request_json(
            "OpenAI chat",
            "http://127.0.0.1:9/v1/chat/completions",
            {"hello": "world"},
            {"Content-Type": "application/json"},
            attempt=1,
            telemetry_sink=None,
        )
