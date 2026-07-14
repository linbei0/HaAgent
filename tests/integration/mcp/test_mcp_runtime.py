"""
tests/integration/mcp/test_mcp_runtime.py - MCP 运行期连接测试

验证同步 HaAgent runtime 可以通过后台事件循环连接和调用异步 MCP server。
"""

import asyncio
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from haagent.mcp.client import McpClientManager, McpToolExecutionError
from haagent.mcp.settings import redact_mcp_secret_text
from haagent.mcp.runtime import McpRuntimeTimeoutError, SyncMcpRuntime
from haagent.mcp.types import McpHttpServerConfig, McpSettings, McpStdioServerConfig
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
import pytest


class ErrorResultSession:
    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> object:
        del tool_name, arguments
        return SimpleNamespace(
            content=[SimpleNamespace(text="quota exceeded")],
            structuredContent=None,
            isError=True,
        )


def test_mcp_client_raises_structured_error_for_mcp_error_result():
    manager = McpClientManager(McpSettings())
    manager._sessions["exa"] = ErrorResultSession()  # type: ignore[assignment]

    with pytest.raises(McpToolExecutionError) as exc_info:
        asyncio.run(manager.call_tool("exa", "web_fetch_exa", {"url": "https://example.com"}))

    assert str(exc_info.value) == "quota exceeded"


def test_sync_mcp_runtime_discovers_and_calls_stdio_tool():
    server_path = Path(__file__).parents[2] / "fixtures" / "fake_mcp_server.py"
    runtime = SyncMcpRuntime(
        McpSettings(
            servers={
                "fixture": McpStdioServerConfig(
                    name="fixture",
                    command=sys.executable,
                    args=[str(server_path)],
                ),
            },
        ),
    )

    try:
        runtime.start()

        tools = runtime.list_tools()
        assert [tool.name for tool in tools] == ["echo"]
        assert tools[0].server_name == "fixture"
        assert runtime.call_tool("fixture", "echo", {"text": "hi"}) == "echo:hi"
        assert runtime.read_resource("fixture", "fixture://hello") == "hello from fixture"
    finally:
        runtime.close()


def test_sync_mcp_runtime_records_failed_server_without_raising():
    runtime = SyncMcpRuntime(
        McpSettings(
            servers={
                "missing": McpStdioServerConfig(
                    name="missing",
                    command=sys.executable,
                    args=["does-not-exist.py"],
                ),
            },
        ),
    )

    try:
        runtime.start()

        statuses = runtime.list_statuses()
        assert statuses[0].name == "missing"
        assert statuses[0].state == "failed"
        assert statuses[0].detail
    finally:
        runtime.close()


def test_mcp_client_closes_partially_connected_http_transport_in_same_task(monkeypatch):
    tasks: dict[str, object] = {}

    class TrackingStreamContext:
        async def __aenter__(self):
            tasks["enter"] = asyncio.current_task()
            return object(), object(), None

        async def __aexit__(self, exc_type, exc, traceback):
            tasks["exit"] = asyncio.current_task()

    monkeypatch.setattr(
        "haagent.mcp.client.streamable_http_client",
        lambda url, http_client: TrackingStreamContext(),
    )
    manager = McpClientManager(
        McpSettings(
            servers={
                "remote": McpHttpServerConfig(
                    name="remote",
                    url="http://127.0.0.1:9/mcp",
                )
            }
        )
    )

    async def fail_register(name, stack, read_stream, write_stream):
        raise RuntimeError("initialize failed")

    monkeypatch.setattr(manager, "_register_session", fail_register)

    asyncio.run(manager.connect_all())

    assert manager.list_statuses()[0].state == "failed"
    assert tasks["exit"] is tasks["enter"]


def test_sync_mcp_runtime_cancels_waiting_call_promptly():
    server_path = Path(__file__).parents[2] / "fixtures" / "fake_mcp_server.py"
    runtime = SyncMcpRuntime(
        McpSettings(
            servers={
                "fixture": McpStdioServerConfig(
                    name="fixture",
                    command=sys.executable,
                    args=[str(server_path)],
                ),
            },
        ),
    )
    token = CancellationToken()

    try:
        runtime.start()
        threading.Timer(0.05, token.cancel).start()
        started = time.perf_counter()

        with pytest.raises(RunCancelled):
            runtime._run(asyncio.sleep(5), cancellation_token=token)

        assert time.perf_counter() - started < 1
    finally:
        runtime.close()


def test_sync_mcp_runtime_times_out_waiting_call_promptly():
    server_path = Path(__file__).parents[2] / "fixtures" / "fake_mcp_server.py"
    runtime = SyncMcpRuntime(
        McpSettings(
            servers={
                "fixture": McpStdioServerConfig(
                    name="fixture",
                    command=sys.executable,
                    args=[str(server_path)],
                ),
            },
        ),
    )

    try:
        runtime.start()
        started = time.perf_counter()

        with pytest.raises(McpRuntimeTimeoutError) as exc_info:
            runtime._run(asyncio.sleep(5), timeout_seconds=0.05)

        assert "timed out after 0.05 seconds" in str(exc_info.value)
        assert time.perf_counter() - started < 1
    finally:
        runtime.close()


def test_empty_mcp_runtime_does_not_start_background_loop(monkeypatch):
    def fail_new_event_loop():
        raise AssertionError("empty MCP settings should not start an event loop")

    monkeypatch.setattr("haagent.mcp.runtime.asyncio.new_event_loop", fail_new_event_loop)

    runtime = SyncMcpRuntime(McpSettings())
    runtime.start()

    assert runtime.list_statuses() == []
    assert runtime.list_tools() == []


def test_mcp_status_redacts_configured_header_and_env_values():
    settings = McpSettings(
        servers={
            "local": McpStdioServerConfig(
                name="local",
                command="bad-secret-token",
                env={"TOKEN": "secret-token"},
            ),
            "remote": McpHttpServerConfig(
                name="remote",
                url="http://example.invalid",
                headers={"Authorization": "Bearer secret-header"},
            ),
        },
    )

    assert "secret-token" not in redact_mcp_secret_text("failed: secret-token", settings)
    assert "secret-header" not in redact_mcp_secret_text("failed: Bearer secret-header", settings)
