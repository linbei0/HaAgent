"""
tests/integration/mcp/test_mcp_runtime.py - MCP 运行期连接测试

验证同步 HaAgent runtime 可以通过后台事件循环连接和调用异步 MCP server。
"""

from pathlib import Path
import sys

from haagent.mcp.settings import redact_mcp_secret_text
from haagent.mcp.runtime import SyncMcpRuntime
from haagent.mcp.types import McpHttpServerConfig, McpSettings, McpStdioServerConfig


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
