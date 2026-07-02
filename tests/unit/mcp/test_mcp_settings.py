"""
tests/unit/mcp/test_mcp_settings.py - MCP 配置解析测试

验证用户级 MCP 配置的缺省、stdio/http 解析和风险等级校验。
"""

import json

import pytest

from haagent.mcp.settings import McpSettingsError, load_mcp_settings
from haagent.mcp.types import McpHttpServerConfig, McpStdioServerConfig


def test_missing_mcp_settings_returns_empty(tmp_path):
    settings = load_mcp_settings(tmp_path / "missing.json")

    assert settings.servers == {}
    assert settings.tool_risks == {}


def test_loads_stdio_and_http_servers(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "servers": {
                    "local": {
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["demo"],
                        "env": {"TOKEN": "secret"},
                    },
                    "remote": {
                        "type": "http",
                        "url": "http://127.0.0.1:8765/mcp",
                        "headers": {"Authorization": "Bearer secret"},
                    },
                },
                "tool_risks": {"local.echo": "medium"},
            },
        ),
        encoding="utf-8",
    )

    settings = load_mcp_settings(path)

    assert isinstance(settings.servers["local"], McpStdioServerConfig)
    assert settings.servers["local"].command == "uvx"
    assert settings.servers["local"].args == ["demo"]
    assert settings.servers["local"].env == {"TOKEN": "secret"}
    assert isinstance(settings.servers["remote"], McpHttpServerConfig)
    assert settings.servers["remote"].url == "http://127.0.0.1:8765/mcp"
    assert settings.tool_risks == {"local.echo": "medium"}


def test_rejects_unknown_risk_level(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps({"servers": {}, "tool_risks": {"local.echo": "trusted"}}),
        encoding="utf-8",
    )

    with pytest.raises(McpSettingsError, match="invalid MCP tool risk"):
        load_mcp_settings(path)
