"""
src/haagent/mcp/settings.py - 用户级 MCP 配置加载

读取并校验 ~/.haagent/mcp.json，返回不含运行期连接状态的配置对象。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from haagent.mcp.types import McpHttpServerConfig, McpRiskLevel, McpSettings, McpStdioServerConfig
from haagent.models.model_connections import user_config_dir


class McpSettingsError(Exception):
    """MCP 配置损坏或不可解析时抛出。"""


def user_mcp_settings_path() -> Path:
    return user_config_dir() / "mcp.json"


def load_mcp_settings(config_path: Path | None = None) -> McpSettings:
    path = config_path or user_mcp_settings_path()
    if not path.exists():
        return McpSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise McpSettingsError(f"invalid MCP settings JSON: {error}") from error
    if not isinstance(raw, dict):
        raise McpSettingsError("MCP settings must be a JSON object")
    return McpSettings(
        servers=_parse_servers(raw.get("servers", {})),
        tool_risks=_parse_tool_risks(raw.get("tool_risks", {})),
    )


def redact_mcp_secret_text(text: str, settings: McpSettings) -> str:
    redacted = text
    secrets: set[str] = set()
    for config in settings.servers.values():
        if isinstance(config, McpStdioServerConfig):
            secrets.update(value for value in config.env.values() if value)
        elif isinstance(config, McpHttpServerConfig):
            secrets.update(value for value in config.headers.values() if value)
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 4:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _parse_servers(value: object) -> dict[str, McpStdioServerConfig | McpHttpServerConfig]:
    if not isinstance(value, dict):
        raise McpSettingsError("MCP servers must be an object")
    servers: dict[str, McpStdioServerConfig | McpHttpServerConfig] = {}
    for name, raw_config in value.items():
        if not isinstance(name, str) or not name.strip():
            raise McpSettingsError("MCP server name must be a non-empty string")
        if not isinstance(raw_config, dict):
            raise McpSettingsError(f"MCP server {name} must be an object")
        config_type = raw_config.get("type")
        if config_type == "stdio":
            servers[name] = McpStdioServerConfig(
                name=name,
                command=_required_string(raw_config, "command", f"MCP server {name}"),
                args=_string_list(raw_config.get("args", []), f"MCP server {name} args"),
                env=_string_map(raw_config.get("env", {}), f"MCP server {name} env"),
                cwd=_optional_string(raw_config.get("cwd"), f"MCP server {name} cwd"),
            )
        elif config_type == "http":
            servers[name] = McpHttpServerConfig(
                name=name,
                url=_required_string(raw_config, "url", f"MCP server {name}"),
                headers=_string_map(raw_config.get("headers", {}), f"MCP server {name} headers"),
            )
        else:
            raise McpSettingsError(f"unsupported MCP server type for {name}: {config_type!r}")
    return servers


def _parse_tool_risks(value: object) -> dict[str, McpRiskLevel]:
    if not isinstance(value, dict):
        raise McpSettingsError("MCP tool_risks must be an object")
    risks: dict[str, McpRiskLevel] = {}
    for key, risk in value.items():
        if not isinstance(key, str) or "." not in key:
            raise McpSettingsError("MCP tool risk key must use <server>.<tool>")
        if risk not in {"low", "medium", "high"}:
            raise McpSettingsError(f"invalid MCP tool risk for {key}: {risk!r}")
        risks[key] = cast(McpRiskLevel, risk)
    return risks


def _required_string(raw: dict[str, Any], field: str, owner: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise McpSettingsError(f"{owner} requires string field {field}")
    return value


def _optional_string(value: object, owner: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise McpSettingsError(f"{owner} must be a string when provided")
    return value


def _string_list(value: object, owner: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise McpSettingsError(f"{owner} must be a list of strings")
    return list(value)


def _string_map(value: object, owner: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise McpSettingsError(f"{owner} must be an object")
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        raise McpSettingsError(f"{owner} must contain only string values")
    return dict(value)
