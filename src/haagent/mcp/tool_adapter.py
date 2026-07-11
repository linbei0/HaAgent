"""
src/haagent/mcp/tool_adapter.py - MCP 工具注册表适配

把 MCP server 暴露的工具描述转换成 HaAgent runtime 可用的动态 ToolDefinition。
"""

from __future__ import annotations

import re

from haagent.mcp.types import McpToolInfo
from haagent.tools.registry import ToolDefinition


_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_mcp_tool_segment(value: str) -> str:
    normalized = _SAFE_SEGMENT_RE.sub("_", value.strip())
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("MCP tool segment must contain at least one safe character")
    return normalized


def mcp_tool_alias(server_name: str, tool_name: str) -> str:
    return f"mcp__{sanitize_mcp_tool_segment(server_name)}__{tool_name}"


def mcp_tool_definitions(tools: list[McpToolInfo]) -> dict[str, ToolDefinition]:
    return {
        mcp_tool_alias(tool.server_name, tool.name): ToolDefinition(
            name=mcp_tool_alias(tool.server_name, tool.name),
            description=tool.description or f"MCP tool {tool.name} from server {tool.server_name}",
            risk_level=tool.risk_level,
            parameters=dict(tool.input_schema),
            execution_effect="external_effect",
        )
        for tool in tools
    }
