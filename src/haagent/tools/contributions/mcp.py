"""
haagent/tools/contributions/mcp.py - 静态 MCP 资源工具 contribution

动态 mcp__* 工具不在此登记，仍由 MCP adapter + ToolRouter 分支处理。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.mcp_tools import list_mcp_resources, read_mcp_resource


def _bind_list_mcp_resources(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return list_mcp_resources(args, deps.mcp_runtime)

    return handler


def _bind_read_mcp_resource(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return read_mcp_resource(args, deps.mcp_runtime)

    return handler


MCP_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="list_mcp_resources",
        description="List resources exposed by connected MCP servers.",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"mcp_resource"}),
        bind_handler=_bind_list_mcp_resources,
    ),
    ToolContribution(
        name="read_mcp_resource",
        description="Read one resource from a connected MCP server by server name and URI.",
        risk_level="medium",
        parameters={
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "uri": {"type": "string"},
            },
            "required": ["server", "uri"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"mcp_resource"}),
        bind_handler=_bind_read_mcp_resource,
    ),
]
