"""
src/haagent/tools/mcp_tools.py - MCP 工具适配器

把 MCP resources 和动态 MCP tools 适配成 ToolRouter 可调用的同步 handler。
"""

from __future__ import annotations

import re
from typing import Any

from haagent.mcp.client import McpServerNotConnectedError, McpToolExecutionError
from haagent.mcp.runtime import McpRuntimeTimeoutError, SyncMcpRuntime
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.tools.base import tool_error


_MCP_TOOL_RE = re.compile(r"^mcp__(?P<server>[^_][A-Za-z0-9_]*?)__(?P<tool>.+)$")


def list_mcp_resources(args: dict[str, Any], runtime: SyncMcpRuntime | None) -> dict[str, Any]:
    del args
    if runtime is None:
        return tool_error("mcp_unavailable", "MCP runtime is not available")
    resources = [
        {
            "server": resource.server_name,
            "uri": resource.uri,
            "name": resource.name,
            "description": resource.description,
            "mime_type": resource.mime_type,
        }
        for resource in runtime.list_resources()
    ]
    return {"status": "success", "resources": resources}


def read_mcp_resource(args: dict[str, Any], runtime: SyncMcpRuntime | None) -> dict[str, Any]:
    if runtime is None:
        return tool_error("mcp_unavailable", "MCP runtime is not available")
    return {"status": "success", "output": runtime.read_resource(args["server"], args["uri"])}


def run_mcp_tool(
    tool_name: str,
    args: dict[str, Any],
    runtime: SyncMcpRuntime | None,
    *,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    if runtime is None:
        return tool_error("mcp_unavailable", "MCP runtime is not available")
    match = _MCP_TOOL_RE.match(tool_name)
    if match is None:
        return tool_error("invalid_mcp_tool_name", f"invalid MCP tool name: {tool_name}")
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    server_name = match.group("server")
    mcp_tool_name = match.group("tool")
    try:
        output = runtime.call_tool(
            server_name,
            mcp_tool_name,
            args,
            cancellation_token=cancellation_token,
        )
    except McpRuntimeTimeoutError as error:
        result = tool_error("mcp_timeout", str(error))
        result["execution_state"] = "unknown"
        return result
    except McpToolExecutionError as error:
        return tool_error("mcp_tool_error", str(error))
    except McpServerNotConnectedError as error:
        return tool_error("mcp_unavailable", str(error))
    return {"status": "success", "output": output}
