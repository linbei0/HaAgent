"""
haagent/tools/registry_fragments/mcp.py - MCP 资源工具注册表

定义连接 MCP 服务后的资源列举和读取工具。
"""

from haagent.tools.registry import ToolDefinition


MCP_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "list_mcp_resources": ToolDefinition(
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
    ),
    "read_mcp_resource": ToolDefinition(
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
    ),
}
