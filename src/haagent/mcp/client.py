"""
src/haagent/mcp/client.py - 异步 MCP client manager

连接 stdio 和 Streamable HTTP MCP server，并暴露工具、资源和调用接口。
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ReadResourceResult

from haagent.mcp.settings import redact_mcp_secret_text
from haagent.mcp.types import (
    McpConnectionStatus,
    McpHttpServerConfig,
    McpResourceInfo,
    McpSettings,
    McpStdioServerConfig,
    McpToolInfo,
)


class McpServerNotConnectedError(Exception):
    """MCP server 未连接或连接已丢失时抛出。"""


class McpToolExecutionError(Exception):
    """MCP server 返回工具执行错误或调用失败时抛出。"""


class McpClientManager:
    def __init__(self, settings: McpSettings) -> None:
        self._settings = settings
        self._sessions: dict[str, ClientSession] = {}
        self._stacks: list[AsyncExitStack] = []
        self._tools: list[McpToolInfo] = []
        self._resources: list[McpResourceInfo] = []
        self._statuses: dict[str, McpConnectionStatus] = {
            name: McpConnectionStatus(name=name, state="configured")
            for name in settings.servers
        }

    async def connect_all(self) -> None:
        for name, config in self._settings.servers.items():
            try:
                if isinstance(config, McpStdioServerConfig):
                    await self._connect_stdio(name, config)
                elif isinstance(config, McpHttpServerConfig):
                    await self._connect_http(name, config)
            except asyncio.CancelledError as error:
                self._statuses[name] = McpConnectionStatus(
                    name=name,
                    state="failed",
                    detail=redact_mcp_secret_text(
                        f"MCP server cancelled during startup: {error or type(error).__name__}",
                        self._settings,
                    ),
                )
            except Exception as error:
                self._statuses[name] = McpConnectionStatus(
                    name=name,
                    state="failed",
                    detail=redact_mcp_secret_text(f"MCP server failed: {error}", self._settings),
                )

    async def close(self) -> None:
        while self._stacks:
            stack = self._stacks.pop()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await stack.aclose()
        self._sessions.clear()

    def list_statuses(self) -> list[McpConnectionStatus]:
        return list(self._statuses.values())

    def list_tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    def list_resources(self) -> list[McpResourceInfo]:
        return list(self._resources)

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._session(server_name)
        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception as error:
            message = redact_mcp_secret_text(str(error) or type(error).__name__, self._settings)
            raise McpToolExecutionError(message) from error
        output = _stringify_tool_result(result)
        if bool(getattr(result, "isError", False) or getattr(result, "is_error", False)):
            message = output or "MCP tool returned an error result"
            raise McpToolExecutionError(redact_mcp_secret_text(message, self._settings))
        return output

    async def read_resource(self, server_name: str, uri: str) -> str:
        session = self._session(server_name)
        result = await session.read_resource(uri)
        return _stringify_resource_result(result)

    async def _connect_stdio(self, name: str, config: McpStdioServerConfig) -> None:
        stack = AsyncExitStack()
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env or None,
            cwd=config.cwd,
        )
        read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
        await self._register_session(name, stack, read_stream, write_stream)

    async def _connect_http(self, name: str, config: McpHttpServerConfig) -> None:
        stack = AsyncExitStack()
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(headers=config.headers or None),
        )
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(config.url, http_client=http_client),
        )
        await self._register_session(name, stack, read_stream, write_stream)

    async def _register_session(
        self,
        name: str,
        stack: AsyncExitStack,
        read_stream: Any,
        write_stream: Any,
    ) -> None:
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        tools = await self._list_session_tools(name, session)
        resources = await self._list_session_resources(name, session)
        self._sessions[name] = session
        self._stacks.append(stack)
        self._tools.extend(tools)
        self._resources.extend(resources)
        self._statuses[name] = McpConnectionStatus(
            name=name,
            state="connected",
            tools=tools,
            resources=resources,
        )

    async def _list_session_tools(self, server_name: str, session: ClientSession) -> list[McpToolInfo]:
        result = await session.list_tools()
        tools = []
        for tool in result.tools:
            risk_level = self._settings.tool_risks.get(f"{server_name}.{tool.name}", "high")
            tools.append(
                McpToolInfo(
                    server_name=server_name,
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema),
                    risk_level=risk_level,
                ),
            )
        return tools

    async def _list_session_resources(
        self,
        server_name: str,
        session: ClientSession,
    ) -> list[McpResourceInfo]:
        result = await session.list_resources()
        return [
            McpResourceInfo(
                server_name=server_name,
                uri=str(resource.uri),
                name=resource.name,
                description=resource.description,
                mime_type=resource.mimeType,
            )
            for resource in result.resources
        ]

    def _session(self, server_name: str) -> ClientSession:
        try:
            return self._sessions[server_name]
        except KeyError as error:
            raise McpServerNotConnectedError(f"MCP server is not connected: {server_name}") from error


def _stringify_tool_result(result: CallToolResult) -> str:
    parts = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    if result.structuredContent is not None:
        return str(result.structuredContent)
    return ""


def _stringify_resource_result(result: ReadResourceResult) -> str:
    parts = []
    for item in result.contents:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            blob = getattr(item, "blob", None)
            if blob is not None:
                parts.append(blob)
    return "\n".join(parts)
