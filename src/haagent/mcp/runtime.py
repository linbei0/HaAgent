"""
src/haagent/mcp/runtime.py - 同步 MCP runtime 包装

用后台 asyncio 事件循环承载 MCP client，向 HaAgent 同步 runtime 暴露调用接口。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Thread
from typing import Any, Coroutine

from haagent.mcp.client import McpClientManager
from haagent.mcp.types import McpConnectionStatus, McpResourceInfo, McpSettings, McpToolInfo


class SyncMcpRuntime:
    def __init__(self, settings: McpSettings) -> None:
        self._settings = settings
        self._manager = McpClientManager(settings)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if not self._settings.servers:
            self._started = True
            return
        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, name="haagent-mcp-runtime", daemon=True)
        self._thread.start()
        self._run(self._manager.connect_all())
        self._started = True

    def close(self) -> None:
        if self._loop is None or self._thread is None:
            self._started = False
            return
        try:
            if self._started:
                self._run(self._manager.close())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
            self._loop.close()
            self._loop = None
            self._thread = None
            self._started = False

    def list_statuses(self) -> list[McpConnectionStatus]:
        return self._manager.list_statuses()

    def list_tools(self) -> list[McpToolInfo]:
        return self._manager.list_tools()

    def list_resources(self) -> list[McpResourceInfo]:
        return self._manager.list_resources()

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        return self._run(self._manager.call_tool(server_name, tool_name, arguments))

    def read_resource(self, server_name: str, uri: str) -> str:
        return self._run(self._manager.read_resource(server_name, uri))

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._loop is None:
            raise RuntimeError("MCP runtime has not been started")
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()
