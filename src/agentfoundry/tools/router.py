"""
agentfoundry/tools/router.py - 工具路由器

校验 allowed_tools，分发本地工具，并为每次调用写入 tool-calls.jsonl。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.tools.base import ToolHandler, ToolRoutingError, tool_error
from agentfoundry.tools.file_tools import apply_patch, file_read, file_search
from agentfoundry.tools.registry import TOOL_REGISTRY
from agentfoundry.tools.shell import shell


class ToolRouter:
    def __init__(
        self,
        allowed_tools: list[str],
        episode_writer: EpisodeWriter,
        workspace_root: Path,
    ) -> None:
        self._allowed_tools = set(allowed_tools)
        self._episode_writer = episode_writer
        self._workspace_root = workspace_root.resolve()
        self._handlers: dict[str, ToolHandler] = {
            "fake_tool": self._fake_tool,
            "file_search": lambda args: file_search(args, self._workspace_root),
            "file_read": lambda args: file_read(args, self._workspace_root),
            "apply_patch": lambda args: apply_patch(args, self._workspace_root),
            "shell": lambda args: shell(args, self._workspace_root),
        }
        self._assert_registry_alignment()

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """执行工具并保证每次调用都写入 tool-calls.jsonl。"""
        started = time.perf_counter()
        try:
            if tool_name not in self._allowed_tools:
                result = tool_error("tool_not_allowed", f"tool is not allowed: {tool_name}")
            elif tool_name not in self._handlers:
                result = tool_error("unknown_tool", f"unknown tool: {tool_name}")
            else:
                result = self._handlers[tool_name](args)
        except Exception as error:
            result = tool_error(type(error).__name__, str(error))

        self._write_trace(tool_name, args, result, started)
        return result

    def raise_for_error(self, result: dict[str, Any]) -> None:
        if result.get("status") == "error":
            error = result.get("error") or {}
            raise ToolRoutingError(str(error.get("message", "tool failed")))

    def _fake_tool(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "args": args}

    def _assert_registry_alignment(self) -> None:
        """Router 和 Registry 必须同步，否则 allowed_tools 审计会和实际执行脱节。"""
        if set(self._handlers) != set(TOOL_REGISTRY):
            missing = sorted(set(TOOL_REGISTRY) - set(self._handlers))
            extra = sorted(set(self._handlers) - set(TOOL_REGISTRY))
            raise ToolRoutingError(f"tool registry mismatch: missing={missing}, extra={extra}")

    def _write_trace(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        started: float,
    ) -> None:
        self._episode_writer.append_tool_call(
            {
                "tool_name": tool_name,
                "args": args,
                "status": result["status"],
                "result": result if result["status"] == "success" else None,
                "error": result.get("error"),
                "duration_seconds": time.perf_counter() - started,
            },
        )
