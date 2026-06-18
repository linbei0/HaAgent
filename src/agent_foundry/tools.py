from __future__ import annotations

import time
from typing import Any

from agent_foundry.episode import EpisodeWriter


class ToolRoutingError(RuntimeError):
    """Raised when a tool cannot be routed or is not allowed."""


class ToolRouter:
    def __init__(self, allowed_tools: list[str], episode_writer: EpisodeWriter) -> None:
        self._allowed_tools = set(allowed_tools)
        self._episode_writer = episode_writer

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if tool_name not in self._allowed_tools:
                raise ToolRoutingError(f"tool is not allowed: {tool_name}")
            if tool_name != "fake_tool":
                raise ToolRoutingError(f"unknown tool: {tool_name}")

            result = {"status": "success", "args": args}
            self._write_trace(tool_name, args, "success", started, result=result)
            return result
        except ToolRoutingError as error:
            self._write_trace(tool_name, args, "error", started, error=str(error))
            raise

    def _write_trace(
        self,
        tool_name: str,
        args: dict[str, Any],
        status: str,
        started: float,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self._episode_writer.append_tool_call(
            {
                "tool_name": tool_name,
                "args": args,
                "status": status,
                "result": result,
                "error": error,
                "duration_seconds": time.perf_counter() - started,
            },
        )
