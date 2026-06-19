"""
agentfoundry/tools/base.py - 工具通用类型

定义工具错误、结构化错误结果和工具处理函数签名。
"""

from __future__ import annotations

from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRoutingError(RuntimeError):
    """Raised when orchestration wants to fail a run on tool errors."""

    def __init__(self, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


def tool_error(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error": {"type": error_type, "message": message}}
