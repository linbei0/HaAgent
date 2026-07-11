"""
haagent/tools/base.py - 工具通用类型

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


def tool_error(
    error_type: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    error: dict[str, Any] = {"type": error_type, "message": message}
    # 额外结构化字段（如 failure_stage）供工具结果与 UI 摘要使用；不强制所有调用方提供。
    for key, value in details.items():
        if value is not None:
            error[key] = value
    return {"status": "error", "error": error}
