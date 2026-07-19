"""
haagent/tools/base.py - 工具通用类型

定义工具错误、结构化错误结果和工具处理函数签名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from haagent.runtime.execution.human_interaction import (
    HumanInteractionHandler,
    HumanInteractionRequest,
    HumanInteractionResponse,
    ToolPermissionRequest,
)


@dataclass(frozen=True)
class ToolExecutionContext:
    """逐次工具执行上下文。

    Router 和 handler 统一通过 ask 申请权限；handler 不直接操作 TUI。
    """

    interaction_handler: HumanInteractionHandler | None = None

    def ask(self, request: ToolPermissionRequest) -> HumanInteractionResponse | None:
        """暂停当前工具调用等待用户决定；无交互入口时返回 None。"""
        if self.interaction_handler is None:
            return None
        summary = dict(request.metadata)
        summary["permission_patterns"] = list(request.patterns)
        summary["permission_always"] = list(request.always)
        question = request.question or f"允许权限 {request.permission} 吗？"
        return self.interaction_handler(
            HumanInteractionRequest(
                interaction_type="approval",
                tool_name=request.permission,
                question=question,
                reason=request.reason,
                risk_level=request.risk_level,
                args_summary=summary,
            ),
        )


ToolHandler = Callable[[dict[str, Any], ToolExecutionContext], dict[str, Any]]


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
    if error_type == "tool_argument_invalid" and "retryable" not in details:
        # 参数错误来自本次模型调用，默认把结构化错误交回下一轮修正；
        # 真正不可恢复的工具/registry 故障应使用独立类型或显式 retryable=False。
        error["retryable"] = True
    # 额外结构化字段（如 failure_stage）供工具结果与 UI 摘要使用；不强制所有调用方提供。
    for key, value in details.items():
        if value is not None:
            error[key] = value
    return {"status": "error", "error": error}
