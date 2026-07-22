"""
haagent/tools/base.py - 工具通用类型

定义工具错误、结构化错误结果和工具处理函数签名。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal

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


class ToolFailureCategory(StrEnum):
    """工具调用内部使用的稳定失败分类。"""

    ARGUMENT = "argument"
    NOT_FOUND = "not_found"
    PERMISSION = "permission"
    POLICY = "policy"
    TRANSIENT = "transient"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    EXECUTION = "execution"
    CONTRACT = "contract"
    CANCELLED = "cancelled"


RecoveryActionName = Literal[
    "correct_arguments",
    "retry_same_call",
    "use_tool",
    "use_alternate_source",
    "inspect_state",
    "ask_user",
    "stop",
]


@dataclass(frozen=True)
class RecoveryAction:
    """失败后交给模型或 orchestrator 的确定性恢复动作。"""

    action: RecoveryActionName
    reason: str
    tool_name: str | None = None
    args: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.action, "reason": self.reason}
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.args is not None:
            payload["args"] = dict(self.args)
        return payload


class ToolRoutingError(RuntimeError):
    """Raised when orchestration wants to fail a run on tool errors."""

    def __init__(self, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


def tool_error(
    error_type: str,
    message: str,
    *,
    category: ToolFailureCategory | str | None = None,
    retryable: bool | None = None,
    recovery: RecoveryAction | dict[str, Any] | None = None,
    execution_state: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """构造稳定工具错误，不再让调用方自由拼接恢复字段。"""

    default_category, default_retryable, default_recovery = _failure_defaults(error_type)
    active_category = ToolFailureCategory(category) if category is not None else default_category
    active_retryable = default_retryable if retryable is None else retryable
    error: dict[str, Any] = {
        "type": error_type,
        "category": active_category.value,
        "message": message,
        "retryable": active_retryable,
    }
    for key, value in details.items():
        if value is not None:
            error[key] = value
    if execution_state is not None:
        error["execution_state"] = execution_state
    active_recovery = recovery if recovery is not None else default_recovery
    result: dict[str, Any] = {"status": "error", "error": error}
    if execution_state is not None:
        result["execution_state"] = execution_state
    if active_recovery is not None:
        result["recovery"] = (
            active_recovery.to_dict()
            if isinstance(active_recovery, RecoveryAction)
            else dict(active_recovery)
        )
    return result


def _failure_defaults(
    error_type: str,
) -> tuple[ToolFailureCategory, bool, RecoveryAction | None]:
    """集中给旧工具错误补齐分类；相同调用自动重试仅限 transient/timeout。"""

    if error_type in {"tool_argument_invalid", "invalid_tool_arguments", "unknown_tool", "tool_not_allowed"}:
        return ToolFailureCategory.ARGUMENT, False, RecoveryAction(
            "correct_arguments", "根据工具 schema 修正参数或选择已允许的工具。",
        )
    if error_type in {"file_not_found", "keyword_not_found", "patch_text_not_found", "job_not_found"}:
        return ToolFailureCategory.NOT_FOUND, False, None
    if error_type in {"approval_required", "approval_pending"}:
        return ToolFailureCategory.PERMISSION, False, RecoveryAction(
            "ask_user", "需要用户批准后才能继续。",
        )
    if error_type in {"approval_denied", "policy_denied", "guardrail_denied", "path_policy_denied"}:
        return ToolFailureCategory.POLICY, False, RecoveryAction(
            "stop", "权限或安全策略已拒绝本次调用。",
        )
    if error_type in {"timeout", "web_connect_timeout", "web_read_timeout"}:
        return ToolFailureCategory.TIMEOUT, True, RecoveryAction(
            "retry_same_call", "这是可安全重放的临时超时。",
        )
    if error_type in {"web_dns_failed", "web_proxy_failed", "web_network_failed", "temporary_io_error"}:
        return ToolFailureCategory.TRANSIENT, True, RecoveryAction(
            "retry_same_call", "这是可安全重放的临时连接失败。",
        )
    if error_type in {"web_search_configuration_error", "web_search_failed", "web_http_error"}:
        return ToolFailureCategory.PROVIDER, False, None
    if error_type in {"tool_registry_invalid", "tool_contract_invalid"}:
        return ToolFailureCategory.CONTRACT, False, RecoveryAction(
            "stop", "工具实现或 registry 合同无效，不能继续猜测执行。",
        )
    if error_type in {"RunCancelled", "run_cancelled", "cancelled"}:
        return ToolFailureCategory.CANCELLED, False, None
    if error_type in {"patch_text_not_unique", "code_run_failed", "search_failed", "job_start_failed"}:
        return ToolFailureCategory.EXECUTION, False, None
    return ToolFailureCategory.EXECUTION, False, None
