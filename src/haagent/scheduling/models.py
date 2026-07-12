"""
haagent/scheduling/models.py - 计划任务领域类型与纯验证

定义 ScheduleDefinition、ScheduleRun、RetryPolicy 及状态枚举；
validate_schedule 只做纯函数校验，不访问磁盘或当前时间。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from haagent.runtime.execution.path_policy import PERMISSION_MODES, PermissionMode

# 计划 web_enabled=True 时必须可用的联网工具（与编辑器/执行器一致）
SCHEDULE_WEB_TOOLS: tuple[str, ...] = ("web_search", "web_fetch")

# parallel 禁止的副作用工具（写文件/补丁/命令/代码执行）
PARALLEL_FORBIDDEN_TOOLS: frozenset[str] = frozenset(
    {
        "file_write",
        "apply_patch",
        "apply_patch_set",
        "shell",
        "code_run",
    }
)


def merge_web_tools(
    allowed_tools: Sequence[str], *, web_enabled: bool
) -> tuple[str, ...]:
    """web_enabled 时把联网工具并入 allowed_tools；关闭时原样返回。"""
    tools = list(allowed_tools)
    if not web_enabled:
        return tuple(tools)
    for name in SCHEDULE_WEB_TOOLS:
        if name not in tools:
            tools.append(name)
    return tuple(tools)


@dataclass(frozen=True)
class RunClaim:
    """claim 后不可变 token；worker/executor/finish 全程传递同一对象。"""

    run_id: str
    worker_id: str
    attempt: int


ScheduleStatus = Literal["active", "paused", "completed", "error", "archived"]
RunStatus = Literal[
    "queued",
    "running",
    "retry_wait",
    "succeeded",
    "failed",
    "needs_attention",
    "cancelled",
    "skipped",
    "interrupted",
]
DestinationKind = Literal["new_session", "resume_session"]
MisfirePolicy = Literal["skip", "latest", "all"]
OverlapPolicy = Literal["skip", "queue", "parallel"]
TriggerKind = Literal["scheduled", "manual"]
FailureCategory = Literal[
    "profile_unavailable",
    "credential_unavailable",
    "workspace_unavailable",
    "schedule_invalid",
    "policy_denied",
    "interaction_required",
    "model_transient",
    "model_permanent",
    "tool_failure",
    "verification_failed",
    "cancelled",
    "worker_interrupted",
    "internal_error",
]

SCHEDULE_STATUSES: frozenset[str] = frozenset(
    {"active", "paused", "completed", "error", "archived"}
)
RUN_STATUSES: frozenset[str] = frozenset(
    {
        "queued",
        "running",
        "retry_wait",
        "succeeded",
        "failed",
        "needs_attention",
        "cancelled",
        "skipped",
        "interrupted",
    }
)
DESTINATION_KINDS: frozenset[str] = frozenset({"new_session", "resume_session"})
MISFIRE_POLICIES: frozenset[str] = frozenset({"skip", "latest", "all"})
OVERLAP_POLICIES: frozenset[str] = frozenset({"skip", "queue", "parallel"})
FAILURE_CATEGORIES: frozenset[str] = frozenset(
    {
        "profile_unavailable",
        "credential_unavailable",
        "workspace_unavailable",
        "schedule_invalid",
        "policy_denied",
        "interaction_required",
        "model_transient",
        "model_permanent",
        "tool_failure",
        "verification_failed",
        "cancelled",
        "worker_interrupted",
        "internal_error",
    }
)


class ScheduleValidationError(ValueError):
    """计划定义校验失败；code 供测试与 UI 结构化映射。"""

    def __init__(self, code: str, message: str) -> None:
        # 失败边界：校验拒绝必须带稳定 code，禁止静默吞错
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: int = 30
    multiplier: float = 2.0
    max_delay_seconds: int = 900


@dataclass(frozen=True)
class ScheduleDefinition:
    id: str
    name: str
    prompt: str
    workspace_root: Path
    destination_kind: DestinationKind
    destination_session_path: Path | None
    connection_id: str
    model: str
    web_enabled: bool
    allowed_tools: tuple[str, ...]
    approval_allowed_tools: tuple[str, ...]
    approved_tools: tuple[str, ...]
    permission_mode: PermissionMode
    dtstart_local: datetime
    timezone: str
    rrule: str | None
    status: ScheduleStatus
    misfire_policy: MisfirePolicy
    overlap_policy: OverlapPolicy
    retry_policy: RetryPolicy
    revision: int


@dataclass(frozen=True)
class ScheduleRun:
    id: str
    schedule_id: str
    schedule_revision: int
    trigger_key: str
    trigger_kind: TriggerKind
    scheduled_for_utc: datetime
    status: RunStatus
    attempt_count: int = 0
    retry_at_utc: datetime | None = None
    worker_id: str | None = None
    lease_expires_at_utc: datetime | None = None
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None
    session_id: str | None = None
    session_path: str | None = None
    episode_path: str | None = None
    summary: str = ""
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    needs_attention_reason: str | None = None
    unread: bool = True
    cancellation_requested: bool = False


def validate_schedule(definition: ScheduleDefinition) -> ScheduleDefinition:
    """校验计划定义；通过时返回同一实例。"""
    if not definition.name or not definition.name.strip():
        raise ScheduleValidationError("empty_name", "计划名称不能为空")
    if not definition.prompt or not definition.prompt.strip():
        raise ScheduleValidationError("empty_prompt", "计划提示词不能为空")

    workspace = definition.workspace_root
    if not workspace.is_absolute():
        raise ScheduleValidationError(
            "relative_workspace",
            "workspace_root 必须是绝对路径",
        )

    if definition.destination_kind not in DESTINATION_KINDS:
        raise ScheduleValidationError(
            "invalid_destination",
            f"无效 destination_kind: {definition.destination_kind!r}",
        )

    if definition.destination_kind == "resume_session":
        if definition.destination_session_path is None:
            raise ScheduleValidationError(
                "resume_requires_session",
                "resume_session 必须提供 destination_session_path",
            )
        if definition.overlap_policy == "parallel":
            # 安全边界：续接 session 禁止并行，避免并发写 session package
            raise ScheduleValidationError(
                "resume_forbids_parallel",
                "resume_session 禁止 overlap_policy=parallel",
            )
    elif definition.destination_session_path is not None:
        raise ScheduleValidationError(
            "new_session_forbids_path",
            "new_session 不得设置 destination_session_path",
        )

    if definition.overlap_policy == "parallel":
        # parallel 仅允许无副作用只读工具，防止并发改同一 workspace
        side = sorted(
            t for t in definition.allowed_tools if t in PARALLEL_FORBIDDEN_TOOLS
        )
        if side:
            raise ScheduleValidationError(
                "parallel_forbids_side_effects",
                f"overlap_policy=parallel 禁止副作用工具: {', '.join(side)}",
            )

    approved = set(definition.approved_tools)
    approval_allowed = set(definition.approval_allowed_tools)
    if not approved.issubset(approval_allowed):
        raise ScheduleValidationError(
            "approved_not_subset",
            "approved_tools 必须是 approval_allowed_tools 的子集",
        )

    retry = definition.retry_policy
    if retry.max_attempts < 1:
        raise ScheduleValidationError(
            "retry_max_attempts",
            "retry_policy.max_attempts 必须 >= 1",
        )
    if retry.initial_delay_seconds < 0:
        raise ScheduleValidationError(
            "retry_initial_delay",
            "retry_policy.initial_delay_seconds 必须 >= 0",
        )
    if retry.multiplier < 1.0:
        raise ScheduleValidationError(
            "retry_multiplier",
            "retry_policy.multiplier 必须 >= 1.0",
        )
    if retry.max_delay_seconds < 1:
        raise ScheduleValidationError(
            "retry_max_delay",
            "retry_policy.max_delay_seconds 必须 >= 1",
        )
    if retry.initial_delay_seconds > retry.max_delay_seconds:
        raise ScheduleValidationError(
            "retry_delay_order",
            "initial_delay_seconds 不得大于 max_delay_seconds",
        )

    if not definition.timezone or not definition.timezone.strip():
        raise ScheduleValidationError("empty_timezone", "timezone 不能为空")

    if definition.status not in SCHEDULE_STATUSES:
        raise ScheduleValidationError(
            "invalid_status",
            f"无效 status: {definition.status!r}",
        )
    if definition.misfire_policy not in MISFIRE_POLICIES:
        raise ScheduleValidationError(
            "invalid_misfire",
            f"无效 misfire_policy: {definition.misfire_policy!r}",
        )
    if definition.overlap_policy not in OVERLAP_POLICIES:
        raise ScheduleValidationError(
            "invalid_overlap",
            f"无效 overlap_policy: {definition.overlap_policy!r}",
        )
    if definition.permission_mode not in PERMISSION_MODES:
        raise ScheduleValidationError(
            "invalid_permission_mode",
            f"无效 permission_mode: {definition.permission_mode!r}",
        )
    if definition.revision < 1:
        raise ScheduleValidationError("invalid_revision", "revision 必须 >= 1")

    if not definition.id or not str(definition.id).strip():
        raise ScheduleValidationError("empty_id", "计划 id 不能为空")
    if not definition.connection_id or not definition.connection_id.strip():
        raise ScheduleValidationError("empty_connection", "connection_id 不能为空")
    if not definition.model or not definition.model.strip():
        raise ScheduleValidationError("empty_model", "model 不能为空")

    return definition
