"""
src/haagent/scheduling/draft.py - ScheduleDraft / SchedulePatch 领域输入

create 用完整 Draft 规范化；update 用显式 Patch（unchanged/clear/set）。
Store 只接收完整 ScheduleDefinition + revision，不认识 UI request。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from haagent.scheduling.models import (
    DestinationKind,
    MisfirePolicy,
    OverlapPolicy,
    PermissionMode,
    RetryPolicy,
    ScheduleDefinition,
    ScheduleStatus,
    ScheduleValidationError,
    merge_web_tools,
    validate_schedule,
)
from haagent.scheduling.recurrence import RecurrenceError, normalize_rrule

T = TypeVar("T")


@dataclass(frozen=True)
class FieldPatch(Generic[T]):
    """可空字段的三态 patch：不改 / 清空 / 设值。"""

    kind: Literal["unchanged", "clear", "set"] = "unchanged"
    value: T | None = None

    def __post_init__(self) -> None:
        # 拒绝非法 kind 与矛盾 value，避免静默当成 set。
        if self.kind not in ("unchanged", "clear", "set"):
            raise ValueError(f"FieldPatch kind must be unchanged|clear|set, got {self.kind!r}")
        if self.kind == "unchanged" and self.value is not None:
            raise ValueError("FieldPatch unchanged must not carry a value")
        if self.kind == "clear" and self.value is not None:
            raise ValueError("FieldPatch clear must not carry a value")
        if self.kind == "set" and self.value is None:
            raise ValueError("FieldPatch set requires a non-None value; use clear() for null")

    @classmethod
    def unchanged(cls) -> FieldPatch[T]:
        return cls("unchanged", None)

    @classmethod
    def clear(cls) -> FieldPatch[T]:
        return cls("clear", None)

    @classmethod
    def set(cls, value: T) -> FieldPatch[T]:
        return cls("set", value)

    def apply(self, current: T | None) -> T | None:
        if self.kind == "unchanged":
            return current
        if self.kind == "clear":
            return None
        return self.value


UNCHANGED: FieldPatch[Any] = FieldPatch.unchanged()


@dataclass(frozen=True)
class ScheduleDraft:
    """创建输入：完整 body，无 id/status/revision。"""

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
    misfire_policy: MisfirePolicy = "latest"
    overlap_policy: OverlapPolicy = "skip"
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True)
class SchedulePatch:
    """更新输入：expected_revision + 字段级 tri-state patch。"""

    expected_revision: int
    name: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    prompt: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    workspace_root: FieldPatch[Path] = field(default_factory=FieldPatch.unchanged)
    destination_kind: FieldPatch[DestinationKind] = field(default_factory=FieldPatch.unchanged)
    destination_session_path: FieldPatch[Path] = field(default_factory=FieldPatch.unchanged)
    connection_id: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    model: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    web_enabled: FieldPatch[bool] = field(default_factory=FieldPatch.unchanged)
    allowed_tools: FieldPatch[tuple[str, ...]] = field(default_factory=FieldPatch.unchanged)
    approval_allowed_tools: FieldPatch[tuple[str, ...]] = field(default_factory=FieldPatch.unchanged)
    approved_tools: FieldPatch[tuple[str, ...]] = field(default_factory=FieldPatch.unchanged)
    permission_mode: FieldPatch[PermissionMode] = field(default_factory=FieldPatch.unchanged)
    dtstart_local: FieldPatch[datetime] = field(default_factory=FieldPatch.unchanged)
    timezone: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    rrule: FieldPatch[str] = field(default_factory=FieldPatch.unchanged)
    misfire_policy: FieldPatch[MisfirePolicy] = field(default_factory=FieldPatch.unchanged)
    overlap_policy: FieldPatch[OverlapPolicy] = field(default_factory=FieldPatch.unchanged)
    retry_policy: FieldPatch[RetryPolicy] = field(default_factory=FieldPatch.unchanged)


def materialize_definition(
    *,
    schedule_id: str,
    status: ScheduleStatus,
    revision: int,
    name: str,
    prompt: str,
    workspace_root: Path,
    destination_kind: DestinationKind,
    destination_session_path: Path | None,
    connection_id: str,
    model: str,
    web_enabled: bool,
    allowed_tools: tuple[str, ...],
    approval_allowed_tools: tuple[str, ...],
    approved_tools: tuple[str, ...],
    permission_mode: PermissionMode,
    dtstart_local: datetime,
    timezone: str,
    rrule: str | None,
    misfire_policy: MisfirePolicy,
    overlap_policy: OverlapPolicy,
    retry_policy: RetryPolicy,
) -> ScheduleDefinition:
    """create/update 共享：normalize rrule + web tools + validate。"""
    try:
        normalized_rrule = normalize_rrule(rrule)
    except RecurrenceError as error:
        raise ScheduleValidationError("invalid_rrule", str(error)) from error
    definition = ScheduleDefinition(
        id=schedule_id,
        name=name,
        prompt=prompt,
        workspace_root=Path(workspace_root).resolve(),
        destination_kind=destination_kind,
        destination_session_path=(
            Path(destination_session_path).resolve()
            if destination_session_path is not None
            else None
        ),
        connection_id=connection_id,
        model=model,
        web_enabled=web_enabled,
        allowed_tools=merge_web_tools(allowed_tools, web_enabled=web_enabled),
        approval_allowed_tools=tuple(approval_allowed_tools),
        approved_tools=tuple(approved_tools),
        permission_mode=permission_mode,
        dtstart_local=dtstart_local,
        timezone=timezone,
        rrule=normalized_rrule,
        status=status,
        misfire_policy=misfire_policy,
        overlap_policy=overlap_policy,
        retry_policy=retry_policy or RetryPolicy(),
        revision=revision,
    )
    return validate_schedule(definition)


def definition_from_draft(
    draft: ScheduleDraft,
    *,
    schedule_id: str,
    status: ScheduleStatus = "active",
    revision: int = 1,
) -> ScheduleDefinition:
    return materialize_definition(
        schedule_id=schedule_id,
        status=status,
        revision=revision,
        name=draft.name,
        prompt=draft.prompt,
        workspace_root=draft.workspace_root,
        destination_kind=draft.destination_kind,
        destination_session_path=draft.destination_session_path,
        connection_id=draft.connection_id,
        model=draft.model,
        web_enabled=draft.web_enabled,
        allowed_tools=draft.allowed_tools,
        approval_allowed_tools=draft.approval_allowed_tools,
        approved_tools=draft.approved_tools,
        permission_mode=draft.permission_mode,
        dtstart_local=draft.dtstart_local,
        timezone=draft.timezone,
        rrule=draft.rrule,
        misfire_policy=draft.misfire_policy,
        overlap_policy=draft.overlap_policy,
        retry_policy=draft.retry_policy,
    )


def apply_patch(current: ScheduleDefinition, patch: SchedulePatch) -> ScheduleDefinition:
    """把 patch 应用到完整 definition，再走同一 materialize。"""
    name = patch.name.apply(current.name)
    prompt = patch.prompt.apply(current.prompt)
    workspace = patch.workspace_root.apply(current.workspace_root)
    dest_kind = patch.destination_kind.apply(current.destination_kind)
    dest_path = patch.destination_session_path.apply(current.destination_session_path)
    connection_id = patch.connection_id.apply(current.connection_id)
    model = patch.model.apply(current.model)
    web_enabled = patch.web_enabled.apply(current.web_enabled)
    allowed = patch.allowed_tools.apply(current.allowed_tools)
    approval_allowed = patch.approval_allowed_tools.apply(current.approval_allowed_tools)
    approved = patch.approved_tools.apply(current.approved_tools)
    permission_mode = patch.permission_mode.apply(current.permission_mode)
    dtstart = patch.dtstart_local.apply(current.dtstart_local)
    timezone = patch.timezone.apply(current.timezone)
    rrule = patch.rrule.apply(current.rrule)
    misfire = patch.misfire_policy.apply(current.misfire_policy)
    overlap = patch.overlap_policy.apply(current.overlap_policy)
    retry = patch.retry_policy.apply(current.retry_policy)

    if name is None or prompt is None or workspace is None or dest_kind is None:
        raise ScheduleValidationError("incomplete_patch", "name/prompt/workspace/destination 不能被清空")
    if connection_id is None or model is None or web_enabled is None:
        raise ScheduleValidationError("incomplete_patch", "connection/model/web_enabled 不能被清空")
    if allowed is None or approval_allowed is None or approved is None:
        raise ScheduleValidationError("incomplete_patch", "工具列表不能被清空为 null")
    if permission_mode is None or dtstart is None or timezone is None:
        raise ScheduleValidationError("incomplete_patch", "时间与权限字段不能被清空")
    if misfire is None or overlap is None or retry is None:
        raise ScheduleValidationError("incomplete_patch", "策略字段不能被清空")

    return materialize_definition(
        schedule_id=current.id,
        status=current.status,
        revision=current.revision + 1,
        name=name,
        prompt=prompt,
        workspace_root=workspace,
        destination_kind=dest_kind,
        destination_session_path=dest_path,
        connection_id=connection_id,
        model=model,
        web_enabled=bool(web_enabled),
        allowed_tools=tuple(allowed),
        approval_allowed_tools=tuple(approval_allowed),
        approved_tools=tuple(approved),
        permission_mode=permission_mode,
        dtstart_local=dtstart,
        timezone=timezone,
        rrule=rrule,
        misfire_policy=misfire,
        overlap_policy=overlap,
        retry_policy=retry,
    )


def with_status(definition: ScheduleDefinition, status: ScheduleStatus) -> ScheduleDefinition:
    return replace(definition, status=status, revision=definition.revision + 1)
