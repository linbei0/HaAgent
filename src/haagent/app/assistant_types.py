"""
haagent/app/assistant_types.py - 个人助手应用层稳定类型

集中定义 CLI、TUI 与应用 Module 共享的状态、结果、请求和真实可替换 Seam。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from haagent.models.types import ModelGateway
from haagent.channels.settings import ChannelPermissionMode
from haagent.runtime.events import RuntimeUiEvent
from haagent.runtime.execution.path_policy import PermissionMode
from haagent.runtime.sandbox.status import SandboxDoctorReport
from haagent.scheduling.models import (
    DestinationKind,
    FailureCategory,
    MisfirePolicy,
    OverlapPolicy,
    RetryPolicy,
    RunStatus,
    ScheduleStatus,
    TriggerKind,
)


GatewayFactory = Callable[..., ModelGateway]
EventSink = Callable[[RuntimeUiEvent], None]


class AssistantServiceError(RuntimeError):
    """应用层无法完成显式请求时抛出。"""


@dataclass(frozen=True)
class AssistantSandboxStatus:
    backend: str
    degraded: bool
    reason: str


@dataclass(frozen=True)
class AssistantWorkspaceStatus:
    workspace_root: Path
    runs_root: Path
    profile_name: str | None
    provider: str | None
    base_url: str | None
    model: str | None
    api_key_env: str | None
    api_key_available: bool
    credential_source_configured: str | None = None
    credential_source_used: str | None = None
    credential_store_available: bool | None = None
    credential_store_error: str | None = None
    profile_error: str | None = None
    current_session_id: str | None = None
    current_turn_count: int | None = None
    web_enabled: bool = False
    external_roots: list[dict[str, str]] | None = None
    permission_mode: PermissionMode = "request_approval"
    sandbox_status: AssistantSandboxStatus = AssistantSandboxStatus(
        backend="local_subprocess",
        degraded=True,
        reason="docker sandbox disabled",
    )
    image_input_supported: bool | None = None
    model_variant: str | None = None


@dataclass(frozen=True)
class AssistantTurnLimitStatus:
    current_max_turns: int | None
    configured_interactive_max_turns: int


@dataclass(frozen=True)
class AssistantSessionStatus:
    session_id: str
    workspace_root: Path
    runs_root: Path
    session_path: Path
    turn_count: int
    max_turns: int | None
    provider: str
    model_profile_name: str | None = None
    model_connection_id: str | None = None
    model: str | None = None
    model_variant: str | None = None
    base_url: str | None = None
    web_enabled: bool = False
    external_roots: list[dict[str, str]] | None = None
    permission_mode: PermissionMode = "request_approval"
    sandbox_status: AssistantSandboxStatus = AssistantSandboxStatus(
        backend="local_subprocess",
        degraded=True,
        reason="docker sandbox disabled",
    )


@dataclass(frozen=True)
class AssistantSessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    workspace_root: Path
    turn_count: int
    first_request: str
    session_path: Path


@dataclass(frozen=True)
class AssistantSessionTurn:
    turn_index: int
    request: str
    summary: str
    status: str
    episode_path: Path
    verification_status: str
    assistant_display_text: str | None = None


@dataclass(frozen=True)
class AssistantSessionCompactResult:
    applied: bool
    reason: str
    original_turn_count: int
    compacted_turn_count: int
    preserved_recent_count: int
    saved_chars: int


@dataclass(frozen=True)
class AssistantCancelResult:
    status: str
    reason: str


@dataclass(frozen=True)
class AssistantModelConnection:
    id: str
    name: str
    provider_id: str
    provider_name: str
    gateway_provider: str
    base_url: str
    api_key_env: str
    credential_source: str
    credential_available: bool
    credential_source_used: str | None
    model_config_diagnostics: tuple[str, ...]
    runtime_kind: str = "remote"


@dataclass(frozen=True)
class AssistantModelTestResult:
    ok: bool
    profile_name: str
    provider: str
    model: str
    message: str


@dataclass(frozen=True)
class AssistantSkillList:
    skills: list[dict[str, object]]
    blocked_project_skill_roots: list[str]


@dataclass(frozen=True)
class AssistantSkillContent:
    name: str
    command_name: str
    content: str


@dataclass(frozen=True)
class AssistantMarketplaceSkill:
    result_id: str
    provider: str
    name: str
    source: str
    summary: str
    detail_url: str
    installable: bool
    quality: dict[str, int | float | str]


@dataclass(frozen=True)
class AssistantMarketplaceSearch:
    status: str
    query: str
    results: list[AssistantMarketplaceSkill]
    warnings: list[str]


@dataclass(frozen=True)
class AssistantMarketplaceInstall:
    name: str
    command_name: str
    skill_dir: Path
    skill_file: Path
    source_url: str


@dataclass(frozen=True)
class ModelConnectionConfigureRequest:
    id: str
    name: str
    provider_id: str
    provider_name: str
    gateway_provider: str
    base_url: str
    api_key_env: str
    credential_source: str
    api_key: str | None = None
    runtime_kind: str = "remote"


@dataclass(frozen=True)
class ModelSelectionRequest:
    connection_id: str
    model: str
    variant: str | None = None


@dataclass(frozen=True, repr=False)
class AssistantChannelInstance:
    id: str
    platform: str
    enabled: bool
    workspace_root: Path
    credential_username: str
    credential_available: bool
    state: str
    metadata: dict[str, str]
    permission_mode: ChannelPermissionMode = "request_approval"

    def __repr__(self) -> str:
        # 禁止在日志/测试失败输出中泄露 bot_token 等 secret。
        return (
            f"AssistantChannelInstance(id={self.id!r}, platform={self.platform!r}, "
            f"enabled={self.enabled!r}, state={self.state!r}, "
            f"permission_mode={self.permission_mode!r}, "
            f"credential_available={self.credential_available!r})"
        )


@dataclass(frozen=True, repr=False)
class AssistantChannelQrStart:
    instance_id: str
    qrcode_id: str
    qrcode_url: str

    def __repr__(self) -> str:
        return (
            f"AssistantChannelQrStart(instance_id={self.instance_id!r}, "
            f"qrcode_id={self.qrcode_id!r})"
        )


@dataclass(frozen=True, repr=False)
class AssistantChannelQrPoll:
    status: str
    instance_id: str
    credential_available: bool = False
    message: str = ""
    # 仅 confirmed 时填充；只展示一次，不写入配置/日志 repr。
    pairing_code: str | None = None

    def __repr__(self) -> str:
        return (
            f"AssistantChannelQrPoll(status={self.status!r}, "
            f"instance_id={self.instance_id!r}, "
            f"credential_available={self.credential_available!r}, "
            f"has_pairing_code={self.pairing_code is not None})"
        )


@dataclass(frozen=True, repr=False)
class AssistantChannelTestResult:
    ok: bool
    instance_id: str
    message: str

    def __repr__(self) -> str:
        return (
            f"AssistantChannelTestResult(ok={self.ok!r}, "
            f"instance_id={self.instance_id!r}, message={self.message!r})"
        )


@dataclass(frozen=True)
class ScheduleCreateRequest:
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
class ScheduleUpdateRequest:
    expected_revision: int
    name: str | None = None
    prompt: str | None = None
    workspace_root: Path | None = None
    destination_kind: DestinationKind | None = None
    destination_session_path: Path | None | object = ...
    connection_id: str | None = None
    model: str | None = None
    web_enabled: bool | None = None
    allowed_tools: tuple[str, ...] | None = None
    approval_allowed_tools: tuple[str, ...] | None = None
    approved_tools: tuple[str, ...] | None = None
    permission_mode: PermissionMode | None = None
    dtstart_local: datetime | None = None
    timezone: str | None = None
    rrule: str | None | object = ...
    misfire_policy: MisfirePolicy | None = None
    overlap_policy: OverlapPolicy | None = None
    retry_policy: RetryPolicy | None = None


@dataclass(frozen=True)
class SchedulePreviewRequest:
    dtstart_local: datetime
    timezone: str
    rrule: str | None
    after: datetime | None = None


@dataclass(frozen=True)
class RunQuery:
    schedule_id: str | None = None
    unread_only: bool = False
    status: RunStatus | None = None
    limit: int = 100


@dataclass(frozen=True)
class AssistantScheduleSummary:
    id: str
    name: str
    status: ScheduleStatus
    timezone: str
    rrule: str | None
    next_run_at_utc: datetime | None
    last_run_at_utc: datetime | None
    workspace_root: Path
    revision: int
    model: str
    connection_id: str


@dataclass(frozen=True)
class AssistantSchedule:
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
    next_run_at_utc: datetime | None = None
    last_run_at_utc: datetime | None = None


@dataclass(frozen=True)
class AssistantScheduleRun:
    id: str
    schedule_id: str
    schedule_revision: int
    trigger_key: str
    trigger_kind: TriggerKind
    scheduled_for_utc: datetime
    status: RunStatus
    attempt_count: int
    summary: str
    unread: bool
    session_id: str | None = None
    session_path: str | None = None
    episode_path: str | None = None
    failure_category: FailureCategory | None = None
    failure_reason: str | None = None
    needs_attention_reason: str | None = None
    started_at_utc: datetime | None = None
    finished_at_utc: datetime | None = None
    cancellation_requested: bool = False


@dataclass(frozen=True)
class BackgroundServiceStatus:
    state: str
    host_type: str
    detail: str = ""
    executable: str | None = None
    last_heartbeat_utc: datetime | None = None


@dataclass(frozen=True)
class ScheduleHostStatus:
    """TUI 内嵌 schedule worker 的可展示状态；不暴露内部对象。"""

    running: bool
    owner_id: str | None = None
    last_error: str | None = None
    fatal: bool = False
