"""
haagent/app/assistant_types.py - 个人助手应用层稳定类型

集中定义 CLI、TUI 与应用 Module 共享的状态、结果、请求和真实可替换 Seam。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from haagent.models.gateway_registry import GatewayCapability
from haagent.models.types import ModelGateway
from haagent.runtime.events import RuntimeUiEvent
from haagent.runtime.execution.path_policy import PermissionMode
from haagent.runtime.sandbox.status import SandboxDoctorReport


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
class AssistantModelProfile:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str
    credential_source: str
    active: bool
    credential_available: bool
    credential_source_used: str | None
    capability: GatewayCapability
    current_session: bool = False


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


@dataclass(frozen=True)
class ModelSelectionRequest:
    connection_id: str
    model: str
