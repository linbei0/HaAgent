"""
haagent/app/assistant_context.py - 应用 Module 私有共享状态

保存 workspace、session 与模型选择状态，不进入 CLI 或 TUI 的公开 Interface。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from haagent.app.assistant_types import GatewayFactory
from haagent.models.model_connections import ModelSelection
from haagent.runtime.session.agent import AgentSession

if TYPE_CHECKING:
    from haagent.scheduling.store import ScheduleStore


@dataclass
class AssistantContext:
    workspace_root: Path
    runs_root: Path
    environ: Mapping[str, str]
    gateway_factory: GatewayFactory
    session_factory: type[AgentSession]
    max_turns: int | None
    enable_web: bool
    initial_resume: str | Path | None
    initial_continue: bool
    session: AgentSession | None = None
    pending_model_selection: ModelSelection | None = None
    last_model_selection: ModelSelection | None = None
    schedule_db_path: Path | None = None
    schedule_store_factory: Callable[[], ScheduleStore] | None = None
    background_adapter_factory: Callable[[], object] | None = None
