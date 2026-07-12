"""
haagent/app/assistant_service.py - 个人助手应用组合根

为 CLI 与 TUI 组合 workspace、session、model、skill、memory 与 schedules 应用 Module。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from haagent.app import (
    channel_usecases,
    memory_usecases,
    model_connection_usecases,
    schedule_usecases,
    session_usecases,
    skill_usecases,
    workspace_usecases,
)
from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import GatewayFactory
from haagent.models.gateway_registry import gateway_from_profile
from haagent.runtime.session.agent import AgentSession
from haagent.runtime.settings import DEFAULT_INTERACTIVE_MAX_TURNS

if TYPE_CHECKING:
    from haagent.scheduling.store import ScheduleStore


class AssistantService:
    """组合应用 Module；具体用例由各 Module 自己实现。"""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        runs_root: Path = Path(".runs"),
        environ: Mapping[str, str] | None = None,
        gateway_factory: GatewayFactory | None = None,
        session_cls: type[AgentSession] = AgentSession,
        max_turns: int | None = DEFAULT_INTERACTIVE_MAX_TURNS,
        enable_web: bool = False,
        initial_resume: str | Path | None = None,
        initial_continue: bool = False,
        schedule_db_path: Path | None = None,
        schedule_store_factory: Callable[[], ScheduleStore] | None = None,
        background_adapter_factory: Callable[[], object] | None = None,
    ) -> None:
        # 所有 Module 共享同一私有状态，避免配置与当前 session 在层间漂移。
        self._context = AssistantContext(
            workspace_root=(workspace_root or Path.cwd()).resolve(),
            runs_root=runs_root,
            environ=os.environ if environ is None else environ,
            gateway_factory=gateway_factory or gateway_from_profile,
            session_factory=session_cls,
            max_turns=max_turns,
            enable_web=enable_web,
            initial_resume=initial_resume,
            initial_continue=initial_continue,
            schedule_db_path=schedule_db_path,
            schedule_store_factory=schedule_store_factory,
            background_adapter_factory=background_adapter_factory,
        )
        self.workspace = workspace_usecases.AssistantWorkspace(self._context)
        self.sessions = session_usecases.AssistantSessions(self._context)
        self.models = model_connection_usecases.AssistantModels(self._context)
        self.skills = skill_usecases.AssistantSkills(self._context)
        self.memory = memory_usecases.AssistantMemory(self._context)
        self.channels = channel_usecases.AssistantChannels(self._context)
        self.schedules = schedule_usecases.AssistantSchedules(self._context)
