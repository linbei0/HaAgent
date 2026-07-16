"""
haagent/app/assistant_context.py - 应用 Module 私有共享状态

保存 workspace、session 与模型选择状态，不进入 CLI 或 TUI 的公开 Interface。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from haagent.app.assistant_types import GatewayFactory
from haagent.context.instruction_cache import InstructionCache
from haagent.models.model_ref import ModelRef
from haagent.models.model_runtime import ModelRuntime
from haagent.runtime.session.agent import AgentSession
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.schema_cache import ToolSchemaCache


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
    pending_model_selection: ModelRef | None = None
    model_runtime: ModelRuntime | None = None
    # workspace.status 缓存世代；session/模型/权限/凭据变化时 +1。
    status_generation: int = 0
    # 交互延迟优化：跨 session/turn 共享的只读缓存服务。
    skill_catalog: SkillCatalogService = field(default_factory=SkillCatalogService)
    instruction_cache: InstructionCache = field(default_factory=InstructionCache)
    tool_schema_cache: ToolSchemaCache = field(default_factory=ToolSchemaCache)
