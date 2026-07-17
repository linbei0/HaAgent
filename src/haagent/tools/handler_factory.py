"""
haagent/tools/handler_factory.py - 静态工具 handler 组合

委托 ToolCatalog 绑定静态 handler；策略、审批和审计仍由 router 负责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.path_policy import PathPolicy
from haagent.runtime.sandbox.base import SandboxBackend
from haagent.skills import SkillSettings
from haagent.skills.catalog import SkillCatalogService
from haagent.tools.base import ToolHandler
from haagent.tools.catalog import ToolRuntimeDeps, default_tool_catalog


def build_static_tool_handlers(
    *,
    workspace_root: Path,
    path_policy: PathPolicy,
    skill_settings: SkillSettings | None,
    cancellation_token: CancellationToken | None,
    mcp_runtime: Any | None,
    sandbox_backend: SandboxBackend | None,
    router_handlers: dict[str, ToolHandler],
    skill_catalog: SkillCatalogService | None = None,
) -> dict[str, ToolHandler]:
    """组合静态工具 handler；策略、审批和审计仍由 router 负责。"""
    deps = ToolRuntimeDeps(
        workspace_root=workspace_root,
        path_policy=path_policy,
        skill_settings=skill_settings,
        cancellation_token=cancellation_token,
        mcp_runtime=mcp_runtime,
        sandbox_backend=sandbox_backend,
        skill_catalog=skill_catalog,
        router_handlers=router_handlers,
    )
    return default_tool_catalog().build_static_handlers(deps)
