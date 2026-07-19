"""
src/haagent/tools/access.py - 工具访问统一控制

根据一次 run 的能力和权限上下文生成唯一工具访问快照；任务、模型 schema
和 Router 都必须使用这份快照，避免各入口分别维护工具可见性规则。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from haagent.models.capabilities import ModelCapabilities
from haagent.skills import load_skill_registry
from haagent.skills.catalog import SkillCatalogService
from haagent.skills.settings import load_skill_settings
from haagent.tools.registry import ToolRuntimeRegistry


@dataclass(frozen=True)
class ToolAccessSnapshot:
    """一次 run 的工具许可结果；模型和 Router 共享同一份结果。"""

    allowed_tools: tuple[str, ...]
    denied_tools: dict[str, str]


class ToolAccessManager:
    """统一计算工具候选集和运行时可用集。"""

    @staticmethod
    def candidate_tools(
        *,
        catalog: Any,
        enable_web: bool,
        has_skills: bool,
        image_attachment_history: bool,
        mcp_tool_names: Iterable[str],
    ) -> list[str]:
        names = list(catalog.chat_default_tools())
        if enable_web:
            names.extend(catalog.chat_web_tools())
        if has_skills:
            names.extend(catalog.chat_skill_tools())
        if image_attachment_history and catalog.has("load_image_attachment"):
            names.append("load_image_attachment")
        mcp_names = list(mcp_tool_names)
        if mcp_names:
            names.extend(mcp_names)
            if catalog.has("list_mcp_resources"):
                names.append("list_mcp_resources")
            if catalog.has("read_mcp_resource"):
                names.append("read_mcp_resource")
        return _unique(names)

    @staticmethod
    def skills_available(workspace_root: Path, catalog: SkillCatalogService | None) -> bool:
        return _skills_available(workspace_root, catalog)

    @classmethod
    def resolve(
        cls,
        requested_tools: Iterable[str],
        *,
        registry: ToolRuntimeRegistry,
        workspace_root: Path,
        mcp_runtime: Any | None,
        model_capabilities: ModelCapabilities | None,
        skill_catalog: SkillCatalogService | None,
        image_attachment_history: bool,
    ) -> ToolAccessSnapshot:
        allowed: list[str] = []
        denied: dict[str, str] = {}
        skills_available = _skills_available(workspace_root, skill_catalog)
        mcp_available = _mcp_available(mcp_runtime)
        vision_supported = model_capabilities is None or model_capabilities.vision != "unsupported"
        for name in _unique(requested_tools):
            reason = cls._denial_reason(
                name,
                registry=registry,
                mcp_available=mcp_available,
                skills_available=skills_available,
                image_attachment_history=image_attachment_history,
                vision_supported=vision_supported,
            )
            if reason is None:
                allowed.append(name)
            else:
                denied[name] = reason
        return ToolAccessSnapshot(tuple(allowed), denied)

    @staticmethod
    def _denial_reason(
        name: str,
        *,
        registry: ToolRuntimeRegistry,
        mcp_available: bool,
        skills_available: bool,
        image_attachment_history: bool,
        vision_supported: bool,
    ) -> str | None:
        if not registry.has(name):
            return "tool_not_registered"
        if name.startswith("mcp__") or name in {"list_mcp_resources", "read_mcp_resource"}:
            if not mcp_available:
                return "mcp_unavailable"
        if name in {"skill_list", "skill_read"} and not skills_available:
            return "skills_unavailable"
        if name == "load_image_attachment":
            if not image_attachment_history:
                return "image_attachment_unavailable"
            if not vision_supported:
                return "vision_unsupported"
        return None


def _unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(name) for name in names))


def _skills_available(workspace_root: Path, catalog: SkillCatalogService | None) -> bool:
    if catalog is not None:
        return bool(catalog.snapshot(workspace_root, load_skill_settings()).skills)
    return bool(load_skill_registry(workspace_root=workspace_root).list_skills())


def _mcp_available(runtime: Any | None) -> bool:
    if runtime is None:
        return False
    statuses = getattr(runtime, "list_statuses", None)
    if not callable(statuses):
        return True
    try:
        return any(getattr(item, "state", None) == "connected" for item in statuses())
    except Exception:
        return False
