"""
haagent/tools/skills.py - Skills 只读工具

提供本地 skills 的元数据列表和按需正文读取，所有调用经 ToolRouter 审计。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.skills import SkillSettings, discover_project_skill_dirs, is_project_root_trusted, load_skill_registry
from haagent.skills.catalog import SkillCatalogService
from haagent.skills.settings import load_skill_settings
from haagent.tools.base import tool_error


def skill_list(
    args: dict[str, Any],
    workspace_root: Path,
    skill_settings: SkillSettings | None = None,
    *,
    skill_catalog: SkillCatalogService | None = None,
) -> dict[str, Any]:
    query = str(args.get("query", "")).strip().lower()
    source_filter = str(args.get("source", "")).strip()
    max_results = _max_results(args.get("max_results"))
    settings = skill_settings or load_skill_settings()
    if skill_catalog is not None:
        skills_iter = skill_catalog.snapshot(workspace_root, settings).skills
    else:
        skills_iter = load_skill_registry(workspace_root=workspace_root, settings=settings).list_skills()
    skills = []
    for skill in skills_iter:
        if source_filter and skill.source != source_filter:
            continue
        haystack = f"{skill.name}\n{skill.description}\n{skill.command_name or ''}".lower()
        if query and query not in haystack:
            continue
        skills.append(
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
                "command_name": skill.command_name or skill.name,
                "user_invocable": skill.user_invocable,
                "disable_model_invocation": skill.disable_model_invocation,
            },
        )
        if len(skills) >= max_results:
            break
    return {
        "status": "success",
        "skills": skills,
        "blocked_project_skill_roots": _blocked_project_skill_roots(workspace_root, settings),
    }


def skill_read(
    args: dict[str, Any],
    workspace_root: Path,
    skill_settings: SkillSettings | None = None,
    *,
    skill_catalog: SkillCatalogService | None = None,
    user_invoked: bool = False,
) -> dict[str, Any]:
    name = str(args.get("name", "")).strip()
    if not name:
        return tool_error("skill_name_required", "skill name is required")
    settings = skill_settings or load_skill_settings()
    if skill_catalog is not None:
        registry = skill_catalog.snapshot(workspace_root, settings).as_registry()
    else:
        registry = load_skill_registry(workspace_root=workspace_root, settings=settings)
    skill = registry.get(name)
    if skill is None:
        return tool_error("skill_not_found", f"skill not found: {name}")
    if skill.disable_model_invocation and not user_invoked:
        command_name = skill.command_name or skill.name
        return tool_error(
            "skill_model_invocation_disabled",
            f"skill can only be invoked explicitly by the user: /{command_name}",
        )
    return {
        "status": "success",
        "name": skill.name,
        "description": skill.description,
        "source": skill.source,
        "command_name": skill.command_name or skill.name,
        "content": skill.content,
        "content_chars": len(skill.content),
    }


def _max_results(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return max(1, min(value, 50))
    return 20


def _blocked_project_skill_roots(workspace_root: Path, settings: SkillSettings) -> list[str]:
    if is_project_root_trusted(workspace_root, settings):
        return []
    return [str(path) for path in discover_project_skill_dirs(workspace_root)]
