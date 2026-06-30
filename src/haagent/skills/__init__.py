"""
haagent/skills/__init__.py - Skills 系统公开接口

导出本地 Markdown skills 的加载、注册和信任配置能力。
"""

from __future__ import annotations

from haagent.skills.loader import discover_project_skill_dirs, get_user_skill_dirs, load_skill_registry
from haagent.skills.registry import SkillRegistry
from haagent.skills.settings import (
    SkillSettings,
    SkillSettingsError,
    is_project_root_trusted,
    load_skill_settings,
    project_trust_root,
    save_skill_settings,
    trust_project_root,
    untrust_project_root,
)
from haagent.skills.types import SkillDefinition, SkillMetadata


__all__ = [
    "SkillDefinition",
    "SkillMetadata",
    "SkillRegistry",
    "SkillSettings",
    "SkillSettingsError",
    "discover_project_skill_dirs",
    "get_user_skill_dirs",
    "is_project_root_trusted",
    "load_skill_registry",
    "load_skill_settings",
    "project_trust_root",
    "save_skill_settings",
    "trust_project_root",
    "untrust_project_root",
]
