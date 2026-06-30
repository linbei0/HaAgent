"""
haagent/skills/registry.py - Skill 注册表

按名称、目录命令名和 alias 管理已发现的 skills。
"""

from __future__ import annotations

from haagent.skills.types import SkillDefinition


class SkillRegistry:
    """存储已加载 skills，并保持覆盖顺序可预测。"""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def register(self, skill: SkillDefinition) -> None:
        for key in (skill.name, skill.command_name, skill.display_name, *skill.aliases):
            if key:
                self._skills[key] = skill
                self._skills[key.lower()] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name) or self._skills.get(name.lower())

    def list_skills(self) -> list[SkillDefinition]:
        unique: dict[tuple[str, str | None], SkillDefinition] = {}
        for skill in self._skills.values():
            unique[(skill.source, skill.path or skill.name)] = skill
        return sorted(unique.values(), key=lambda skill: skill.command_name or skill.name)
