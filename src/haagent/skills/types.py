"""
haagent/skills/types.py - Skill 数据模型

定义 HaAgent 本地 Markdown skills 的稳定运行时结构。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillDefinition:
    """一个已加载的本地 skill。"""

    name: str
    description: str
    content: str
    source: str
    path: str | None = None
    base_dir: str | None = None
    command_name: str | None = None
    display_name: str | None = None
    aliases: tuple[str, ...] = ()
    user_invocable: bool = True
    disable_model_invocation: bool = False


@dataclass(frozen=True)
class SkillMetadata:
    """从 SKILL.md frontmatter 或正文推导出的元数据。"""

    name: str
    description: str
    aliases: tuple[str, ...] = ()
    user_invocable: bool = True
    disable_model_invocation: bool = False
