"""
haagent/skills/loader.py - 本地 Markdown Skills 加载器

发现用户级和受信任项目级 SKILL.md，并解析紧凑元数据。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import yaml

from haagent.skills.registry import SkillRegistry
from haagent.skills.settings import SkillSettings, is_project_root_trusted, load_skill_settings, user_config_dir
from haagent.skills.types import SkillDefinition, SkillMetadata


logger = logging.getLogger(__name__)

DEFAULT_PROJECT_SKILL_DIRS = (".haagent/skills", ".agents/skills", ".claude/skills")
USER_COMPAT_SKILL_DIRS = ((".agents", "skills"), (".claude", "skills"))
BUILTIN_SKILL_DIR = Path(__file__).with_name("builtin")
BUILTIN_SKILL_PREFIX = "haagent-"


def load_skill_registry(
    *,
    workspace_root: Path | None = None,
    config_dir: Path | None = None,
    user_skill_dirs: Iterable[str | Path] | None = None,
    settings: SkillSettings | None = None,
) -> SkillRegistry:
    """加载内置、用户级和受信任项目级 skills。"""
    registry = SkillRegistry()
    for skill in load_skills_from_dirs(
        [BUILTIN_SKILL_DIR],
        source="builtin",
        create_missing=False,
    ):
        registry.register(skill)
    for skill in load_skills_from_dirs(
        get_user_skill_dirs(config_dir=config_dir) if user_skill_dirs is None else user_skill_dirs,
        source="user",
        create_missing=user_skill_dirs is None,
    ):
        _register_non_builtin_skill(registry, skill)

    if workspace_root is not None:
        resolved_settings = settings or load_skill_settings(config_dir=config_dir)
        if is_project_root_trusted(workspace_root, resolved_settings):
            for skill in load_skills_from_dirs(
                discover_project_skill_dirs(workspace_root),
                source="project",
                create_missing=False,
            ):
                _register_non_builtin_skill(registry, skill)
    return registry


def _register_non_builtin_skill(registry: SkillRegistry, skill: SkillDefinition) -> None:
    """保留 HaAgent 内置 skill 名称，避免外部内容静默覆盖产品规则。"""

    if skill.name.lower().startswith(BUILTIN_SKILL_PREFIX):
        logger.warning("Ignoring external skill reserved for HaAgent: %s", skill.name)
        return
    registry.register(skill)


def get_user_skill_dirs(*, config_dir: Path | None = None) -> list[Path]:
    config = config_dir or user_config_dir()
    return [config / "skills", *(Path.home().joinpath(*parts) for parts in USER_COMPAT_SKILL_DIRS)]


def discover_project_skill_dirs(
    workspace_root: str | Path,
    project_skill_dirs: Iterable[str] | None = None,
) -> list[Path]:
    """从 workspace 向上到 git root 发现项目 skills 目录。"""
    start = Path(workspace_root).expanduser().resolve()
    if start.is_file():
        start = start.parent
    relative_dirs = _valid_project_skill_dirs(project_skill_dirs or DEFAULT_PROJECT_SKILL_DIRS)
    git_root = _find_git_root(start)
    current = start
    levels: list[Path] = []
    while True:
        levels.append(current)
        if git_root is not None and current == git_root:
            break
        if git_root is None and current == Path.home().resolve():
            break
        parent = current.parent
        if parent == current:
            break
        current = parent

    roots: list[Path] = []
    seen: set[Path] = set()
    for base in reversed(levels):
        for rel in relative_dirs:
            candidate = (base / rel).resolve()
            if candidate in seen or not candidate.is_dir():
                continue
            seen.add(candidate)
            roots.append(candidate)
    return roots


def load_skills_from_dirs(
    directories: Iterable[str | Path] | None,
    *,
    source: str,
    create_missing: bool = True,
) -> list[SkillDefinition]:
    skills: list[SkillDefinition] = []
    if not directories:
        return skills
    seen: set[Path] = set()
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        if create_missing:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_path = child / "SKILL.md"
            if not skill_path.exists() or skill_path in seen:
                continue
            seen.add(skill_path)
            content = skill_path.read_text(encoding="utf-8")
            metadata = parse_skill_metadata(child.name, content)
            skills.append(
                SkillDefinition(
                    name=metadata.name,
                    description=metadata.description,
                    content=content,
                    source=source,
                    path=str(skill_path),
                    base_dir=str(child),
                    command_name=child.name,
                    display_name=metadata.name if metadata.name != child.name else None,
                    aliases=metadata.aliases,
                    user_invocable=metadata.user_invocable,
                    disable_model_invocation=metadata.disable_model_invocation,
                ),
            )
    return skills


def parse_skill_metadata(default_name: str, content: str) -> SkillMetadata:
    frontmatter, body = _split_frontmatter(content)
    data = _parse_frontmatter(frontmatter)
    name = _optional_str(data.get("name")) if data else None
    description = _optional_str(data.get("description")) if data else None
    aliases = _str_tuple(data.get("aliases")) if data else ()
    if not name:
        name = _heading_name(body) or default_name
    if not description:
        description = _first_body_paragraph(body) or f"Skill: {name}"
    return SkillMetadata(
        name=name,
        description=description,
        aliases=aliases,
        user_invocable=_bool_value(data.get("user-invocable"), default=True) if data else True,
        disable_model_invocation=_bool_value(data.get("disable-model-invocation"), default=False) if data else False,
    )


def _split_frontmatter(content: str) -> tuple[str | None, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, content
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return None, content


def _parse_frontmatter(frontmatter: str | None) -> dict[str, object]:
    if not frontmatter:
        return {}
    try:
        raw = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        logger.debug("Ignoring malformed skill frontmatter: %s", error)
        return {}
    return raw if isinstance(raw, dict) else {}


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _bool_value(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def _heading_name(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or None
    return None


def _first_body_paragraph(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped
    return None


def _valid_project_skill_dirs(project_skill_dirs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in project_skill_dirs:
        rel = Path(str(raw).strip())
        if rel.is_absolute() or ".." in rel.parts:
            logger.warning("Ignoring unsafe project skill dir: %s", raw)
            continue
        paths.append(rel)
    return paths


def _find_git_root(start: Path) -> Path | None:
    current = start
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent
