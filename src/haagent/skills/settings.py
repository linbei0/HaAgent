"""
haagent/skills/settings.py - Skills 用户级信任配置

读取和写入项目 skills 的显式信任边界。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


USER_CONFIG_DIR_NAME = ".haagent"
SKILL_SETTINGS_FILE = "skills.json"


class SkillSettingsError(RuntimeError):
    """Skills 配置损坏或不可写时抛出。"""


@dataclass(frozen=True)
class SkillSettings:
    version: int = 1
    trusted_project_roots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "trusted_project_roots": list(self.trusted_project_roots),
        }


def user_config_dir() -> Path:
    return Path.home() / USER_CONFIG_DIR_NAME


def skill_settings_path(config_dir: Path | None = None) -> Path:
    return (config_dir or user_config_dir()) / SKILL_SETTINGS_FILE


def load_skill_settings(*, config_dir: Path | None = None) -> SkillSettings:
    path = skill_settings_path(config_dir)
    if not path.exists():
        return SkillSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SkillSettingsError(f"skills config is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise SkillSettingsError("skills config must be a JSON object")
    version = raw.get("version", 1)
    if version != 1:
        raise SkillSettingsError(f"unsupported skills config version: {version}")
    roots = raw.get("trusted_project_roots", [])
    if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
        raise SkillSettingsError("trusted_project_roots must be a list of strings")
    normalized = tuple(_normalize_root(root) for root in roots if root.strip())
    return SkillSettings(version=1, trusted_project_roots=normalized)


def save_skill_settings(settings: SkillSettings, *, config_dir: Path | None = None) -> Path:
    path = skill_settings_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = SkillSettings(
        version=1,
        trusted_project_roots=tuple(_normalize_root(root) for root in settings.trusted_project_roots),
    )
    path.write_text(json.dumps(normalized.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def trust_project_root(root: Path, *, config_dir: Path | None = None) -> SkillSettings:
    settings = load_skill_settings(config_dir=config_dir)
    normalized = _normalize_root(str(project_trust_root(root)))
    roots = list(settings.trusted_project_roots)
    if normalized not in roots:
        roots.append(normalized)
    next_settings = SkillSettings(version=1, trusted_project_roots=tuple(roots))
    save_skill_settings(next_settings, config_dir=config_dir)
    return next_settings


def untrust_project_root(root: Path, *, config_dir: Path | None = None) -> SkillSettings:
    settings = load_skill_settings(config_dir=config_dir)
    normalized = _normalize_root(str(project_trust_root(root)))
    next_settings = SkillSettings(
        version=1,
        trusted_project_roots=tuple(item for item in settings.trusted_project_roots if item != normalized),
    )
    save_skill_settings(next_settings, config_dir=config_dir)
    return next_settings


def project_trust_root(root: Path) -> Path:
    current = root.expanduser().resolve()
    if current.is_file():
        current = current.parent
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return root.expanduser().resolve()
        current = parent


def is_project_root_trusted(root: Path, settings: SkillSettings) -> bool:
    trust_root = project_trust_root(root).resolve()
    return _normalize_root(str(trust_root)) in settings.trusted_project_roots


def _normalize_root(root: str) -> str:
    return str(Path(root).expanduser().resolve())
