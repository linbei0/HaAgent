"""
haagent/runtime/settings/__init__.py - runtime 级用户设置

集中管理 HaAgent 运行时默认值，并读写用户级 settings.json 中的非模型配置。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from haagent.models.provider_profile import user_settings_path

DEFAULT_INTERACTIVE_MAX_TURNS = 200
DEFAULT_RUN_MAX_TURNS = 3
DEFAULT_SMOKE_MAX_TURNS = 12
DEFAULT_DOGFOOD_MAX_TURNS = 16
DEFAULT_CHECK_EVAL_MAX_TURNS = 5


class RuntimeSettingsError(ValueError):
    """runtime settings 损坏或字段非法时抛出。"""


@dataclass(frozen=True)
class RuntimeSettings:
    interactive_max_turns: int = DEFAULT_INTERACTIVE_MAX_TURNS


def load_runtime_settings(*, config_path: Path | None = None) -> RuntimeSettings:
    path = config_path or user_settings_path()
    if not path.exists():
        return RuntimeSettings()
    raw = _read_settings_json(path)
    value = raw.get("interactive_max_turns", DEFAULT_INTERACTIVE_MAX_TURNS)
    return RuntimeSettings(interactive_max_turns=_positive_int(value, "interactive_max_turns"))


def save_runtime_settings(
    settings: RuntimeSettings,
    *,
    config_path: Path | None = None,
) -> Path:
    path = config_path or user_settings_path()
    raw = _read_settings_json(path) if path.exists() else {}
    raw["interactive_max_turns"] = _positive_int(
        settings.interactive_max_turns,
        "interactive_max_turns",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_settings_json(path, raw)
    return path


def set_interactive_max_turns(
    max_turns: int,
    *,
    config_path: Path | None = None,
) -> RuntimeSettings:
    settings = RuntimeSettings(interactive_max_turns=_positive_int(max_turns, "interactive_max_turns"))
    save_runtime_settings(settings, config_path=config_path)
    return settings


def _read_settings_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeSettingsError(f"settings config is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise RuntimeSettingsError("settings config must be a JSON object")
    return raw


def _write_settings_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSettingsError(f"{field_name} must be a positive integer")
    if value <= 0:
        raise RuntimeSettingsError(f"{field_name} must be a positive integer")
    return value
