"""
haagent/runtime/settings/__init__.py - runtime 级用户设置

集中管理 HaAgent 运行时默认值，并读写用户级 settings.json 中的非模型配置。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from haagent.models.model_connections import user_settings_path
from haagent.runtime.execution.retry import RetryPolicy
from haagent.runtime.sandbox.settings import (
    SandboxSettings,
    SandboxSettingsError,
    load_sandbox_settings,
)

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
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    model_retry: RetryPolicy = field(default_factory=RetryPolicy)


def load_runtime_settings(*, config_path: Path | None = None) -> RuntimeSettings:
    path = config_path or user_settings_path()
    if not path.exists():
        return RuntimeSettings()
    raw = _read_settings_json(path)
    value = raw.get("interactive_max_turns", DEFAULT_INTERACTIVE_MAX_TURNS)
    try:
        sandbox = load_sandbox_settings(raw.get("sandbox"))
    except SandboxSettingsError as error:
        raise RuntimeSettingsError(str(error)) from error
    return RuntimeSettings(
        interactive_max_turns=_positive_int(value, "interactive_max_turns"),
        sandbox=sandbox,
        model_retry=_load_model_retry(raw.get("model_retry")),
    )


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


def _load_model_retry(raw: object) -> RetryPolicy:
    if raw is None:
        return RetryPolicy()
    if not isinstance(raw, dict):
        raise RuntimeSettingsError("model_retry must be an object")
    allowed_keys = {
        "max_attempts",
        "minimum_delay_seconds",
        "base_delay_seconds",
        "throttling_base_delay_seconds",
        "max_delay_seconds",
        "max_server_retry_after_seconds",
    }
    unknown_keys = set(raw) - allowed_keys
    if unknown_keys:
        raise RuntimeSettingsError("model_retry contains unknown fields")
    defaults = RetryPolicy()
    max_attempts = _retry_int(raw.get("max_attempts", defaults.max_attempts), "max_attempts")
    if not 1 <= max_attempts <= 5:
        raise RuntimeSettingsError("model_retry.max_attempts must be between 1 and 5")
    base_delay_seconds = _retry_seconds(
        raw.get("base_delay_seconds", defaults.base_delay_seconds),
        "base_delay_seconds",
    )
    minimum_delay_seconds = _retry_seconds(
        raw.get("minimum_delay_seconds", defaults.minimum_delay_seconds),
        "minimum_delay_seconds",
    )
    throttling_base_delay_seconds = _retry_seconds(
        raw.get("throttling_base_delay_seconds", defaults.throttling_base_delay_seconds),
        "throttling_base_delay_seconds",
    )
    max_delay_seconds = _retry_seconds(
        raw.get("max_delay_seconds", defaults.max_delay_seconds),
        "max_delay_seconds",
    )
    max_server_retry_after_seconds = _retry_seconds(
        raw.get(
            "max_server_retry_after_seconds",
            defaults.max_server_retry_after_seconds,
        ),
        "max_server_retry_after_seconds",
    )
    if max_delay_seconds < max(base_delay_seconds, throttling_base_delay_seconds):
        raise RuntimeSettingsError("model_retry.max_delay_seconds must cover base delays")
    if max_delay_seconds < minimum_delay_seconds:
        raise RuntimeSettingsError("model_retry.max_delay_seconds must cover minimum_delay_seconds")
    return RetryPolicy(
        max_attempts=max_attempts,
        minimum_delay_seconds=minimum_delay_seconds,
        base_delay_seconds=base_delay_seconds,
        throttling_base_delay_seconds=throttling_base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
        max_server_retry_after_seconds=max_server_retry_after_seconds,
    )


def _retry_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSettingsError(f"model_retry.{field_name} must be an integer")
    return value


def _retry_seconds(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeSettingsError(f"model_retry.{field_name} must be a positive number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeSettingsError(f"model_retry.{field_name} must be a positive number")
    return seconds
