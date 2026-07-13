"""
tests/unit/runtime/test_runtime_settings.py - runtime 设置读写测试

验证交互式 turn 默认值从用户 settings.json 读取，并且写入时保留其它设置。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.settings import (
    DEFAULT_INTERACTIVE_MAX_TURNS,
    DEFAULT_PROGRESS_GUARD_MODE,
    RuntimeSettingsError,
    load_runtime_settings,
    set_interactive_max_turns,
)


def test_missing_runtime_settings_uses_interactive_default(tmp_path) -> None:
    settings = load_runtime_settings(config_path=tmp_path / "missing.json")

    assert settings.interactive_max_turns == DEFAULT_INTERACTIVE_MAX_TURNS
    assert settings.progress_guard_mode == DEFAULT_PROGRESS_GUARD_MODE


def test_runtime_settings_reads_progress_guard_mode(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"progress_guard_mode": "block"}, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path=settings_path)

    assert settings.progress_guard_mode == "block"


@pytest.mark.parametrize("raw_value", ["quiet", 1, True, "WARN", ""])
def test_runtime_settings_rejects_invalid_progress_guard_mode(tmp_path, raw_value) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"progress_guard_mode": raw_value}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeSettingsError, match="progress_guard_mode"):
        load_runtime_settings(config_path=settings_path)


def test_setting_interactive_max_turns_preserves_active_model(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    active_model = {"connection_id": "local", "model": "deepseek-chat"}
    settings_path.write_text(
        json.dumps({"active_model": active_model}, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = set_interactive_max_turns(80, config_path=settings_path)

    assert settings.interactive_max_turns == 80
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved == {"active_model": active_model, "interactive_max_turns": 80}


@pytest.mark.parametrize("raw_value", [0, -1, "80", "0", "abc", None])
def test_runtime_settings_rejects_non_positive_interactive_max_turns(tmp_path, raw_value) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"interactive_max_turns": raw_value}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeSettingsError):
        load_runtime_settings(config_path=settings_path)


def test_runtime_settings_reads_model_retry(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "model_retry": {
                    "max_attempts": 2,
                    "base_delay_seconds": 0.2,
                    "throttling_base_delay_seconds": 1,
                    "max_delay_seconds": 4,
                    "max_server_retry_after_seconds": 30,
                }
            }
        ),
        encoding="utf-8",
    )

    settings = load_runtime_settings(config_path=settings_path)

    assert settings.model_retry.max_attempts == 2
    assert settings.model_retry.max_delay_seconds == 4


@pytest.mark.parametrize(
    "raw_value",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"base_delay_seconds": 0},
        {"max_delay_seconds": 0.1, "base_delay_seconds": 1},
    ],
)
def test_runtime_settings_rejects_invalid_model_retry(tmp_path, raw_value) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"model_retry": raw_value}), encoding="utf-8")

    with pytest.raises(RuntimeSettingsError):
        load_runtime_settings(config_path=settings_path)
