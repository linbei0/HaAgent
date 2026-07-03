"""
tests/unit/runtime/test_runtime_settings.py - runtime 设置读写测试

验证交互式 turn 默认值从用户 settings.json 读取，并且写入时保留其它设置。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.settings import (
    DEFAULT_INTERACTIVE_MAX_TURNS,
    RuntimeSettingsError,
    load_runtime_settings,
    set_interactive_max_turns,
)


def test_missing_runtime_settings_uses_interactive_default(tmp_path) -> None:
    settings = load_runtime_settings(config_path=tmp_path / "missing.json")

    assert settings.interactive_max_turns == DEFAULT_INTERACTIVE_MAX_TURNS


def test_setting_interactive_max_turns_preserves_active_profile(tmp_path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"active_profile": "local"}, ensure_ascii=False),
        encoding="utf-8",
    )

    settings = set_interactive_max_turns(80, config_path=settings_path)

    assert settings.interactive_max_turns == 80
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved == {"active_profile": "local", "interactive_max_turns": 80}


@pytest.mark.parametrize("raw_value", [0, -1, "80", "0", "abc", None])
def test_runtime_settings_rejects_non_positive_interactive_max_turns(tmp_path, raw_value) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"interactive_max_turns": raw_value}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeSettingsError):
        load_runtime_settings(config_path=settings_path)
