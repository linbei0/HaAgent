"""
tests/unit/models/test_provider_profile.py - provider profile 用户设置文件测试

验证模型配置读写不会覆盖 runtime settings 中的其它顶层字段。
"""

from __future__ import annotations

import json

from haagent.models.provider_profile import (
    ProviderProfileRecord,
    delete_provider_profile,
    save_active_profile,
    save_provider_profile,
)


def test_save_active_profile_preserves_runtime_settings(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"interactive_max_turns": 80}, ensure_ascii=False),
        encoding="utf-8",
    )

    save_active_profile("local", config_dir=config_dir)

    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved == {"interactive_max_turns": 80, "active_profile": "local"}


def test_delete_active_profile_preserves_runtime_settings(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://example.test",
            model="model",
            api_key_env="API_KEY",
        ),
        config_dir=config_dir,
    )
    settings_path = config_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"active_profile": "local", "interactive_max_turns": 80}, ensure_ascii=False),
        encoding="utf-8",
    )

    delete_provider_profile("local", config_dir=config_dir)

    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved == {"interactive_max_turns": 80}
