"""
tests/unit/models/test_model_connections.py - 模型连接用户设置文件测试

验证供应商连接、模型选择和凭据写入不会泄漏真实 API key。
"""

from __future__ import annotations

import json

from haagent.models.credentials import FakeCredentialStore
from haagent.models.model_connections import (
    ModelSelection,
    ProviderConnectionRecord,
    load_active_model_selection,
    load_model_selection_profile,
    save_active_model_selection,
    save_provider_connection_with_key,
)


def test_provider_connections_allow_two_keys_for_same_provider(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    store = FakeCredentialStore()
    personal = ProviderConnectionRecord(
        id="requesty-personal",
        name="personal",
        provider_id="requesty",
        provider_name="Requesty",
        gateway_provider="openai-chat",
        base_url="https://router.requesty.ai/v1",
        api_key_env="REQUESTY_API_KEY",
    )
    work = ProviderConnectionRecord(
        id="requesty-work",
        name="work",
        provider_id="requesty",
        provider_name="Requesty",
        gateway_provider="openai-chat",
        base_url="https://router.requesty.ai/v1",
        api_key_env="REQUESTY_WORK_API_KEY",
    )

    save_provider_connection_with_key(
        personal,
        "sk-personal",
        credential_store=store,
        config_dir=config_dir,
    )
    save_provider_connection_with_key(
        work,
        "sk-work",
        credential_store=store,
        config_dir=config_dir,
    )

    personal_profile = load_model_selection_profile(
        ModelSelection(connection_id="requesty-personal", model="openai/gpt-5.2-chat"),
        credential_store=store,
        environ={},
        config_dir=config_dir,
    )
    work_profile = load_model_selection_profile(
        ModelSelection(connection_id="requesty-work", model="openai/gpt-5.2-chat"),
        credential_store=store,
        environ={},
        config_dir=config_dir,
    )
    providers_text = (config_dir / "providers.json").read_text(encoding="utf-8")

    assert personal_profile.api_key == "sk-personal"
    assert work_profile.api_key == "sk-work"
    assert store.values["connection:requesty-personal"] == "sk-personal"
    assert store.values["connection:requesty-work"] == "sk-work"
    assert "sk-personal" not in providers_text
    assert "sk-work" not in providers_text


def test_save_active_model_selection_preserves_runtime_settings(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"interactive_max_turns": 80}, ensure_ascii=False),
        encoding="utf-8",
    )

    save_active_model_selection(
        ModelSelection(connection_id="requesty-personal", model="openai/gpt-5.2-chat"),
        config_dir=config_dir,
    )

    selection = load_active_model_selection(config_dir=config_dir)
    saved = json.loads(settings_path.read_text(encoding="utf-8"))

    assert selection == ModelSelection(connection_id="requesty-personal", model="openai/gpt-5.2-chat")
    assert saved == {
        "interactive_max_turns": 80,
        "active_model": {
            "connection_id": "requesty-personal",
            "model": "openai/gpt-5.2-chat",
        },
    }
