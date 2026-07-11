"""
tests/unit/models/test_model_connections.py - 模型连接用户设置文件测试

验证供应商连接、模型选择和凭据写入不会泄漏真实 API key。
"""

from __future__ import annotations

import json

from haagent.models.credentials import FakeCredentialStore
from haagent.models.model_connections import (
    ModelRoute,
    ModelSelection,
    ProviderConnectionRecord,
    delete_provider_connection,
    load_active_model_selection,
    load_model_route,
    load_model_selection_profile,
    load_provider_connection_record,
    provider_connection_credential_status,
    save_active_model_selection,
    save_fallback_model_selection,
    save_provider_connection,
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


def test_version_two_connection_defaults_to_remote_runtime(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    config_dir.mkdir(parents=True)
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "legacy",
                        "name": "Legacy",
                        "provider_id": "openai",
                        "provider_name": "OpenAI",
                        "gateway_provider": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "OPENAI_API_KEY",
                        "credential_source": "env",
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )

    connection = load_provider_connection_record(
        "legacy",
        config_path=config_dir / "providers.json",
    )

    assert connection.runtime_kind == "remote"


def test_local_connection_without_credentials_loads_profile_and_writes_version_three(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    connection = ProviderConnectionRecord(
        id="ollama-local",
        name="Ollama",
        provider_id="ollama",
        provider_name="Ollama",
        gateway_provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",
        credential_source="none",
        runtime_kind="ollama",
    )

    path = save_provider_connection(connection, config_dir=config_dir)
    profile = load_model_selection_profile(
        ModelSelection(connection_id="ollama-local", model="qwen3:8b"),
        environ={},
        config_dir=config_dir,
    )
    status = provider_connection_credential_status(
        "ollama-local",
        environ={},
        config_dir=config_dir,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == 3
    assert payload["connections"][0]["runtime_kind"] == "ollama"
    assert profile.api_key == ""
    assert profile.runtime_kind == "ollama"
    assert status.api_key_available is True
    assert status.credential_source_used == "none"


def test_model_route_loads_single_fallback_and_cloud_consent(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    primary = ModelSelection(connection_id="ollama-local", model="qwen3:8b")
    fallback = ModelSelection(connection_id="openai-cloud", model="gpt-5.2")

    save_active_model_selection(primary, config_dir=config_dir)
    save_fallback_model_selection(
        fallback,
        cloud_fallback_consent=True,
        config_dir=config_dir,
    )

    assert load_model_route(config_dir=config_dir) == ModelRoute(
        primary=primary,
        fallback=fallback,
        cloud_fallback_consent=True,
    )


def test_deleting_fallback_connection_clears_fallback_settings(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    primary = ProviderConnectionRecord(
        id="primary",
        name="Primary",
        provider_id="ollama",
        provider_name="Ollama",
        gateway_provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env="",
        credential_source="none",
        runtime_kind="ollama",
    )
    fallback = ProviderConnectionRecord(
        id="fallback",
        name="Fallback",
        provider_id="lm-studio",
        provider_name="LM Studio",
        gateway_provider="openai",
        base_url="http://127.0.0.1:1234/v1",
        api_key_env="",
        credential_source="none",
        runtime_kind="lm_studio",
    )
    save_provider_connection(primary, config_dir=config_dir)
    save_provider_connection(fallback, config_dir=config_dir)
    save_active_model_selection(
        ModelSelection(connection_id="primary", model="qwen"),
        config_dir=config_dir,
    )
    save_fallback_model_selection(
        ModelSelection(connection_id="fallback", model="gemma"),
        cloud_fallback_consent=False,
        config_dir=config_dir,
    )

    delete_provider_connection("fallback", config_dir=config_dir)

    assert load_model_route(config_dir=config_dir) == ModelRoute(
        primary=ModelSelection(connection_id="primary", model="qwen"),
        fallback=None,
        cloud_fallback_consent=False,
    )


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
