"""
tests/unit/models/test_model_connections.py - 模型连接用户设置文件测试

验证供应商连接、模型选择和凭据写入不会泄漏真实 API key。
"""

from __future__ import annotations

import json

import pytest

from tests.support.model_credentials import FakeCredentialStore
from haagent.models.model_connections import (
    ModelRoute,
    ModelSelection,
    ProviderConnectionRecord,
    ProviderProfileError,
    delete_provider_connection,
    load_active_model_selection,
    load_model_route,
    load_model_selection_profile,
    load_providers_config_snapshot,
    provider_connection_credential_status,
    save_active_model_selection,
    save_fallback_model_selection,
    save_provider_connection,
    save_provider_connection_with_key,
)


def _snapshot(config_dir):
    return load_providers_config_snapshot(config_dir / "providers.json")


def _save(connection, config_dir):
    return save_provider_connection(connection, snapshot=_snapshot(config_dir))


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
        snapshot=_snapshot(config_dir),
        credential_store=store,
    )
    save_provider_connection_with_key(
        work,
        "sk-work",
        snapshot=_snapshot(config_dir),
        credential_store=store,
    )
    snapshot = _snapshot(config_dir)

    personal_profile = load_model_selection_profile(
        ModelSelection(connection_id="requesty-personal", model="openai/gpt-5.2-chat"),
        snapshot=snapshot,
        credential_store=store,
        environ={},
    )
    work_profile = load_model_selection_profile(
        ModelSelection(connection_id="requesty-work", model="openai/gpt-5.2-chat"),
        snapshot=snapshot,
        credential_store=store,
        environ={},
    )
    providers_text = (config_dir / "providers.json").read_text(encoding="utf-8")

    assert personal_profile.api_key == "sk-personal"
    assert work_profile.api_key == "sk-work"
    assert store.values["connection:requesty-personal"] == "sk-personal"
    assert store.values["connection:requesty-work"] == "sk-work"
    assert "sk-personal" not in providers_text
    assert "sk-work" not in providers_text


def test_local_connection_without_credentials_loads_profile_and_writes_version_four(tmp_path) -> None:
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

    path = _save(connection, config_dir)
    snapshot = _snapshot(config_dir)
    profile = load_model_selection_profile(
        ModelSelection(connection_id="ollama-local", model="qwen3:8b"),
        snapshot=snapshot,
        environ={},
    )
    status = provider_connection_credential_status(
        connection,
        environ={},
        config_dir=config_dir,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == 4
    assert payload["connections"][0]["runtime_kind"] == "ollama"
    assert profile.api_key == ""
    assert profile.runtime_kind == "ollama"
    assert status.api_key_available is True
    assert status.credential_source_used == "none"


def test_v4_models_options_and_variants_resolve_on_profile(tmp_path) -> None:
    from haagent.models.model_options import ModelParameterConfig

    config_dir = tmp_path / ".haagent"
    connection = ProviderConnectionRecord(
        id="openai-main",
        name="OpenAI",
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        credential_source="env",
        models={
            "gpt-4.1": ModelParameterConfig(
                options={"temperature": 0.2},
                variants={"fast": {"temperature": 0.9}},
            ),
        },
    )
    _save(connection, config_dir)
    snapshot = _snapshot(config_dir)

    default_profile = load_model_selection_profile(
        ModelSelection(connection_id="openai-main", model="gpt-4.1"),
        snapshot=snapshot,
        environ={"OPENAI_API_KEY": "sk-test"},
    )
    fast_profile = load_model_selection_profile(
        ModelSelection(connection_id="openai-main", model="gpt-4.1", variant="fast"),
        snapshot=snapshot,
        environ={"OPENAI_API_KEY": "sk-test"},
    )

    assert default_profile.request_config.options == {"temperature": 0.2}
    assert fast_profile.variant == "fast"
    assert fast_profile.request_config.options == {"temperature": 0.9}


def test_save_connection_preserves_existing_models_when_record_empty(tmp_path) -> None:
    from haagent.models.model_options import ModelParameterConfig

    config_dir = tmp_path / ".haagent"
    with_models = ProviderConnectionRecord(
        id="openai-main",
        name="OpenAI",
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        credential_source="env",
        models={
            "gpt-4.1": ModelParameterConfig(options={"temperature": 0.3}, variants={}),
        },
    )
    _save(with_models, config_dir)
    # TUI 连接向导通常不带 models；更新不得静默丢弃已有 options。
    without_models = ProviderConnectionRecord(
        id="openai-main",
        name="OpenAI Renamed",
        provider_id="openai",
        provider_name="OpenAI",
        gateway_provider="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        credential_source="env",
    )
    path = _save(without_models, config_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 4
    assert payload["connections"][0]["name"] == "OpenAI Renamed"
    assert payload["connections"][0]["models"]["gpt-4.1"]["options"]["temperature"] == 0.3


def test_v4_rejects_unknown_structural_field(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"
    config_dir.mkdir()
    payload = {
        "version": 4,
        "connections": [
            {
                "id": "openai-main",
                "name": "OpenAI",
                "provider_id": "openai",
                "provider_name": "OpenAI",
                "gateway_provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "credential_source": "env",
                "runtime_kind": "remote",
                "models": {
                    "gpt-test": {
                        "options": {"temperature": 0.2},
                        "unexpected": True,
                    }
                },
            }
        ],
    }
    path = config_dir / "providers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProviderProfileError, match="unknown field"):
        load_providers_config_snapshot(path).require_valid()


def test_providers_snapshot_is_stable_after_file_changes(tmp_path) -> None:
    from haagent.models.model_options import ModelParameterConfig

    config_dir = tmp_path / ".haagent"
    path = _save(
        ProviderConnectionRecord(
            id="openai-main",
            name="OpenAI",
            provider_id="openai",
            provider_name="OpenAI",
            gateway_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            credential_source="env",
            models={
                "gpt-test": ModelParameterConfig(options={"temperature": 0.2}, variants={})
            },
        ),
        config_dir,
    )
    snapshot = load_providers_config_snapshot(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["connections"][0]["models"]["gpt-test"]["options"]["temperature"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_model_selection_profile(
        ModelSelection(connection_id="openai-main", model="gpt-test"),
        snapshot=snapshot,
        environ={"OPENAI_API_KEY": "test"},
    )

    assert profile.request_config.options == {"temperature": 0.2}


def test_snapshot_rejects_configured_model_missing_from_bound_catalog(tmp_path) -> None:
    from haagent.models.model_options import ModelParameterConfig

    config_dir = tmp_path / ".haagent"
    path = _save(
        ProviderConnectionRecord(
            id="openai-main",
            name="OpenAI",
            provider_id="openai",
            provider_name="OpenAI",
            gateway_provider="openai",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            credential_source="env",
            models={"not-in-catalog": ModelParameterConfig(options={"temperature": 0.2}, variants={})},
        ),
        config_dir,
    )
    snapshot = load_providers_config_snapshot(path).bind_available_models(
        {"openai-main": {"gpt-test"}}
    )

    with pytest.raises(ProviderProfileError, match="not available"):
        load_model_selection_profile(
            ModelSelection(connection_id="openai-main", model="not-in-catalog"),
            snapshot=snapshot,
            environ={"OPENAI_API_KEY": "test"},
        )


def test_native_catalog_gateways_can_be_persisted(tmp_path) -> None:
    config_dir = tmp_path / ".haagent"

    for gateway_provider in ("anthropic", "google"):
        connection = ProviderConnectionRecord(
            id=gateway_provider,
            name=gateway_provider,
            provider_id=gateway_provider,
            provider_name=gateway_provider.title(),
            gateway_provider=gateway_provider,
            base_url=f"https://{gateway_provider}.example/v1",
            api_key_env=f"{gateway_provider.upper()}_API_KEY",
        )

        _save(connection, config_dir)

        assert _snapshot(config_dir).connection(gateway_provider) == connection


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
    _save(primary, config_dir)
    _save(fallback, config_dir)
    save_active_model_selection(
        ModelSelection(connection_id="primary", model="qwen"),
        config_dir=config_dir,
    )
    save_fallback_model_selection(
        ModelSelection(connection_id="fallback", model="gemma"),
        cloud_fallback_consent=False,
        config_dir=config_dir,
    )

    delete_provider_connection("fallback", snapshot=_snapshot(config_dir))

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
