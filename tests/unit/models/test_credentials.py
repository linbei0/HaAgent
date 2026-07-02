"""
tests/unit/models/test_credentials.py - API key 凭据解析测试

验证环境变量、系统凭据库和显式明文 fallback 的读取优先级与泄漏边界。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from haagent.models.credentials import (
    CredentialError,
    CredentialRecord,
    FakeCredentialStore,
    resolve_api_key,
    save_insecure_api_key,
)
from haagent.models.provider_profile import (
    ProviderProfileError,
    ProviderProfileRecord,
    delete_provider_profile,
    load_provider_profile,
    load_active_profile_name,
    load_provider_profile_record,
    save_active_profile,
    save_provider_profile,
    save_provider_profile_with_key,
)


def test_env_api_key_has_highest_priority(tmp_path: Path) -> None:
    store = FakeCredentialStore({"profile:local": "keyring-secret"})
    record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
    )

    resolved = resolve_api_key(
        record,
        environ={"OPENAI_API_KEY": "env-secret"},
        credential_store=store,
        config_dir=tmp_path,
    )

    assert resolved.api_key == "env-secret"
    assert resolved.credential_source_used == "env"
    assert resolved.api_key_available is True
    assert "env-secret" not in repr(resolved)


def test_keyring_used_when_env_missing(tmp_path: Path) -> None:
    store = FakeCredentialStore({"profile:local": "keyring-secret"})
    record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
    )

    resolved = resolve_api_key(record, environ={}, credential_store=store, config_dir=tmp_path)

    assert resolved.api_key == "keyring-secret"
    assert resolved.credential_source_used == "keyring"
    assert resolved.credential_store_available is True


def test_env_overrides_keyring(tmp_path: Path) -> None:
    store = FakeCredentialStore({"profile:local": "keyring-secret"})
    record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
    )

    status = resolve_api_key(
        record,
        environ={"OPENAI_API_KEY": "env-secret"},
        credential_store=store,
        config_dir=tmp_path,
    )

    assert status.api_key == "env-secret"
    assert status.credential_source_used == "env"


def test_keyring_unavailable_reports_error_without_insecure_fallback(tmp_path: Path) -> None:
    store = FakeCredentialStore(available=False, error="backend unavailable")
    record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
    )

    status = resolve_api_key(record, environ={}, credential_store=store, config_dir=tmp_path)

    assert status.api_key is None
    assert status.api_key_available is False
    assert status.credential_source_used is None
    assert status.credential_store_available is False
    assert status.credential_store_error == "backend unavailable"
    assert not (tmp_path / "insecure_credentials.json").exists()


def test_insecure_file_requires_explicit_source(tmp_path: Path) -> None:
    save_insecure_api_key("local", "plain-secret", config_dir=tmp_path)
    keyring_record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="keyring",
    )
    insecure_record = CredentialRecord(
        profile_name="local",
        api_key_env="OPENAI_API_KEY",
        credential_source="insecure_file",
    )

    keyring_status = resolve_api_key(
        keyring_record,
        environ={},
        credential_store=FakeCredentialStore({}),
        config_dir=tmp_path,
    )
    insecure_status = resolve_api_key(
        insecure_record,
        environ={},
        credential_store=FakeCredentialStore({}),
        config_dir=tmp_path,
    )

    assert keyring_status.api_key is None
    assert insecure_status.api_key == "plain-secret"
    assert insecure_status.credential_source_used == "insecure_file"
    assert "plain-secret" not in repr(insecure_status)


def test_save_provider_profile_with_keyring_key_does_not_write_secret(tmp_path: Path) -> None:
    store = FakeCredentialStore()
    record = ProviderProfileRecord(
        name="router",
        provider="openai-chat",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="keyring",
    )

    save_provider_profile_with_key(
        record,
        "sk-test-secret",
        credential_store=store,
        config_dir=tmp_path,
    )

    saved = load_provider_profile_record("router", config_path=tmp_path / "providers.json")
    loaded = load_provider_profile(
        "router",
        config_path=tmp_path / "providers.json",
        credential_store=store,
        environ={},
        config_dir=tmp_path,
    )
    assert saved.name == "router"
    assert loaded.api_key == "sk-test-secret"
    assert store.values["profile:router"] == "sk-test-secret"
    assert "sk-test-secret" not in (tmp_path / "providers.json").read_text(encoding="utf-8")


def test_save_provider_profile_with_keyring_none_key_saves_profile_without_keyring_write(tmp_path: Path) -> None:
    store = FakeCredentialStore()
    record = ProviderProfileRecord(
        name="router",
        provider="openai-chat",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="keyring",
    )

    save_provider_profile_with_key(
        record,
        None,
        credential_store=store,
        config_dir=tmp_path,
    )

    saved = load_provider_profile_record("router", config_path=tmp_path / "providers.json")
    assert saved.name == "router"
    assert store.values == {}


def test_save_provider_profile_with_keyring_blank_key_saves_profile_without_keyring_write(tmp_path: Path) -> None:
    store = FakeCredentialStore()
    record = ProviderProfileRecord(
        name="router",
        provider="openai-chat",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="keyring",
    )

    save_provider_profile_with_key(
        record,
        "   ",
        credential_store=store,
        config_dir=tmp_path,
    )

    saved = load_provider_profile_record("router", config_path=tmp_path / "providers.json")
    assert saved.name == "router"
    assert store.values == {}


def test_save_provider_profile_with_env_none_key_saves_profile(tmp_path: Path) -> None:
    record = ProviderProfileRecord(
        name="router",
        provider="openai-chat",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="env",
    )

    save_provider_profile_with_key(record, None, config_dir=tmp_path)

    saved = load_provider_profile_record("router", config_path=tmp_path / "providers.json")
    assert saved.credential_source == "env"


def test_save_provider_profile_with_key_rejects_env_secret_without_writing_file(tmp_path: Path) -> None:
    record = ProviderProfileRecord(
        name="router",
        provider="openai-chat",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="env",
    )

    with pytest.raises(ProviderProfileError, match="env credential_source does not allow saving api_key"):
        save_provider_profile_with_key(record, "sk-test-secret", config_dir=tmp_path)

    assert not (tmp_path / "providers.json").exists()


def test_save_provider_profile_with_invalid_record_rejects_without_writing_state(tmp_path: Path) -> None:
    store = FakeCredentialStore()
    record = ProviderProfileRecord(
        name="router",
        provider="unsupported",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.2-chat",
        api_key_env="OPENROUTER_API_KEY",
        credential_source="keyring",
    )

    with pytest.raises(ProviderProfileError, match="unsupported provider"):
        save_provider_profile_with_key(
            record,
            "sk-test-secret",
            credential_store=store,
            config_dir=tmp_path,
        )

    assert not (tmp_path / "providers.json").exists()
    assert store.values == {}


def test_delete_provider_profile_removes_record_and_repoints_active_profile(tmp_path: Path) -> None:
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=tmp_path,
    )
    save_provider_profile(
        ProviderProfileRecord(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
        ),
        config_dir=tmp_path,
    )
    save_active_profile("local", config_dir=tmp_path)

    delete_provider_profile("local", config_dir=tmp_path)

    with pytest.raises(ProviderProfileError, match="provider profile not found: local"):
        load_provider_profile_record("local", config_path=tmp_path / "providers.json")
    assert load_provider_profile_record("router", config_path=tmp_path / "providers.json").name == "router"
    assert load_active_profile_name(settings_path=tmp_path / "settings.json") == "router"


def test_delete_last_provider_profile_clears_active_settings(tmp_path: Path) -> None:
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=tmp_path,
    )
    save_active_profile("local", config_dir=tmp_path)

    delete_provider_profile("local", config_dir=tmp_path)

    assert (tmp_path / "providers.json").read_text(encoding="utf-8") == '{\n  "profiles": []\n}\n'
    assert not (tmp_path / "settings.json").exists()


def test_delete_missing_provider_profile_fails_without_changing_file(tmp_path: Path) -> None:
    save_provider_profile(
        ProviderProfileRecord(
            name="local",
            provider="openai-chat",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        config_dir=tmp_path,
    )
    before = (tmp_path / "providers.json").read_text(encoding="utf-8")

    with pytest.raises(ProviderProfileError, match="provider profile not found: missing"):
        delete_provider_profile("missing", config_dir=tmp_path)

    assert (tmp_path / "providers.json").read_text(encoding="utf-8") == before


def test_insecure_file_rejects_empty_key(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="API key is required"):
        save_insecure_api_key("local", "", config_dir=tmp_path)
