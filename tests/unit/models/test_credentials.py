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


def test_insecure_file_rejects_empty_key(tmp_path: Path) -> None:
    with pytest.raises(CredentialError, match="API key is required"):
        save_insecure_api_key("local", "", config_dir=tmp_path)
