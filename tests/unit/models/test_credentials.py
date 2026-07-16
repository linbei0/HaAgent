"""
tests/unit/models/test_credentials.py - API key 凭据解析测试

验证环境变量、系统凭据库和显式明文 fallback 的读取优先级与泄漏边界。
"""

from __future__ import annotations

from pathlib import Path

from haagent.models.config.credentials import (
    CredentialRecord,
    resolve_api_key,
)
from tests.support.model_credentials import FakeCredentialStore


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


def test_fake_credential_store_delete_password() -> None:
    store = FakeCredentialStore({"channel:weixin:wx-1:bot_token": "secret-token"})
    assert store.get_password("haagent", "channel:weixin:wx-1:bot_token") == "secret-token"
    store.delete_password("haagent", "channel:weixin:wx-1:bot_token")
    assert store.get_password("haagent", "channel:weixin:wx-1:bot_token") is None
    # 重复删除不报错
    store.delete_password("haagent", "channel:weixin:wx-1:bot_token")
