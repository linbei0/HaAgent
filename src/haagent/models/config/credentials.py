"""
haagent/models/config/credentials.py - 用户级 API key 凭据解析

封装环境变量、系统凭据库和显式明文 fallback 的读取与状态检查。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol


KEYRING_SERVICE_NAME = "haagent"
INSECURE_CREDENTIALS_FILE = "insecure_credentials.json"
SUPPORTED_CREDENTIAL_SOURCES = {"env", "keyring", "insecure_file"}


class CredentialError(RuntimeError):
    """凭据读取或写入失败时抛出。"""


class CredentialStore(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None:
        """从系统凭据库读取密码。"""

    def set_password(self, service_name: str, username: str, password: str) -> None:
        """写入系统凭据库。"""

    def delete_password(self, service_name: str, username: str) -> None:
        """删除系统凭据库中的密码；不存在时视为成功。"""


@dataclass(frozen=True)
class CredentialRecord:
    profile_name: str
    api_key_env: str
    credential_source: str = "keyring"
    credential_username: str | None = None


@dataclass(frozen=True)
class CredentialStatus:
    api_key_available: bool
    credential_source_configured: str
    credential_source_used: str | None
    credential_store_available: bool | None
    credential_store_error: str | None = None


@dataclass(frozen=True)
class ResolvedCredential(CredentialStatus):
    api_key: str | None = field(default=None, repr=False)


class KeyringCredentialStore:
    """对 keyring 依赖做一层薄封装，便于测试替换。"""

    def get_password(self, service_name: str, username: str) -> str | None:
        try:
            import keyring
        except Exception as error:  # pragma: no cover - 依赖缺失由集成测试覆盖
            raise CredentialError(str(error)) from error
        try:
            return keyring.get_password(service_name, username)
        except Exception as error:
            raise CredentialError(str(error)) from error

    def set_password(self, service_name: str, username: str, password: str) -> None:
        try:
            import keyring
        except Exception as error:  # pragma: no cover - 依赖缺失由集成测试覆盖
            raise CredentialError(str(error)) from error
        try:
            keyring.set_password(service_name, username, password)
        except Exception as error:
            raise CredentialError(str(error)) from error

    def delete_password(self, service_name: str, username: str) -> None:
        # 删除渠道实例等场景需要清理 keyring；条目不存在时静默成功。
        try:
            import keyring
            from keyring.errors import PasswordDeleteError
        except Exception as error:  # pragma: no cover
            raise CredentialError(str(error)) from error
        try:
            keyring.delete_password(service_name, username)
        except PasswordDeleteError:
            return
        except Exception as error:
            raise CredentialError(str(error)) from error


def resolve_api_key(
    record: CredentialRecord,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> ResolvedCredential:
    """按 env > configured source 的顺序解析 API key。"""
    _validate_credential_source(record.credential_source)
    environment = os.environ if environ is None else environ
    env_value = environment.get(record.api_key_env)
    if env_value:
        return ResolvedCredential(
            api_key_available=True,
            credential_source_configured=record.credential_source,
            credential_source_used="env",
            credential_store_available=True,
            api_key=env_value,
        )
    if record.credential_source == "env":
        return _missing(record, credential_store_available=None)
    if record.credential_source == "keyring":
        return _resolve_keyring(record, credential_store=credential_store)
    return _resolve_insecure_file(record, config_dir=config_dir)


def credential_status(
    record: CredentialRecord,
    *,
    environ: Mapping[str, str] | None = None,
    credential_store: CredentialStore | None = None,
    config_dir: Path | None = None,
) -> CredentialStatus:
    resolved = resolve_api_key(
        record,
        environ=environ,
        credential_store=credential_store,
        config_dir=config_dir,
    )
    return CredentialStatus(
        api_key_available=resolved.api_key_available,
        credential_source_configured=resolved.credential_source_configured,
        credential_source_used=resolved.credential_source_used,
        credential_store_available=resolved.credential_store_available,
        credential_store_error=resolved.credential_store_error,
    )


def save_connection_keyring_api_key(
    connection_id: str,
    api_key: str,
    *,
    credential_store: CredentialStore | None = None,
) -> None:
    if not api_key.strip():
        raise CredentialError("API key is required")
    store = credential_store or KeyringCredentialStore()
    store.set_password(KEYRING_SERVICE_NAME, _connection_credential_username(connection_id), api_key)


def save_connection_insecure_api_key(connection_id: str, api_key: str, *, config_dir: Path) -> Path:
    if not api_key.strip():
        raise CredentialError("API key is required")
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / INSECURE_CREDENTIALS_FILE
    data = _read_insecure_credentials(path) if path.exists() else {}
    data[_connection_credential_username(connection_id)] = api_key
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _resolve_keyring(
    record: CredentialRecord,
    *,
    credential_store: CredentialStore | None,
) -> ResolvedCredential:
    store = credential_store or KeyringCredentialStore()
    try:
        api_key = store.get_password(KEYRING_SERVICE_NAME, _record_credential_username(record))
    except CredentialError as error:
        return ResolvedCredential(
            api_key_available=False,
            credential_source_configured=record.credential_source,
            credential_source_used=None,
            credential_store_available=False,
            credential_store_error=str(error),
            api_key=None,
        )
    if not api_key:
        return _missing(record, credential_store_available=True)
    return ResolvedCredential(
        api_key_available=True,
        credential_source_configured=record.credential_source,
        credential_source_used="keyring",
        credential_store_available=True,
        api_key=api_key,
    )


def _resolve_insecure_file(
    record: CredentialRecord,
    *,
    config_dir: Path | None,
) -> ResolvedCredential:
    if config_dir is None:
        raise CredentialError("config_dir is required for insecure_file credentials")
    path = config_dir / INSECURE_CREDENTIALS_FILE
    data = _read_insecure_credentials(path) if path.exists() else {}
    api_key = data.get(_record_credential_username(record))
    if not api_key:
        return _missing(record, credential_store_available=None)
    return ResolvedCredential(
        api_key_available=True,
        credential_source_configured=record.credential_source,
        credential_source_used="insecure_file",
        credential_store_available=None,
        api_key=api_key,
    )


def _missing(
    record: CredentialRecord,
    *,
    credential_store_available: bool | None,
) -> ResolvedCredential:
    return ResolvedCredential(
        api_key_available=False,
        credential_source_configured=record.credential_source,
        credential_source_used=None,
        credential_store_available=credential_store_available,
        api_key=None,
    )


def _read_insecure_credentials(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CredentialError(f"insecure credential file is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise CredentialError("insecure credential file must be a JSON object")
    values: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            values[key] = value
    return values


def _credential_username(profile_name: str) -> str:
    return f"profile:{profile_name}"


def _connection_credential_username(connection_id: str) -> str:
    return f"connection:{connection_id}"


def _record_credential_username(record: CredentialRecord) -> str:
    return record.credential_username or _credential_username(record.profile_name)


def _validate_credential_source(source: str) -> None:
    if source not in SUPPORTED_CREDENTIAL_SOURCES:
        raise CredentialError(f"unsupported credential_source: {source}")
