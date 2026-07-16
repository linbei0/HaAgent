"""
tests/support/model_credentials.py - 模型凭据测试替身

提供不触碰系统 keyring 的内存 CredentialStore 实现。
"""

from __future__ import annotations

from typing import Mapping

from haagent.models.config.credentials import CredentialError


class FakeCredentialStore:
    """测试用内存凭据库。"""

    def __init__(
        self,
        values: Mapping[str, str] | None = None,
        *,
        available: bool = True,
        error: str = "credential store unavailable",
    ) -> None:
        self.values = dict(values or {})
        self.available = available
        self.error = error

    def get_password(self, service_name: str, username: str) -> str | None:
        if not self.available:
            raise CredentialError(self.error)
        return self.values.get(username)

    def set_password(self, service_name: str, username: str, password: str) -> None:
        if not self.available:
            raise CredentialError(self.error)
        self.values[username] = password

    def delete_password(self, service_name: str, username: str) -> None:
        if not self.available:
            raise CredentialError(self.error)
        self.values.pop(username, None)
