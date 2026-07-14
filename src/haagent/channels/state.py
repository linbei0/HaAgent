"""
haagent/channels/state.py - 渠道 SQLite 本地状态

保存 binding、receipt、cursor、配对 hash 与 owner；禁止写入 secret 与消息正文。
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


class PairingError(RuntimeError):
    """配对码错误、过期或重复使用时显式失败。"""


@dataclass(frozen=True)
class ChannelBinding:
    workspace_root: str
    session_id: str


@dataclass(frozen=True)
class TemporaryChannelPermission:
    """渠道实例的限时权限；仅保存非敏感绑定与绝对过期时间。"""

    instance_id: str
    owner_sender_id: str
    workspace_root: str
    permission_mode: str
    expires_at: datetime


@dataclass(frozen=True)
class PermanentChannelPermission:
    """永久自动批准的身份与 workspace 绑定，不包含 secret。"""

    instance_id: str
    owner_sender_id: str
    workspace_root: str
    permission_mode: str


class ChannelStateStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS channel_bindings (
                binding_key TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                conversation_kind TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                thread_id TEXT,
                workspace_root TEXT NOT NULL,
                session_id TEXT NOT NULL,
                owner_sender_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS inbound_receipts (
                instance_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                received_at TEXT NOT NULL,
                PRIMARY KEY (instance_id, message_id)
            );
            CREATE TABLE IF NOT EXISTS transport_cursors (
                instance_id TEXT NOT NULL,
                cursor_name TEXT NOT NULL,
                cursor_value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (instance_id, cursor_name)
            );
            CREATE TABLE IF NOT EXISTS pairing_tokens (
                instance_id TEXT PRIMARY KEY,
                code_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_owners (
                instance_id TEXT PRIMARY KEY,
                owner_sender_id TEXT NOT NULL,
                paired_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_temporary_permissions (
                instance_id TEXT PRIMARY KEY,
                owner_sender_id TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                permission_mode TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_permanent_permission_bindings (
                instance_id TEXT PRIMARY KEY,
                owner_sender_id TEXT NOT NULL,
                workspace_root TEXT NOT NULL,
                permission_mode TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def upsert_binding(
        self,
        *,
        binding_key: str,
        instance_id: str,
        platform: str,
        conversation_kind: str,
        conversation_id: str,
        thread_id: str | None,
        workspace_root: str,
        session_id: str,
        owner_sender_id: str,
    ) -> None:
        now = _utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO channel_bindings (
                binding_key, instance_id, platform, conversation_kind, conversation_id,
                thread_id, workspace_root, session_id, owner_sender_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(binding_key) DO UPDATE SET
                workspace_root=excluded.workspace_root,
                session_id=excluded.session_id,
                owner_sender_id=excluded.owner_sender_id,
                updated_at=excluded.updated_at
            """,
            (
                binding_key,
                instance_id,
                platform,
                conversation_kind,
                conversation_id,
                thread_id,
                workspace_root,
                session_id,
                owner_sender_id,
                now,
            ),
        )
        self._conn.commit()

    def get_binding(self, binding_key: str) -> ChannelBinding | None:
        row = self._conn.execute(
            "SELECT workspace_root, session_id FROM channel_bindings WHERE binding_key = ?",
            (binding_key,),
        ).fetchone()
        if row is None:
            return None
        return ChannelBinding(
            workspace_root=row["workspace_root"],
            session_id=row["session_id"],
        )

    def delete_binding(self, binding_key: str) -> bool:
        """删除单条 binding；返回是否实际删除了行。禁止外部直访 _conn。"""
        cursor = self._conn.execute(
            "DELETE FROM channel_bindings WHERE binding_key = ?",
            (binding_key,),
        )
        self._conn.commit()
        return int(cursor.rowcount or 0) > 0

    def has_receipt(self, instance_id: str, message_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM inbound_receipts WHERE instance_id = ? AND message_id = ?",
            (instance_id, message_id),
        ).fetchone()
        return row is not None

    def commit_accepted_batch(
        self,
        *,
        instance_id: str,
        message_ids: list[str],
    ) -> None:
        # 仅在 Actor 接受后提交 receipt；cursor 由 transport 成功批次单独推进。
        now = _utc_now_iso()
        try:
            for message_id in message_ids:
                self._conn.execute(
                    "INSERT OR IGNORE INTO inbound_receipts (instance_id, message_id, received_at) VALUES (?, ?, ?)",
                    (instance_id, message_id, now),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def set_cursor(self, instance_id: str, cursor_name: str, cursor_value: str) -> None:
        now = _utc_now_iso()
        self._conn.execute(
            """
            INSERT INTO transport_cursors (instance_id, cursor_name, cursor_value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(instance_id, cursor_name) DO UPDATE SET
                cursor_value=excluded.cursor_value,
                updated_at=excluded.updated_at
            """,
            (instance_id, cursor_name, cursor_value, now),
        )
        self._conn.commit()

    def get_cursor(self, instance_id: str, cursor_name: str) -> str | None:
        row = self._conn.execute(
            "SELECT cursor_value FROM transport_cursors WHERE instance_id = ? AND cursor_name = ?",
            (instance_id, cursor_name),
        ).fetchone()
        return None if row is None else str(row["cursor_value"])

    def create_pairing_token(
        self,
        instance_id: str,
        code: str,
        *,
        expires_in_seconds: int = 600,
        expires_at: datetime | None = None,
    ) -> None:
        salt = secrets.token_hex(16)
        code_hash = _hash_code(code, salt)
        exp = expires_at or (datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds))
        self._conn.execute(
            """
            INSERT INTO pairing_tokens (instance_id, code_hash, salt, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(instance_id) DO UPDATE SET
                code_hash=excluded.code_hash,
                salt=excluded.salt,
                expires_at=excluded.expires_at
            """,
            (instance_id, code_hash, salt, exp.astimezone(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def consume_pairing_token(self, instance_id: str, code: str, *, sender_id: str = "") -> str:
        row = self._conn.execute(
            "SELECT code_hash, salt, expires_at FROM pairing_tokens WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        if row is None:
            raise PairingError("pairing token missing")
        expires_at = datetime.fromisoformat(str(row["expires_at"]))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at:
            self._conn.execute("DELETE FROM pairing_tokens WHERE instance_id = ?", (instance_id,))
            self._conn.commit()
            raise PairingError("pairing token expired")
        expected = str(row["code_hash"])
        actual = _hash_code(code, str(row["salt"]))
        if not secrets.compare_digest(expected, actual):
            raise PairingError("invalid pairing code")
        if not sender_id:
            raise PairingError("sender_id required")
        # 成功后立即删除 token，并固化 owner。
        self._conn.execute("DELETE FROM pairing_tokens WHERE instance_id = ?", (instance_id,))
        self.set_owner(instance_id, sender_id, commit=False)
        self._conn.commit()
        return sender_id

    def set_owner(self, instance_id: str, owner_sender_id: str, *, commit: bool = True) -> None:
        self._conn.execute(
            """
            INSERT INTO channel_owners (instance_id, owner_sender_id, paired_at)
            VALUES (?, ?, ?)
            ON CONFLICT(instance_id) DO UPDATE SET
                owner_sender_id=excluded.owner_sender_id,
                paired_at=excluded.paired_at
            """,
            (instance_id, owner_sender_id, _utc_now_iso()),
        )
        if commit:
            self._conn.commit()

    def get_owner(self, instance_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT owner_sender_id FROM channel_owners WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        return None if row is None else str(row["owner_sender_id"])

    def set_temporary_permission(
        self,
        *,
        instance_id: str,
        owner_sender_id: str,
        workspace_root: str,
        permission_mode: str,
        expires_at: datetime,
    ) -> None:
        if permission_mode != "auto_approve":
            raise ValueError("temporary channel permission must be auto_approve")
        expires = expires_at.astimezone(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO channel_temporary_permissions (
                instance_id, owner_sender_id, workspace_root, permission_mode, expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(instance_id) DO UPDATE SET
                owner_sender_id=excluded.owner_sender_id,
                workspace_root=excluded.workspace_root,
                permission_mode=excluded.permission_mode,
                expires_at=excluded.expires_at,
                updated_at=excluded.updated_at
            """,
            (instance_id, owner_sender_id, workspace_root, permission_mode, expires, _utc_now_iso()),
        )
        self._conn.commit()

    def get_temporary_permission(
        self,
        instance_id: str,
        *,
        owner_sender_id: str,
        workspace_root: str,
    ) -> TemporaryChannelPermission | None:
        row = self._conn.execute(
            "SELECT * FROM channel_temporary_permissions WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(str(row["expires_at"]))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        current = datetime.now(timezone.utc)
        if (
            str(row["owner_sender_id"]) != owner_sender_id
            or str(row["workspace_root"]) != workspace_root
            or str(row["permission_mode"]) != "auto_approve"
            or current >= expires_at
        ):
            self.clear_temporary_permission(instance_id)
            return None
        return TemporaryChannelPermission(
            instance_id=instance_id,
            owner_sender_id=str(row["owner_sender_id"]),
            workspace_root=str(row["workspace_root"]),
            permission_mode="auto_approve",
            expires_at=expires_at.astimezone(timezone.utc),
        )

    def clear_temporary_permission(self, instance_id: str) -> None:
        self._conn.execute(
            "DELETE FROM channel_temporary_permissions WHERE instance_id = ?",
            (instance_id,),
        )
        self._conn.commit()

    def clear_instance_permissions(self, instance_id: str) -> None:
        """清除实例的临时权限和永久身份绑定；模式配置由 channels.json 管理。"""
        self._conn.execute(
            "DELETE FROM channel_temporary_permissions WHERE instance_id = ?",
            (instance_id,),
        )
        self._conn.execute(
            "DELETE FROM channel_permanent_permission_bindings WHERE instance_id = ?",
            (instance_id,),
        )
        self._conn.commit()

    def set_permanent_permission_binding(
        self,
        *,
        instance_id: str,
        owner_sender_id: str,
        workspace_root: str,
        permission_mode: str,
    ) -> None:
        if permission_mode != "auto_approve":
            raise ValueError("permanent channel permission must be auto_approve")
        self._conn.execute(
            """
            INSERT INTO channel_permanent_permission_bindings (
                instance_id, owner_sender_id, workspace_root, permission_mode, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(instance_id) DO UPDATE SET
                owner_sender_id=excluded.owner_sender_id,
                workspace_root=excluded.workspace_root,
                permission_mode=excluded.permission_mode,
                updated_at=excluded.updated_at
            """,
            (instance_id, owner_sender_id, workspace_root, permission_mode, _utc_now_iso()),
        )
        self._conn.commit()

    def get_permanent_permission_binding(
        self,
        instance_id: str,
        *,
        owner_sender_id: str,
        workspace_root: str,
    ) -> PermanentChannelPermission | None:
        row = self._conn.execute(
            "SELECT * FROM channel_permanent_permission_bindings WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["owner_sender_id"]) != owner_sender_id
            or str(row["workspace_root"]) != workspace_root
            or str(row["permission_mode"]) != "auto_approve"
        ):
            self._conn.execute(
                "DELETE FROM channel_permanent_permission_bindings WHERE instance_id = ?",
                (instance_id,),
            )
            self._conn.commit()
            return None
        return PermanentChannelPermission(
            instance_id=instance_id,
            owner_sender_id=owner_sender_id,
            workspace_root=workspace_root,
            permission_mode="auto_approve",
        )

    def clear_permanent_permission_binding(self, instance_id: str) -> None:
        self._conn.execute(
            "DELETE FROM channel_permanent_permission_bindings WHERE instance_id = ?",
            (instance_id,),
        )
        self._conn.commit()

    def get_pairing_status(self, instance_id: str) -> dict[str, str | None]:
        """
        返回配对 token 状态摘要；永不包含明文码或 hash。

        state: none | pending | expired
        """
        row = self._conn.execute(
            "SELECT expires_at FROM pairing_tokens WHERE instance_id = ?",
            (instance_id,),
        ).fetchone()
        if row is None:
            return {"state": "none", "expires_at": None}
        expires_at = datetime.fromisoformat(str(row["expires_at"]))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        expires_iso = expires_at.astimezone(timezone.utc).isoformat()
        if datetime.now(timezone.utc) > expires_at:
            # 惰性清理过期 token，避免 status 长期显示 pending。
            self._conn.execute("DELETE FROM pairing_tokens WHERE instance_id = ?", (instance_id,))
            self._conn.commit()
            return {"state": "expired", "expires_at": expires_iso}
        return {"state": "pending", "expires_at": expires_iso}

    def instance_status_summary(self, instance_id: str) -> dict[str, str]:
        """CLI/status 用脱敏摘要：owner、cursor 有无、pairing 状态。"""
        owner = self.get_owner(instance_id) or "(unpaired)"
        cursor_value = self.get_cursor(instance_id, "get_updates_buf")
        pairing = self.get_pairing_status(instance_id)
        return {
            "owner": owner,
            "cursor": "set" if cursor_value else "empty",
            "pairing": str(pairing["state"] or "none"),
        }

    def delete_instance_state(self, instance_id: str) -> None:
        """删除动态状态；不删除 HaAgent session package。"""
        self.reset_instance_identity(instance_id)

    def reset_instance_identity(self, instance_id: str) -> None:
        """机器人身份变化时原子删除所有与旧身份绑定的动态状态。"""
        tables = (
            "channel_bindings",
            "inbound_receipts",
            "transport_cursors",
            "pairing_tokens",
            "channel_owners",
            "channel_temporary_permissions",
            "channel_permanent_permission_bindings",
        )
        try:
            for table in tables:
                self._conn.execute(f"DELETE FROM {table} WHERE instance_id = ?", (instance_id,))
            self._conn.commit()
        except Exception:
            # 身份边界必须全有或全无，禁止部分旧状态泄漏给新机器人。
            self._conn.rollback()
            raise

    def purge_old_receipts(self, *, older_than_days: int = 7) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM inbound_receipts WHERE received_at < ?",
            (cutoff,),
        )
        self._conn.commit()
        return int(cursor.rowcount or 0)


def _hash_code(code: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{code}".encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
