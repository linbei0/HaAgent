"""
tests/unit/channels/test_state.py - 渠道 SQLite 状态与配对测试

验证 binding、receipt、cursor、pairing 与 secret 不进入数据库。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from haagent.channels.state import ChannelStateStore, PairingError


@pytest.fixture
def store(tmp_path: Path) -> ChannelStateStore:
    return ChannelStateStore(tmp_path / "channels.sqlite3")


def test_binding_create_read_update(store: ChannelStateStore) -> None:
    store.upsert_binding(
        binding_key="weixin:wx-1:dm:user-a",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="user-a",
        thread_id=None,
        workspace_root="/ws",
        session_id="sess-1",
        owner_sender_id="user-a",
    )
    binding = store.get_binding("weixin:wx-1:dm:user-a")
    assert binding is not None
    assert binding.session_id == "sess-1"
    store.upsert_binding(
        binding_key="weixin:wx-1:dm:user-a",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="user-a",
        thread_id=None,
        workspace_root="/ws",
        session_id="sess-2",
        owner_sender_id="user-a",
    )
    assert store.get_binding("weixin:wx-1:dm:user-a").session_id == "sess-2"


def test_receipt_claim_is_atomic_dedup(store: ChannelStateStore) -> None:
    assert store.has_receipt("wx-1", "msg-1") is False
    first = store.claim_receipt("wx-1", "msg-1")
    second = store.claim_receipt("wx-1", "msg-1")
    assert first is True
    assert second is False
    assert store.has_receipt("wx-1", "msg-1") is True


def test_cursor_and_batch_commit_transactional(store: ChannelStateStore) -> None:
    # 只有 Actor 接受后才同事务写 receipt + cursor。
    store.commit_accepted_batch(
        instance_id="wx-1",
        message_ids=["m-1", "m-2"],
        cursor_name="get_updates_buf",
        cursor_value="cursor-2",
    )
    assert store.get_cursor("wx-1", "get_updates_buf") == "cursor-2"
    assert store.claim_receipt("wx-1", "m-1") is False
    assert store.claim_receipt("wx-1", "m-2") is False


def test_pairing_hash_only_and_owner_flow(store: ChannelStateStore) -> None:
    store.create_pairing_token("wx-1", "ABCD1234", expires_in_seconds=600)
    # 配对码原文不得落盘。
    raw_db = Path(store.path).read_bytes()
    assert b"ABCD1234" not in raw_db

    with pytest.raises(PairingError):
        store.consume_pairing_token("wx-1", "WRONG000")

    owner = store.consume_pairing_token("wx-1", "ABCD1234", sender_id="owner-1")
    assert owner == "owner-1"
    assert store.get_owner("wx-1") == "owner-1"

    # 成功后 token 删除，重复使用失败。
    with pytest.raises(PairingError):
        store.consume_pairing_token("wx-1", "ABCD1234", sender_id="other")


def test_expired_pairing_fails(store: ChannelStateStore) -> None:
    store.create_pairing_token(
        "wx-1",
        "EXPIRE01",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(PairingError, match="expired"):
        store.consume_pairing_token("wx-1", "EXPIRE01", sender_id="u1")


def test_sqlite_has_no_secret_tokens(store: ChannelStateStore, tmp_path: Path) -> None:
    store.upsert_binding(
        binding_key="weixin:wx-1:dm:user-a",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="user-a",
        thread_id=None,
        workspace_root=str(tmp_path),
        session_id="sess-1",
        owner_sender_id="user-a",
    )
    store.commit_accepted_batch(
        instance_id="wx-1",
        message_ids=["m-1"],
        cursor_name="get_updates_buf",
        cursor_value="buf-1",
    )
    store.set_owner("wx-1", "user-a")
    raw = Path(store.path).read_bytes()
    assert b"test-bot-token-secret" not in raw
    assert b"context-token-secret" not in raw


def test_temporary_permission_requires_matching_owner_workspace_and_unexpired_time(
    store: ChannelStateStore,
) -> None:
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    store.set_temporary_permission(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root="C:/work",
        permission_mode="auto_approve",
        expires_at=now + timedelta(minutes=30),
    )

    active = store.get_temporary_permission(
        "wx-1",
        owner_sender_id="owner-1",
        workspace_root="C:/work",
        now=now,
    )
    assert active is not None
    assert active.permission_mode == "auto_approve"
    assert (
        store.get_temporary_permission(
            "wx-1",
            owner_sender_id="other-owner",
            workspace_root="C:/work",
            now=now,
        )
        is None
    )
    assert (
        store.get_temporary_permission(
            "wx-1",
            owner_sender_id="owner-1",
            workspace_root="C:/other-work",
            now=now,
        )
        is None
    )

    store.set_temporary_permission(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root="C:/work",
        permission_mode="auto_approve",
        expires_at=now + timedelta(minutes=1),
    )
    assert (
        store.get_temporary_permission(
            "wx-1",
            owner_sender_id="owner-1",
            workspace_root="C:/work",
            now=now + timedelta(minutes=2),
        )
        is None
    )


def test_reset_instance_identity_removes_all_dynamic_state(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    store.set_owner("wx-1", "owner-1")
    store.set_cursor("wx-1", "get_updates_buf", "cursor-1")
    assert store.claim_receipt("wx-1", "message-1") is True
    store.upsert_binding(
        binding_key="weixin:wx-1:dm:owner-1",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="owner-1",
        thread_id=None,
        workspace_root=str(tmp_path),
        session_id="session-1",
        owner_sender_id="owner-1",
    )
    store.create_pairing_token("wx-1", "PAIRCODE", expires_at=expires_at)
    store.set_temporary_permission(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root=str(tmp_path),
        permission_mode="auto_approve",
        expires_at=expires_at,
    )
    store.set_permanent_permission_binding(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root=str(tmp_path),
        permission_mode="auto_approve",
    )

    store.reset_instance_identity("wx-1")

    assert store.get_owner("wx-1") is None
    assert store.get_cursor("wx-1", "get_updates_buf") is None
    assert store.has_receipt("wx-1", "message-1") is False
    assert store.get_binding("weixin:wx-1:dm:owner-1") is None
    assert store.get_pairing_status("wx-1")["state"] == "none"
    assert store.get_temporary_permission(
        "wx-1", owner_sender_id="owner-1", workspace_root=str(tmp_path)
    ) is None
    assert store.get_permanent_permission_binding(
        "wx-1", owner_sender_id="owner-1", workspace_root=str(tmp_path)
    ) is None
    store.close()
