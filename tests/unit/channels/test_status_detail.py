"""
tests/unit/channels/test_status_detail.py - 渠道状态详情与配对查询

验证 pairing 状态（无明文）、owner、cursor 摘要可供 gateway status 使用。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from haagent.channels.state import ChannelStateStore


def test_pairing_status_none_when_missing(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    try:
        status = store.get_pairing_status("wx-1")
        assert status["state"] == "none"
        assert status.get("expires_at") is None
        assert "code" not in status
        assert "hash" not in status
    finally:
        store.close()


def test_pairing_status_pending_and_expired(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    try:
        store.create_pairing_token("wx-1", "ABCD1234", expires_in_seconds=600)
        pending = store.get_pairing_status("wx-1")
        assert pending["state"] == "pending"
        assert pending.get("expires_at")
        # 不得暴露明文码
        assert "ABCD1234" not in str(pending)
        assert "code" not in pending

        store.create_pairing_token(
            "wx-1",
            "EXPIRE01",
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        )
        expired = store.get_pairing_status("wx-1")
        assert expired["state"] == "expired"
    finally:
        store.close()


def test_instance_status_summary_for_cli(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    try:
        store.set_owner("wx-1", "owner-sender")
        store.set_cursor("wx-1", "get_updates_buf", "cursor-value-secret-ish")
        store.create_pairing_token("wx-1", "PAIRCODE", expires_in_seconds=120)
        summary = store.instance_status_summary("wx-1")
        assert summary["owner"] == "owner-sender"
        assert summary["cursor"] == "set"
        assert summary["pairing"] == "pending"
        assert "PAIRCODE" not in str(summary)
        assert "cursor-value" not in str(summary)
    finally:
        store.close()
