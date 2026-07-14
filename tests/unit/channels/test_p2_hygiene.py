"""
tests/unit/channels/test_p2_hygiene.py - P2 卫生合同

覆盖：stop 错误暴露、receipt 清理与媒体入站忽略。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from haagent.channels.adapters.weixin.adapter import WeixinAdapter
from haagent.channels.adapters.weixin.types import WeixinInboundMessage, WeixinUpdates
from haagent.channels.manager import ChannelManager
from haagent.channels.runtime import ChannelGatewayRuntime
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage
from tests.support.channel_adapter import FakeChannelAdapter as FakeAdapter


def test_delete_binding_public_api(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    store.upsert_binding(
        binding_key="weixin:wx-1:dm:u1",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="u1",
        thread_id=None,
        workspace_root=str(tmp_path),
        session_id="sess-1",
        owner_sender_id="u1",
    )
    assert store.get_binding("weixin:wx-1:dm:u1") is not None
    assert store.delete_binding("weixin:wx-1:dm:u1") is True
    assert store.get_binding("weixin:wx-1:dm:u1") is None
    assert store.delete_binding("weixin:wx-1:dm:u1") is False
    store.close()


def test_manager_stop_records_adapter_errors(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")

    class _BadAdapter:
        platform = "fake"
        instance_id = "bad-1"
        state = "connected"
        last_error = ""

        async def start(self, on_message):
            return None

        async def stop(self) -> None:
            raise RuntimeError("stop boom")

        async def send_text(self, **kwargs):
            from haagent.channels.types import SendResult

            return SendResult(ok=True)

        async def set_typing(self, **kwargs):
            return None

    manager = ChannelManager(
        state=store,
        default_workspace_root=tmp_path,
        service_factory=lambda root: None,
    )
    asyncio.run(manager.attach_adapter(_BadAdapter()))
    errors = asyncio.run(manager.stop())
    assert any("stop boom" in e for e in errors)
    store.close()


def test_manager_clears_recovered_adapter_error(tmp_path: Path) -> None:
    store = ChannelStateStore(tmp_path / "channels.sqlite3")

    class _Adapter:
        platform = "fake"
        instance_id = "recovering-1"
        state = "reconnecting"
        last_error = "temporary failure"

        async def start(self, on_message):
            return None

        async def stop(self) -> None:
            return None

    adapter = _Adapter()
    manager = ChannelManager(
        state=store,
        default_workspace_root=tmp_path,
        service_factory=lambda root: None,
    )
    asyncio.run(manager.attach_adapter(adapter))
    assert manager.status()[0]["last_error"] == "temporary failure"

    adapter.state = "connected"
    adapter.last_error = ""
    assert manager.status()[0]["last_error"] == ""

    asyncio.run(manager.stop())
    store.close()


def test_runtime_load_purges_old_receipts(tmp_path: Path) -> None:
    state_path = tmp_path / "channels.sqlite3"
    store = ChannelStateStore(state_path)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store._conn.execute(
        "INSERT INTO inbound_receipts (instance_id, message_id, received_at) VALUES (?, ?, ?)",
        ("wx-1", "old-msg", old),
    )
    store._conn.execute(
        "INSERT INTO inbound_receipts (instance_id, message_id, received_at) VALUES (?, ?, ?)",
        ("wx-1", "new-msg", datetime.now(timezone.utc).isoformat()),
    )
    store._conn.commit()
    store.close()

    cfg = tmp_path / "channels.json"
    cfg.write_text('{"version":1,"instances":[]}', encoding="utf-8")
    runtime = ChannelGatewayRuntime(
        config_path=cfg,
        state_path=state_path,
        default_workspace_root=tmp_path,
        service_factory=lambda root: None,
    )
    runtime.load()
    asyncio.run(runtime.stop())
    reloaded = ChannelStateStore(state_path)
    assert reloaded.has_receipt("wx-1", "old-msg") is False
    assert reloaded.has_receipt("wx-1", "new-msg") is True
    reloaded.close()


def test_weixin_media_message_ignored_advances_cursor(tmp_path: Path) -> None:
    """纯媒体入站：不进模型，但可推进 cursor 以免卡死。"""

    class _Proto:
        def __init__(self) -> None:
            self.calls = 0

        async def get_updates(self, *, cursor: str = ""):
            self.calls += 1
            if self.calls == 1:
                return WeixinUpdates(
                    messages=[
                        WeixinInboundMessage(
                            message_id="img-1",
                            from_user_id="u1",
                            text="",
                            context_token="tok",
                            raw={
                                "message_type": 2,
                                "item_list": [{"type": 2, "image": {"url": "x"}}],
                            },
                        )
                    ],
                    cursor="after-media",
                )
            return WeixinUpdates(messages=[], cursor="after-media")

        async def notify_start(self) -> None:
            return None

        async def notify_stop(self) -> None:
            return None

        async def aclose(self) -> None:
            return None

    seen: list[Any] = []

    async def _handler(msg: InboundChannelMessage) -> str:
        seen.append(msg)
        return "accepted"

    async def _run() -> None:
        persisted: list[str] = []
        adapter = WeixinAdapter(
            instance_id="wx-1",
            protocol=_Proto(),
            poll_interval=0.01,
            initial_cursor="",
            on_cursor_persist=persisted.append,
        )
        task = asyncio.create_task(adapter.start(_handler))
        await asyncio.sleep(0.08)
        await adapter.stop()
        await task
        assert seen == []
        assert persisted == ["after-media"]

    asyncio.run(_run())
