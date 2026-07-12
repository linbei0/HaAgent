"""
tests/unit/channels/test_p2_hygiene.py - P2 卫生合同

覆盖：公开 delete_binding、stop 不吞错误、receipt 清理、媒体入站忽略。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from haagent.channels.adapters.weixin.adapter import WeixinAdapter
from haagent.channels.adapters.weixin.media import WeixinMediaNotImplemented, download_media
from haagent.channels.adapters.weixin.types import WeixinInboundMessage, WeixinUpdates
from haagent.channels.manager import ChannelManager
from haagent.channels.runtime import ChannelGatewayRuntime
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage


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


def test_manager_new_uses_public_delete_binding(tmp_path: Path) -> None:
    """/new 不得直接访问 state._conn，须走 delete_binding。"""
    store = ChannelStateStore(tmp_path / "channels.sqlite3")
    deleted: list[str] = []
    original = store.delete_binding

    def _track(key: str) -> bool:
        deleted.append(key)
        return original(key)

    store.delete_binding = _track  # type: ignore[method-assign]
    store.upsert_binding(
        binding_key="fake:f1:dm:owner",
        instance_id="f1",
        platform="fake",
        conversation_kind="dm",
        conversation_id="owner",
        thread_id=None,
        workspace_root=str(tmp_path),
        session_id="old-sess",
        owner_sender_id="owner",
    )
    store.set_owner("f1", "owner")

    class _Svc:
        def sessions(self):
            return self

        def create(self, **kwargs):
            return SimpleNamespace(session_id="new-sess")

        def resume(self, session_id: str):
            raise AssertionError("should create after /new")

        def run_prompt_events(self, prompt: str):
            yield from ()

        def cancel_current_run(self) -> bool:
            return False

        def set_permission_mode(self, mode: str) -> None:
            return None

        def set_human_interaction_handler(self, handler) -> None:
            return None

    from types import SimpleNamespace

    from haagent.channels.adapters.fake import FakeAdapter

    manager = ChannelManager(
        state=store,
        default_workspace_root=tmp_path,
        service_factory=lambda root: _Svc(),
    )
    adapter = FakeAdapter(instance_id="f1")
    asyncio.run(manager.attach_adapter(adapter))

    address = ChannelAddress(
        instance_id="f1",
        platform="fake",
        conversation_kind="dm",
        conversation_id="owner",
    )
    handle = ChannelReplyHandle(platform="fake", payload={})
    msg = InboundChannelMessage(
        address=address,
        message_id="new-1",
        sender_id="owner",
        text="/new",
        received_at=datetime.now(timezone.utc),
        reply_handle=handle,
    )
    outcome = asyncio.run(manager._on_message(msg))
    assert outcome in {"control", "accepted", "queued"}
    assert deleted == [address.binding_key()]
    asyncio.run(manager.stop())
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
    asyncio.run(manager.sync_adapter_states())
    assert manager.status()[0]["last_error"] == "temporary failure"

    adapter.state = "connected"
    adapter.last_error = ""
    asyncio.run(manager.sync_adapter_states())
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
    assert runtime.state is not None
    assert runtime.state.has_receipt("wx-1", "old-msg") is False
    assert runtime.state.has_receipt("wx-1", "new-msg") is True
    asyncio.run(runtime.stop())


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

        async def aclose(self) -> None:
            return None

    seen: list[Any] = []

    async def _handler(msg: InboundChannelMessage) -> str:
        seen.append(msg)
        return "accepted"

    async def _run() -> None:
        adapter = WeixinAdapter(
            instance_id="wx-1",
            protocol=_Proto(),
            poll_interval=0.01,
            initial_cursor="",
        )
        task = asyncio.create_task(adapter.start(_handler))
        await asyncio.sleep(0.08)
        await adapter.stop()
        await task
        assert seen == []
        assert adapter.cursor == "after-media"

    asyncio.run(_run())


def test_download_media_explicitly_unimplemented() -> None:
    with pytest.raises(WeixinMediaNotImplemented):
        download_media("any")


def test_channels_weixin_optional_deps_document_media_phase() -> None:
    """文本阶段 qrcode；媒体阶段 cryptography 单独说明，不强制安装。"""
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "channels-weixin" in text
    assert "qrcode" in text
    # 媒体阶段依赖须在注释或独立 extra 中可见，避免 silent 漂移。
    assert "cryptography" in text or "channels-weixin-media" in text
