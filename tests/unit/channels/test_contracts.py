"""
tests/unit/channels/test_contracts.py - 渠道公共合同与 FakeAdapter 测试

验证 ChannelAddress、ChannelReplyHandle、capabilities 与 FakeAdapter 行为。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from haagent.channels.adapters.fake import FakeAdapter
from haagent.channels.types import (
    ChannelAddress,
    ChannelCapabilities,
    ChannelReplyHandle,
    InboundChannelMessage,
    UnsupportedCapabilityError,
)


def test_channel_address_binding_key_stable_and_distinct() -> None:
    dm = ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="user-a",
    )
    group = ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="group",
        conversation_id="g-1",
    )
    thread = ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="channel",
        conversation_id="c-1",
        thread_id="t-9",
    )
    assert dm.binding_key() == "weixin:wx-1:dm:user-a"
    assert group.binding_key() == "weixin:wx-1:group:g-1"
    assert thread.binding_key() == "weixin:wx-1:channel:c-1:t-9"
    assert len({dm.binding_key(), group.binding_key(), thread.binding_key()}) == 3


def test_channel_reply_handle_repr_hides_payload() -> None:
    handle = ChannelReplyHandle(platform="weixin", payload={"context_token": "secret-token-xyz"})
    text = repr(handle)
    assert "secret-token-xyz" not in text
    assert "context_token" not in text
    assert "ChannelReplyHandle" in text


def test_fake_adapter_start_receive_send_and_typing() -> None:
    async def run() -> None:
        adapter = FakeAdapter(instance_id="fake-1")
        received: list[InboundChannelMessage] = []

        async def on_message(message: InboundChannelMessage) -> None:
            received.append(message)

        await adapter.start(on_message)
        address = ChannelAddress(
            instance_id="fake-1",
            platform="fake",
            conversation_kind="dm",
            conversation_id="user-1",
        )
        handle = ChannelReplyHandle(platform="fake", payload={"token": "hidden"})
        message = InboundChannelMessage(
            address=address,
            message_id="m-1",
            sender_id="user-1",
            text="hello",
            received_at=datetime.now(timezone.utc),
            reply_handle=handle,
        )
        await adapter.emit(message)
        assert len(received) == 1
        assert received[0].text == "hello"

        await adapter.set_typing(address, True, reply_handle=handle)
        result = await adapter.send_text(address, "pong", reply_handle=handle)
        await adapter.set_typing(address, False, reply_handle=handle)
        assert result.ok is True
        assert adapter.sent_texts == ["pong"]
        assert adapter.typing_events == [(True, address.binding_key()), (False, address.binding_key())]
        await adapter.stop()

    asyncio.run(run())


def test_fake_adapter_rejects_unsupported_capability() -> None:
    async def run() -> None:
        adapter = FakeAdapter(
            instance_id="fake-2",
            capabilities=ChannelCapabilities(
                message_editing=False,
                native_streaming=False,
                typing=False,
                buttons=False,
                threads=False,
                inbound_media=frozenset(),
                outbound_media=frozenset(),
            ),
        )
        address = ChannelAddress(
            instance_id="fake-2",
            platform="fake",
            conversation_kind="dm",
            conversation_id="user-1",
        )
        with pytest.raises(UnsupportedCapabilityError):
            await adapter.set_typing(address, True)

    asyncio.run(run())
