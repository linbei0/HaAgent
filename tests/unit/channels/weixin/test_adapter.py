"""
tests/unit/channels/weixin/test_adapter.py - WeixinAdapter 映射与生命周期测试
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from haagent.channels.adapters.weixin.adapter import WeixinAdapter
from haagent.channels.adapters.weixin.types import (
    WeixinAuthenticationExpired,
    WeixinInboundMessage,
    WeixinProtocolError,
    WeixinSendResult,
    WeixinUpdates,
)
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage


@dataclass
class FakeProtocol:
    updates_queue: list[WeixinUpdates] = field(default_factory=list)
    sent: list[dict[str, Any]] = field(default_factory=list)
    typing_calls: list[tuple[bool, str]] = field(default_factory=list)
    tickets: dict[str, str] = field(default_factory=dict)
    fail_mode: str | None = None
    get_updates_calls: int = 0
    lifecycle_calls: list[str] = field(default_factory=list)

    async def get_updates(self, *, cursor: str = "") -> WeixinUpdates:
        self.get_updates_calls += 1
        self.lifecycle_calls.append("get_updates")
        if self.fail_mode == "auth":
            raise WeixinAuthenticationExpired("expired", errcode=-14)
        if self.fail_mode == "protocol":
            raise WeixinProtocolError("boom", errcode=-999)
        if self.updates_queue:
            return self.updates_queue.pop(0)
        return WeixinUpdates(messages=[], cursor=cursor)

    async def notify_start(self) -> None:
        self.lifecycle_calls.append("notify_start")

    async def notify_stop(self) -> None:
        self.lifecycle_calls.append("notify_stop")

    async def send_text(self, *, to_user_id: str, text: str, context_token: str) -> WeixinSendResult:
        self.sent.append({"to": to_user_id, "text": text, "ctx": context_token})
        return WeixinSendResult(ok=True)

    async def get_typing_ticket(self, *, context_token: str) -> str:
        ticket = self.tickets.get(context_token, f"ticket-for-{context_token[:4]}")
        self.tickets[context_token] = ticket
        return ticket

    async def send_typing(self, *, typing_ticket: str, active: bool, context_token: str) -> None:
        self.typing_calls.append((active, typing_ticket))

    async def aclose(self) -> None:
        return None


def _wx_msg(
    *,
    message_id: str = "m1",
    from_user_id: str = "u1",
    text: str = "hello",
    context_token: str = "ctx-secret",
) -> WeixinInboundMessage:
    return WeixinInboundMessage(
        message_id=message_id,
        from_user_id=from_user_id,
        text=text,
        context_token=context_token,
    )


def test_only_user_dm_text_received() -> None:
    async def _run() -> None:
        proto = FakeProtocol(
            updates_queue=[
                WeixinUpdates(
                    messages=[
                        _wx_msg(message_id="m1", text="hi"),
                        WeixinInboundMessage(
                            message_id="sys1",
                            from_user_id="",
                            text="system",
                            context_token="",
                        ),
                    ],
                    cursor="c1",
                )
            ]
        )
        received: list[InboundChannelMessage] = []

        async def on_message(msg: InboundChannelMessage) -> None:
            received.append(msg)

        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(on_message)
        deadline = asyncio.get_event_loop().time() + 2
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        await adapter.stop()
        assert len(received) == 1
        assert received[0].text == "hi"
        assert received[0].sender_id == "u1"
        assert received[0].address.conversation_kind == "dm"
        assert received[0].message_id == "m1"
        assert "ctx-secret" not in repr(received[0].reply_handle)

    asyncio.run(_run())


def test_user_message_type_is_delivered() -> None:
    """iLink message_type=1 表示用户消息，必须进入渠道运行时。"""
    adapter = WeixinAdapter(instance_id="wx-1", protocol=object(), poll_interval=0.01)
    message = WeixinInboundMessage(
        message_id="user-1",
        from_user_id="u1",
        text="hello",
        context_token="ctx",
        raw={"message_type": 1},
    )

    inbound = adapter._to_inbound(message)

    assert inbound is not None
    assert inbound.message_id == "user-1"
    assert inbound.text == "hello"


def test_bot_message_type_is_ignored() -> None:
    """iLink message_type=2 表示机器人消息，不能再次进入 Agent。"""
    adapter = WeixinAdapter(instance_id="wx-1", protocol=object(), poll_interval=0.01)
    message = WeixinInboundMessage(
        message_id="bot-1",
        from_user_id="bot-user",
        text="assistant reply",
        context_token="ctx",
        raw={"message_type": 2},
    )

    assert adapter._to_inbound(message) is None


def test_reply_handle_not_serialized() -> None:
    handle = ChannelReplyHandle(platform="weixin", payload={"context_token": "secret-ctx"})
    assert "secret-ctx" not in repr(handle)
    # 不得 JSON 序列化友好
    with pytest.raises(TypeError):
        import json

        json.dumps(handle)  # type: ignore[arg-type]


def test_text_split_at_3000() -> None:
    async def _run() -> None:
        proto = FakeProtocol()
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto)
        address = ChannelAddress(
            instance_id="wx-1",
            platform="weixin",
            conversation_kind="dm",
            conversation_id="u1",
        )
        handle = ChannelReplyHandle(platform="weixin", payload={"context_token": "ctx", "to_user_id": "u1"})
        long_text = "a" * 6500
        result = await adapter.send_text(address, long_text, reply_handle=handle)
        assert result.ok
        assert len(proto.sent) >= 2
        assert all(len(item["text"]) <= 3000 for item in proto.sent)
        assert sum(len(item["text"]) for item in proto.sent) == 6500

    asyncio.run(_run())


def test_typing_ticket_cached_and_closed() -> None:
    async def _run() -> None:
        proto = FakeProtocol()
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto)
        address = ChannelAddress(
            instance_id="wx-1",
            platform="weixin",
            conversation_kind="dm",
            conversation_id="u1",
        )
        handle = ChannelReplyHandle(platform="weixin", payload={"context_token": "ctx-t", "to_user_id": "u1"})
        await adapter.set_typing(address, True, reply_handle=handle)
        await adapter.set_typing(address, True, reply_handle=handle)
        await adapter.set_typing(address, False, reply_handle=handle)
        # 同 context 只取一次 ticket
        assert len(proto.tickets) == 1
        assert proto.typing_calls[0][0] is True
        assert proto.typing_calls[-1][0] is False

    asyncio.run(_run())


def test_auth_expired_stops_and_marks_state() -> None:
    async def _run() -> None:
        proto = FakeProtocol(fail_mode="auth")
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda m: asyncio.sleep(0))
        deadline = asyncio.get_event_loop().time() + 2
        while adapter.state not in {"auth_expired", "failed", "stopped"} and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        assert adapter.state == "auth_expired"
        await adapter.stop()

    asyncio.run(_run())


def test_start_and_stop_notify_weixin_around_poll_loop() -> None:
    async def _run() -> None:
        proto = FakeProtocol()
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda message: "accepted")
        deadline = asyncio.get_event_loop().time() + 2
        while proto.get_updates_calls == 0 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)
        await adapter.stop()

        assert proto.lifecycle_calls[0:2] == ["notify_start", "get_updates"]
        assert proto.lifecycle_calls[-1] == "notify_stop"

    asyncio.run(_run())


def test_start_notification_failure_retries_before_polling() -> None:
    class RetryStartProtocol(FakeProtocol):
        notify_start_calls = 0

        async def notify_start(self) -> None:
            self.notify_start_calls += 1
            self.lifecycle_calls.append("notify_start")
            if self.notify_start_calls == 1:
                raise RuntimeError("notify start transport unavailable")

        async def get_updates(self, *, cursor: str = "") -> WeixinUpdates:
            assert self.notify_start_calls >= 2
            self.get_updates_calls += 1
            self.lifecycle_calls.append("get_updates")
            if self.get_updates_calls == 1:
                return WeixinUpdates(messages=[_wx_msg(message_id="m-after-start-retry")], cursor="c2")
            return WeixinUpdates(messages=[], cursor=cursor)

    async def _run() -> None:
        proto = RetryStartProtocol()
        received: list[InboundChannelMessage] = []
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda message: received.append(message) or "accepted")
        deadline = asyncio.get_event_loop().time() + 2
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert [message.message_id for message in received] == ["m-after-start-retry"]
        assert proto.lifecycle_calls[:3] == ["notify_start", "notify_start", "get_updates"]
        assert adapter.state == "connected"
        assert adapter.last_error == ""
        await adapter.stop()

    asyncio.run(_run())


def test_auth_expired_calls_notify_start_and_recovers_without_relogin() -> None:
    class RecoveringProtocol(FakeProtocol):
        notify_start_calls = 0

        async def notify_start(self) -> None:
            self.notify_start_calls += 1
            self.lifecycle_calls.append("notify_start")

        async def get_updates(self, *, cursor: str = "") -> WeixinUpdates:
            self.get_updates_calls += 1
            self.lifecycle_calls.append("get_updates")
            if self.notify_start_calls < 2:
                raise WeixinAuthenticationExpired("session expired", errcode=-14)
            if self.get_updates_calls == 2:
                return WeixinUpdates(messages=[_wx_msg(message_id="m-after-notify-start")], cursor="c2")
            return WeixinUpdates(messages=[], cursor=cursor)

    async def _run() -> None:
        proto = RecoveringProtocol()
        received: list[InboundChannelMessage] = []
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda message: received.append(message) or "accepted")
        deadline = asyncio.get_event_loop().time() + 2
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert [message.message_id for message in received] == ["m-after-notify-start"]
        assert proto.notify_start_calls == 2
        assert adapter.state == "connected"
        await adapter.stop()

    asyncio.run(_run())


def test_permanent_protocol_error_enters_failed_without_retry() -> None:
    async def _run() -> None:
        proto = FakeProtocol(fail_mode="protocol")
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda message: "accepted")
        deadline = asyncio.get_running_loop().time() + 2
        while adapter.state != "failed" and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

        assert adapter.state == "failed"
        assert adapter.last_error == "boom"
        calls = proto.get_updates_calls
        await asyncio.sleep(0.05)
        assert proto.get_updates_calls == calls
        await adapter.stop()

    asyncio.run(_run())


def test_transient_transport_error_stays_reconnecting_until_success() -> None:
    entered_retry = asyncio.Event()
    release_retry = asyncio.Event()

    class TransientProtocol(FakeProtocol):
        async def get_updates(self, *, cursor: str = "") -> WeixinUpdates:
            self.get_updates_calls += 1
            if self.get_updates_calls == 1:
                raise RuntimeError("temporary transport error")
            if self.get_updates_calls == 2:
                entered_retry.set()
                await release_retry.wait()
                return WeixinUpdates(messages=[_wx_msg(message_id="m-recovered")], cursor="recovered")
            return WeixinUpdates(messages=[], cursor=cursor)

    async def _run() -> None:
        proto = TransientProtocol()
        received: list[InboundChannelMessage] = []
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(lambda message: received.append(message) or "accepted")
        await asyncio.wait_for(entered_retry.wait(), timeout=2)
        assert adapter.state == "reconnecting"
        release_retry.set()
        deadline = asyncio.get_event_loop().time() + 2
        while not received and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)

        assert [message.message_id for message in received] == ["m-recovered"]
        assert adapter.state == "connected"
        assert adapter.last_error == ""
        await adapter.stop()

    asyncio.run(_run())


def test_partial_batch_failure_does_not_advance_cursor() -> None:
    async def _run() -> None:
        msg = _wx_msg(message_id="m-fail")
        proto = FakeProtocol(
            updates_queue=[
                WeixinUpdates(messages=[msg], cursor="advanced"),
            ]
        )
        # handler 抛错模拟 Actor 拒绝
        async def bad_handler(m: InboundChannelMessage) -> None:
            raise RuntimeError("actor failed")

        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        # 注入初始 cursor
        adapter._cursor = "old-cursor"
        await adapter.start(bad_handler)
        await asyncio.sleep(0.15)
        await adapter.stop()
        # 失败不得推进 cursor
        assert adapter.cursor == "old-cursor"

    asyncio.run(_run())


def test_send_without_context_token_fails() -> None:
    async def _run() -> None:
        proto = FakeProtocol()
        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto)
        address = ChannelAddress(
            instance_id="wx-1",
            platform="weixin",
            conversation_kind="dm",
            conversation_id="u1",
        )
        result = await adapter.send_text(address, "hi", reply_handle=None)
        assert result.ok is False
        assert result.error

    asyncio.run(_run())


def test_successful_batch_persists_cursor_via_callback() -> None:
    async def _run() -> None:
        persisted: list[str] = []
        proto = FakeProtocol(
            updates_queue=[
                WeixinUpdates(messages=[_wx_msg(message_id="m1", text="hi")], cursor="cursor-next"),
            ]
        )
        received: list[InboundChannelMessage] = []

        async def on_message(msg: InboundChannelMessage) -> str:
            received.append(msg)
            return "accepted"

        adapter = WeixinAdapter(
            instance_id="wx-1",
            protocol=proto,
            poll_interval=0.01,
            on_cursor_persist=lambda value: persisted.append(value),
        )
        await adapter.start(on_message)
        deadline = asyncio.get_event_loop().time() + 2
        while not persisted and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        await adapter.stop()
        assert received
        assert adapter.cursor == "cursor-next"
        assert persisted == ["cursor-next"]

    asyncio.run(_run())


def test_handler_reject_does_not_persist_cursor() -> None:
    async def _run() -> None:
        persisted: list[str] = []
        proto = FakeProtocol(
            updates_queue=[
                WeixinUpdates(messages=[_wx_msg(message_id="m-rej")], cursor="should-not"),
            ]
        )

        async def on_message(msg: InboundChannelMessage) -> str:
            return "busy"

        adapter = WeixinAdapter(
            instance_id="wx-1",
            protocol=proto,
            poll_interval=0.01,
            initial_cursor="old",
            on_cursor_persist=lambda value: persisted.append(value),
        )
        await adapter.start(on_message)
        await asyncio.sleep(0.15)
        await adapter.stop()
        assert adapter.cursor == "old"
        assert persisted == []

    asyncio.run(_run())
