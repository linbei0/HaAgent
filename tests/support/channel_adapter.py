"""
tests/support/channel_adapter.py - 渠道测试内存 Adapter

记录入站、文本和 typing 行为，不作为生产平台或配置入口。
"""

from __future__ import annotations

from haagent.channels.adapter import InboundMessageHandler
from haagent.channels.types import (
    ChannelAddress,
    ChannelReplyHandle,
    InboundChannelMessage,
    SendResult,
)


class FakeChannelAdapter:
    def __init__(self, instance_id: str, *, platform: str = "weixin") -> None:
        self.instance_id = instance_id
        self.platform = platform
        self.state = "stopped"
        self.last_error = ""
        self._on_message: InboundMessageHandler | None = None
        self.sent_texts: list[str] = []
        self.typing_events: list[tuple[bool, str]] = []

    async def start(self, on_message: InboundMessageHandler) -> None:
        self._on_message = on_message
        self.state = "connected"

    async def stop(self) -> None:
        self.state = "stopped"
        self._on_message = None

    async def emit(self, message: InboundChannelMessage) -> None:
        if self._on_message is None:
            raise RuntimeError("FakeChannelAdapter is not started")
        await self._on_message(message)

    async def send_text(
        self,
        address: ChannelAddress,
        text: str,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> SendResult:
        del address, reply_handle
        if not text:
            return SendResult(ok=False, error="empty_text")
        self.sent_texts.append(text)
        return SendResult(ok=True)

    async def set_typing(
        self,
        address: ChannelAddress,
        active: bool,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> None:
        del reply_handle
        self.typing_events.append((active, address.binding_key()))
