"""
haagent/channels/adapters/fake.py - 测试用 FakeAdapter

用于验证会话绑定、事件呈现、审批与并发，不访问真实平台。
"""

from __future__ import annotations

from haagent.channels.adapter import InboundMessageHandler
from haagent.channels.types import (
    ChannelAddress,
    ChannelCapabilities,
    ChannelReplyHandle,
    InboundChannelMessage,
    SendResult,
    UnsupportedCapabilityError,
)


class FakeAdapter:
    """内存 Adapter：记录投递并支持测试侧注入入站消息。"""

    def __init__(
        self,
        instance_id: str,
        *,
        platform: str = "fake",
        capabilities: ChannelCapabilities | None = None,
    ) -> None:
        self.instance_id = instance_id
        self.platform = platform
        self.capabilities = capabilities or ChannelCapabilities(
            message_editing=False,
            native_streaming=False,
            typing=True,
            buttons=False,
            threads=False,
            inbound_media=frozenset(),
            outbound_media=frozenset(),
        )
        self._on_message: InboundMessageHandler | None = None
        self._started = False
        self.sent_texts: list[str] = []
        self.typing_events: list[tuple[bool, str]] = []
        self._send_counter = 0

    async def start(self, on_message: InboundMessageHandler) -> None:
        self._on_message = on_message
        self._started = True

    async def stop(self) -> None:
        self._started = False
        self._on_message = None

    async def emit(self, message: InboundChannelMessage) -> None:
        if not self._started or self._on_message is None:
            raise RuntimeError("FakeAdapter is not started")
        await self._on_message(message)

    async def send_text(
        self,
        address: ChannelAddress,
        text: str,
        *,
        reply_handle: ChannelReplyHandle | None = None,
        reply_to_message_id: str | None = None,
    ) -> SendResult:
        del address, reply_handle, reply_to_message_id
        if not text:
            return SendResult(ok=False, error="empty_text")
        self._send_counter += 1
        self.sent_texts.append(text)
        return SendResult(ok=True, message_id=f"fake-out-{self._send_counter}")

    async def set_typing(
        self,
        address: ChannelAddress,
        active: bool,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> None:
        del reply_handle
        # 能力未声明时必须显式失败，禁止假装成功。
        if not self.capabilities.typing:
            raise UnsupportedCapabilityError("typing is not supported")
        self.typing_events.append((active, address.binding_key()))
