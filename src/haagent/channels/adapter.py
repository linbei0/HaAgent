"""
haagent/channels/adapter.py - 渠道 Adapter Protocol

平台传输、入站解析、发送与 typing 的最小契约；不含 Agent 逻辑。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from haagent.channels.types import (
    ChannelAddress,
    ChannelCapabilities,
    ChannelReplyHandle,
    InboundChannelMessage,
    SendResult,
)

# 返回 outcome 字符串时 Adapter 可据此决定是否推进 cursor；None 视为可接受。
InboundMessageHandler = Callable[
    [InboundChannelMessage],
    Awaitable[str | None] | Awaitable[None] | str | None,
]


class ChannelAdapter(Protocol):
    instance_id: str
    platform: str
    capabilities: ChannelCapabilities

    async def start(self, on_message: InboundMessageHandler) -> None:
        """启动平台连接并注册入站回调。"""

    async def stop(self) -> None:
        """关闭连接并清理资源。"""

    async def send_text(
        self,
        address: ChannelAddress,
        text: str,
        *,
        reply_handle: ChannelReplyHandle | None = None,
        reply_to_message_id: str | None = None,
    ) -> SendResult:
        """发送纯文本回复。"""

    async def set_typing(
        self,
        address: ChannelAddress,
        active: bool,
        *,
        reply_handle: ChannelReplyHandle | None = None,
    ) -> None:
        """设置 typing 指示；不支持时必须显式失败。"""
