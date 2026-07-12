"""
haagent/channels/__init__.py - 聊天渠道网关公共导出

暴露渠道核心类型与适配器契约，供应用层与测试复用。
"""

from __future__ import annotations

from haagent.channels.types import (
    ChannelAddress,
    ChannelCapabilities,
    ChannelReplyHandle,
    InboundChannelMessage,
    SendResult,
    UnsupportedCapabilityError,
)

__all__ = [
    "ChannelAddress",
    "ChannelCapabilities",
    "ChannelReplyHandle",
    "InboundChannelMessage",
    "SendResult",
    "UnsupportedCapabilityError",
]
