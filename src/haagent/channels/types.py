"""
haagent/channels/types.py - 渠道消息与能力公共合同

定义前端无关的地址、入站消息、回复句柄与能力声明，敏感 payload 不得进入 repr。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


class UnsupportedCapabilityError(RuntimeError):
    """平台未声明的能力被调用时显式失败，禁止静默降级。"""


@dataclass(frozen=True)
class ChannelAddress:
    instance_id: str
    platform: str
    conversation_kind: Literal["dm", "group", "channel", "thread"]
    conversation_id: str
    thread_id: str | None = None

    def binding_key(self) -> str:
        base = f"{self.platform}:{self.instance_id}:{self.conversation_kind}:{self.conversation_id}"
        if self.thread_id:
            return f"{base}:{self.thread_id}"
        return base


@dataclass(frozen=True, repr=False)
class ChannelReplyHandle:
    """仅原始 Adapter 可消费 payload；不得序列化或落盘。"""

    platform: str
    payload: object

    def __repr__(self) -> str:
        # 敏感 context_token 等不得出现在日志或测试失败输出中。
        return f"ChannelReplyHandle(platform={self.platform!r})"


@dataclass(frozen=True)
class InboundChannelMessage:
    address: ChannelAddress
    message_id: str
    sender_id: str
    text: str
    received_at: datetime
    reply_handle: ChannelReplyHandle
    reply_to_message_id: str | None = None


@dataclass(frozen=True)
class ChannelCapabilities:
    message_editing: bool
    native_streaming: bool
    typing: bool
    buttons: bool
    threads: bool
    inbound_media: frozenset[str]
    outbound_media: frozenset[str]


@dataclass(frozen=True)
class SendResult:
    ok: bool
    message_id: str | None = None
    error: str | None = None
