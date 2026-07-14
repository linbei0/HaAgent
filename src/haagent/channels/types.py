"""
haagent/channels/types.py - 渠道消息与能力公共合同

定义前端无关的地址、入站消息、回复句柄与能力声明，敏感 payload 不得进入 repr。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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
    reply_handle: ChannelReplyHandle


@dataclass(frozen=True)
class SendResult:
    ok: bool
    error: str | None = None
