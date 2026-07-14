"""
haagent/channels/adapters/weixin/types.py - 微信 iLink 请求/响应与错误分类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


class WeixinProtocolError(RuntimeError):
    """微信协议层错误基类。"""

    def __init__(self, message: str, *, errcode: int | None = None) -> None:
        super().__init__(message)
        self.errcode = errcode


class WeixinAuthenticationExpired(WeixinProtocolError):
    """微信长轮询 session 失效（如 errcode=-14），优先 notifyStart 恢复。"""


class WeixinRateLimited(WeixinProtocolError):
    """频率限制，调用方应受控退避。"""


class WeixinUnsupportedBaseUrl(WeixinProtocolError):
    """base URL 不在官方 allowlist。"""


@dataclass(frozen=True)
class WeixinQrCode:
    qrcode_url: str
    qrcode_id: str


@dataclass(frozen=True, repr=False)
class WeixinQrStatus:
    status: Literal[
        "wait",
        "scaned",
        "confirmed",
        "expired",
        "unknown",
        "scaned_but_redirect",
        "binded_redirect",
        "need_verifycode",
        "verify_code_blocked",
    ]
    bot_token: str | None = None
    ilink_bot_id: str | None = None
    ilink_user_id: str | None = None
    base_url: str | None = None

    def __repr__(self) -> str:
        # 禁止在日志/测试失败输出中泄露 bot_token。
        return (
            f"WeixinQrStatus(status={self.status!r}, "
            f"ilink_bot_id={self.ilink_bot_id!r}, "
            f"has_token={bool(self.bot_token)})"
        )


@dataclass(frozen=True, repr=False)
class WeixinInboundMessage:
    message_id: str
    from_user_id: str
    text: str
    context_token: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"WeixinInboundMessage(message_id={self.message_id!r}, "
            f"from_user_id={self.from_user_id!r}, text_len={len(self.text)})"
        )


@dataclass(frozen=True)
class WeixinUpdates:
    messages: list[WeixinInboundMessage]
    cursor: str
