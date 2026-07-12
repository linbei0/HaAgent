"""
haagent/channels/adapters/weixin/__init__.py - 微信 iLink Adapter 包
"""

from __future__ import annotations

from haagent.channels.adapters.weixin.types import (
    WeixinAuthenticationExpired,
    WeixinProtocolError,
    WeixinRateLimited,
    WeixinUnsupportedBaseUrl,
)

__all__ = [
    "WeixinAuthenticationExpired",
    "WeixinProtocolError",
    "WeixinRateLimited",
    "WeixinUnsupportedBaseUrl",
]
