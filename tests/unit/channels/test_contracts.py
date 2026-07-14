"""
tests/unit/channels/test_contracts.py - 渠道身份与 secret 合同测试

验证稳定身份键与敏感回复句柄的脱敏边界。
"""

from __future__ import annotations

from haagent.channels.types import (
    ChannelAddress,
    ChannelReplyHandle,
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
