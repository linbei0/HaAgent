"""
tests/unit/channels/test_bugfix_approvals_keyring_qr.py - 多会话审批/keyring/QR 清理

覆盖：
1. 多 binding 审批不饥饿、不空转
2. keyring 故障与凭据缺失区分
3. QR 登录失败/过期/超时清理 HTTP client
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from haagent.app.assistant_context import AssistantContext
from haagent.app.channel_usecases import AssistantChannels
from haagent.channels.adapters.weixin.types import WeixinProtocolError, WeixinQrCode, WeixinQrStatus
from haagent.channels.interactions import InteractionBroker
from haagent.channels.runtime import ChannelGatewayRuntime
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, save_channel_settings
from haagent.channels.types import ChannelAddress
from tests.support.model_credentials import FakeCredentialStore
from haagent.models.gateway_registry import gateway_from_profile
from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.runtime.session.agent import AgentSession


def _addr(conversation_id: str) -> ChannelAddress:
    return ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id=conversation_id,
    )


def test_wait_for_pending_filters_by_binding_key() -> None:
    """第二个会话必须能看到自己的审批，不能被第一个卡住。"""
    broker = InteractionBroker(timeout_seconds=5.0)
    a1 = _addr("owner-1")
    a2 = _addr("owner-2")
    req = HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="run?",
    )
    holders: list[object] = []

    def worker(address: ChannelAddress) -> None:
        holders.append(
            broker.request_approval(
                req,
                owner_sender_id=address.conversation_id,
                binding_key=address.binding_key(),
            )
        )

    t1 = threading.Thread(target=worker, args=(a1,))
    t2 = threading.Thread(target=worker, args=(a2,))
    t1.start()
    t2.start()
    time.sleep(0.05)

    p2 = broker.wait_for_pending(binding_key=a2.binding_key(), timeout=2.0)
    assert p2 is not None
    assert p2.binding_key == a2.binding_key()
    p1 = broker.wait_for_pending(binding_key=a1.binding_key(), timeout=2.0)
    assert p1 is not None
    assert p1.binding_key == a1.binding_key()
    assert p1.nonce != p2.nonce

    broker.resolve(p2.nonce, approved=True, sender_id="owner-2", binding_key=a2.binding_key())
    broker.resolve(p1.nonce, approved=True, sender_id="owner-1", binding_key=a1.binding_key())
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert len(holders) == 2


def test_wait_for_pending_skips_seen_nonce_without_busy_spin() -> None:
    """已提示过的 pending 不应被立即反复返回造成空转。"""
    broker = InteractionBroker(timeout_seconds=5.0)
    address = _addr("owner-1")
    req = HumanInteractionRequest(interaction_type="approval", tool_name="shell", question="run?")

    def worker() -> None:
        broker.request_approval(
            req,
            owner_sender_id="owner-1",
            binding_key=address.binding_key(),
        )

    thread = threading.Thread(target=worker)
    thread.start()
    first = broker.wait_for_pending(binding_key=address.binding_key(), timeout=2.0)
    assert first is not None
    started = time.perf_counter()
    second = broker.wait_for_pending(
        binding_key=address.binding_key(),
        exclude_nonces={first.nonce},
        timeout=0.25,
    )
    elapsed = time.perf_counter() - started
    assert second is None
    # 应真正等待到接近 timeout，而不是微秒级空转返回。
    assert elapsed >= 0.15
    broker.resolve(first.nonce, approved=False, sender_id="owner-1", binding_key=address.binding_key())
    thread.join(timeout=2)


def test_runtime_keyring_failure_not_reported_as_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = tmp_path / "channels.json"
    save_channel_settings(
        config,
        ChannelSettings(
            version=1,
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:wx-1:bot_token",
                    metadata={},
                )
            ],
        ),
    )
    store = FakeCredentialStore(available=False, error="keyring locked")
    runtime = ChannelGatewayRuntime(
        config_path=config,
        state_path=tmp_path / "channels.sqlite3",
        default_workspace_root=workspace,
        service_factory=lambda root: object(),
        credential_store=store,
    )
    with pytest.raises(RuntimeError) as exc:
        runtime.build_adapters()
    text = str(exc.value).lower()
    assert "keyring" in text or "credential store" in text or "locked" in text
    assert "re-login" not in text


def test_list_instances_surfaces_credential_store_error(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = FakeCredentialStore({"channel:weixin:wx-1:bot_token": "tok"})
    channels = AssistantChannels(
        AssistantContext(
            workspace_root=workspace,
            runs_root=workspace / ".runs",
            environ={},
            gateway_factory=gateway_from_profile,
            session_factory=AgentSession,
            max_turns=4,
            enable_web=False,
            initial_resume=None,
            initial_continue=False,
        ),
        config_dir=tmp_path,
        credential_store=store,
        protocol_factory=lambda **kw: None,
    )
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            version=1,
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:wx-1:bot_token",
                    metadata={},
                )
            ],
        ),
    )
    ok = channels.list_instances()[0]
    assert ok.credential_available is True

    store.available = False
    store.error = "backend broken"
    bad = channels.list_instances()[0]
    assert bad.credential_available is False
    assert bad.state == "credential_store_error"
    assert bad.state != "auth_expired"


class _TrackingClient:
    def __init__(self, **kwargs: Any) -> None:
        self.closed = False
        self.polls = 0
        self._status = "wait"

    async def get_qrcode(self) -> WeixinQrCode:
        return WeixinQrCode(qrcode_url="https://example.com/qr", qrcode_id="qr-1")

    async def poll_qrcode_status(self, qrcode_id: str) -> WeixinQrStatus:
        self.polls += 1
        if self._status == "error":
            raise WeixinProtocolError("boom", errcode=-1)
        return WeixinQrStatus(status=self._status)

    async def aclose(self) -> None:
        self.closed = True


def test_qr_login_failed_status_closes_client(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    clients: list[_TrackingClient] = []

    def factory(**kwargs: Any) -> _TrackingClient:
        client = _TrackingClient(**kwargs)
        clients.append(client)
        return client

    channels = AssistantChannels(
        AssistantContext(
            workspace_root=workspace,
            runs_root=workspace / ".runs",
            environ={},
            gateway_factory=gateway_from_profile,
            session_factory=AgentSession,
            max_turns=4,
            enable_web=False,
            initial_resume=None,
            initial_continue=False,
        ),
        config_dir=tmp_path,
        credential_store=FakeCredentialStore(),
        protocol_factory=factory,
    )
    start = asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    clients[0]._status = "expired"
    poll = asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id=start.qrcode_id))
    assert poll.status == "expired"
    assert clients[0].closed is True
    assert "wx-1" not in channels._pending_qr


def test_qr_login_restart_closes_previous_client(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    clients: list[_TrackingClient] = []

    def factory(**kwargs: Any) -> _TrackingClient:
        client = _TrackingClient(**kwargs)
        clients.append(client)
        return client

    channels = AssistantChannels(
        AssistantContext(
            workspace_root=workspace,
            runs_root=workspace / ".runs",
            environ={},
            gateway_factory=gateway_from_profile,
            session_factory=AgentSession,
            max_turns=4,
            enable_web=False,
            initial_resume=None,
            initial_continue=False,
        ),
        config_dir=tmp_path,
        credential_store=FakeCredentialStore(),
        protocol_factory=factory,
    )
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    assert len(clients) == 2
    assert clients[0].closed is True
    assert clients[1].closed is False


def test_cancel_weixin_qr_login_closes_client(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    clients: list[_TrackingClient] = []

    def factory(**kwargs: Any) -> _TrackingClient:
        client = _TrackingClient(**kwargs)
        clients.append(client)
        return client

    channels = AssistantChannels(
        AssistantContext(
            workspace_root=workspace,
            runs_root=workspace / ".runs",
            environ={},
            gateway_factory=gateway_from_profile,
            session_factory=AgentSession,
            max_turns=4,
            enable_web=False,
            initial_resume=None,
            initial_continue=False,
        ),
        config_dir=tmp_path,
        credential_store=FakeCredentialStore(),
        protocol_factory=factory,
    )
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.cancel_weixin_qr_login("wx-1"))
    assert clients[0].closed is True
    assert "wx-1" not in channels._pending_qr
