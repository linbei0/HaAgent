"""
tests/unit/channels/test_p1_contracts.py - P1 渠道合同失败测试

覆盖 pairing 重发、workspace 更新、permission 锁定、preflight、
DM 过滤、typing finally、presenter 静默工具摘要。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from haagent.app.assistant_context import AssistantContext
from haagent.app.channel_usecases import AssistantChannels
from haagent.channels.adapters.weixin.adapter import WeixinAdapter
from haagent.channels.adapters.weixin.types import (
    WeixinInboundMessage,
    WeixinUpdates,
)
from haagent.channels.presenter import ChannelPresenter, SendText, SetTyping
from haagent.channels.session_actor import ChannelSessionActor
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, save_channel_settings
from haagent.channels.state import ChannelStateStore
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage
from haagent.models.gateway_registry import gateway_from_profile
from haagent.runtime.events.types import (
    AssistantMessageEvent,
    FailureNoticeEvent,
    SessionLifecycleEvent,
    ToolActivityEvent,
)
from haagent.runtime.session.agent import AgentSession
from tests.support.channel_adapter import FakeChannelAdapter as FakeAdapter
from tests.support.model_credentials import FakeCredentialStore


# ---- helpers ----


class _FakeWeixinProtocol:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.polls = 0

    async def get_qrcode(self):
        from haagent.channels.adapters.weixin.types import WeixinQrCode

        return WeixinQrCode(qrcode_url="https://example.com/qr", qrcode_id="qr-1")

    async def poll_qrcode_status(self, qrcode_id: str):
        from haagent.channels.adapters.weixin.types import WeixinQrStatus

        self.polls += 1
        if self.polls < 2:
            return WeixinQrStatus(status="wait")
        return WeixinQrStatus(
            status="confirmed",
            bot_token="bot-secret-token",
            ilink_bot_id="bot-abc",
            ilink_user_id="user-xyz",
            base_url="https://ilinkai.weixin.qq.com",
        )

    async def get_updates(self, *, cursor: str = ""):
        return WeixinUpdates(messages=[], cursor=cursor or "c1")

    async def aclose(self) -> None:
        return None


def _context(workspace: Path) -> AssistantContext:
    return AssistantContext(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        environ={},
        gateway_factory=gateway_from_profile,
        session_factory=AgentSession,
        max_turns=8,
        enable_web=False,
        initial_resume=None,
        initial_continue=False,
    )


def _channels(tmp_path: Path, workspace: Path, store: FakeCredentialStore) -> AssistantChannels:
    return AssistantChannels(
        _context(workspace),
        config_dir=tmp_path,
        credential_store=store,
        protocol_factory=_FakeWeixinProtocol,
    )


def _seed_instance(tmp_path: Path, workspace: Path, store: FakeCredentialStore) -> AssistantChannels:
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    return channels


# ---- P1-6 set_workspace + re-issue already exists; set_workspace new ----


def test_set_workspace_root_updates_channels_json(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    new_ws = tmp_path / "other"
    new_ws.mkdir()
    store = FakeCredentialStore()
    channels = _seed_instance(tmp_path, workspace, store)

    updated = channels.set_workspace_root("wx-1", new_ws)
    assert updated.workspace_root.resolve() == new_ws.resolve()
    settings = __import__("haagent.channels.settings", fromlist=["load_channel_settings"]).load_channel_settings(
        tmp_path / "channels.json"
    )
    assert settings.instances[0].workspace_root.resolve() == new_ws.resolve()


def test_set_workspace_rejects_missing_path(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    store = FakeCredentialStore()
    channels = _seed_instance(tmp_path, workspace, store)
    from haagent.app.assistant_types import AssistantServiceError

    with pytest.raises(AssistantServiceError):
        channels.set_workspace_root("wx-1", tmp_path / "missing")


# ---- P1-13 presenter tool silence ----


def test_tool_summary_only_after_silence(monkeypatch) -> None:
    from haagent.channels import presenter as presenter_module

    clock = [0.0]
    monkeypatch.setattr(presenter_module.time, "monotonic", lambda: clock[0])
    presenter = ChannelPresenter()
    presenter.handle(SessionLifecycleEvent("s", 1, "turn_started", "start"))
    # 刚开始立刻 tool：不应发摘要
    early = presenter.handle(ToolActivityEvent("s", 1, 1, "shell", "started", "run tests"))
    assert not any(isinstance(a, SendText) for a in early)
    # 模拟静默后
    clock[0] = 9.0
    late = presenter.handle(ToolActivityEvent("s", 1, 2, "file_read", "started", "read"))
    texts = [a for a in late if isinstance(a, SendText)]
    assert texts
    assert "file_read" in texts[0].text


# ---- P1-12 typing finally on actor exception path ----


def test_actor_turns_typing_off_on_exception(tmp_path: Path) -> None:
    from haagent.channels.interactions import InteractionBroker
    from tests.unit.channels.test_session_actor import FakeAssistantService, _address, _message

    state = ChannelStateStore(tmp_path / "s.sqlite3")
    adapter = FakeAdapter(instance_id="wx-1")
    service = FakeAssistantService(tmp_path / "ws")
    service.emit_lifecycle = True

    # 让 run 抛错
    def _boom(*args, **kwargs):
        raise RuntimeError("model boom")

    service._run_prompt = _boom  # type: ignore[method-assign]
    actor = ChannelSessionActor(
        binding_key="k",
        address=_address(),
        owner_sender_id="owner-1",
        workspace_root=tmp_path / "ws",
        state=state,
        adapter=adapter,
        service=service,
        broker=InteractionBroker(),
    )
    (tmp_path / "ws").mkdir(exist_ok=True)

    async def _run() -> None:
        await adapter.start(lambda m: None)
        msg = _message("hi", message_id="m-ex")
        await actor.submit(msg)
        deadline = asyncio.get_event_loop().time() + 3
        while actor._active and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)
        await actor.close()
        await adapter.stop()

    asyncio.run(_run())
    # FakeAdapter 记录 typing；至少有一次 active=False
    typing_offs = [c for c in adapter.typing_events if c[0] is False]
    assert typing_offs, f"expected typing off, got {adapter.typing_events}"
    state.close()


# ---- P1-11 filter empty text / non-text ----


def test_weixin_drops_empty_text_even_with_context_token() -> None:
    async def _run() -> None:
        from dataclasses import dataclass, field

        @dataclass
        class Proto:
            updates_queue: list = field(default_factory=list)
            get_updates_calls: int = 0

            async def get_updates(self, *, cursor: str = ""):
                self.get_updates_calls += 1
                if self.updates_queue:
                    return self.updates_queue.pop(0)
                return WeixinUpdates(messages=[], cursor=cursor)

            async def notify_start(self):
                return None

            async def notify_stop(self):
                return None

            async def aclose(self):
                return None

        proto = Proto(
            updates_queue=[
                WeixinUpdates(
                    messages=[
                        WeixinInboundMessage(
                            message_id="empty1",
                            from_user_id="u1",
                            text="   ",
                            context_token="ctx",
                        ),
                        WeixinInboundMessage(
                            message_id="ok1",
                            from_user_id="u1",
                            text="hello",
                            context_token="ctx",
                        ),
                    ],
                    cursor="c1",
                )
            ]
        )
        received: list = []

        async def on_message(msg):
            received.append(msg)

        adapter = WeixinAdapter(instance_id="wx-1", protocol=proto, poll_interval=0.01)
        await adapter.start(on_message)
        deadline = asyncio.get_event_loop().time() + 2
        while len(received) < 1 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        await adapter.stop()
        assert len(received) == 1
        assert received[0].text == "hello"

    asyncio.run(_run())


def test_weixin_drops_group_and_non_text_message_types() -> None:
    """group / media / non-user message_type must not reach manager."""
    msg_group = WeixinInboundMessage(
        message_id="g1",
        from_user_id="u1",
        text="hi group",
        context_token="ctx",
        raw={"message_type": 3, "room_id": "room-1"},
    )
    msg_media = WeixinInboundMessage(
        message_id="media1",
        from_user_id="u1",
        text="",
        context_token="ctx",
        raw={"message_type": 2, "item_list": [{"type": 2, "image_item": {}}]},
    )
    adapter = WeixinAdapter(instance_id="wx-1", protocol=object(), poll_interval=0.01)
    assert adapter._to_inbound(msg_group) is None
    assert adapter._to_inbound(msg_media) is None


# ---- P1-9 permission lock ----


def test_channel_actor_forces_request_approval(tmp_path: Path) -> None:
    from haagent.channels.interactions import InteractionBroker
    from tests.unit.channels.test_session_actor import FakeAssistantService, _address, _message

    state = ChannelStateStore(tmp_path / "s.sqlite3")
    adapter = FakeAdapter(instance_id="wx-1")
    service = FakeAssistantService(tmp_path / "ws")
    (tmp_path / "ws").mkdir(exist_ok=True)

    actor = ChannelSessionActor(
        binding_key="k",
        address=_address(),
        owner_sender_id="owner-1",
        workspace_root=tmp_path / "ws",
        state=state,
        adapter=adapter,
        service=service,
        broker=InteractionBroker(),
    )

    async def _run() -> None:
        await adapter.start(lambda m: None)
        await actor.submit(_message("hi"))
        deadline = asyncio.get_event_loop().time() + 3
        while actor._active and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)
        await actor.close()
        await adapter.stop()

    asyncio.run(_run())
    assert service.sessions.permission_modes
    assert service.sessions.permission_modes[0] == "request_approval"
    state.close()


# ---- P1-10 preflight ----


def test_gateway_preflight_fails_without_model_key(tmp_path: Path, monkeypatch, capsys) -> None:
    from haagent import cli
    from haagent import cli_commands
    from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, save_channel_settings
    from haagent.models.credentials import KEYRING_SERVICE_NAME

    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    save_channel_settings(
        config_dir / "channels.json",
        ChannelSettings(
            version=1,
            instances=[
                ChannelInstanceConfig(
                    id="f1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:f1:bot_token",
                    metadata={},
                )
            ],
        ),
    )
    monkeypatch.setattr(
        cli_commands,
        "_gateway_model_preflight",
        lambda workspace_root: (False, "model credential unavailable; configure via TUI first"),
    )

    code = cli.main(["gateway", "run", "--workspace-root", str(workspace)])
    text = capsys.readouterr().out + capsys.readouterr().err
    assert code != 0
    assert "model" in text.lower() or "credential" in text.lower() or "configure" in text.lower()
