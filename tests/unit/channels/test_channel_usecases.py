"""
tests/unit/channels/test_channel_usecases.py - 渠道应用用例测试

验证微信 QR 登录写 keyring、配置落盘不含 token、删除清理 state/credential。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from haagent.app.assistant_context import AssistantContext
from haagent.app.channel_usecases import AssistantChannels
from haagent.channels.adapters.weixin.types import WeixinQrCode, WeixinQrStatus
from haagent.channels.process_lock import GatewayInstanceLock
from haagent.channels.settings import (
    ChannelInstanceConfig,
    ChannelSettings,
    ChannelSettingsError,
    load_channel_settings,
    save_channel_settings,
)
from haagent.channels.state import ChannelStateStore
from haagent.models.config.credentials import KEYRING_SERVICE_NAME
from tests.support.model_credentials import FakeCredentialStore
from haagent.models.gateway_registry import gateway_from_resolved
from haagent.runtime.session.agent import AgentSession


class _FakeWeixinProtocol:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._token = kwargs.get("bot_token", "")
        self.polls = 0

    async def get_qrcode(self) -> WeixinQrCode:
        return WeixinQrCode(qrcode_url="https://example.com/qr", qrcode_id="qr-1")

    async def poll_qrcode_status(self, qrcode_id: str) -> WeixinQrStatus:
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
        from haagent.channels.adapters.weixin.types import WeixinUpdates

        return WeixinUpdates(messages=[], cursor=cursor or "c1")

    async def notify_start(self) -> None:
        return None

    async def notify_stop(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _context(workspace: Path) -> AssistantContext:
    return AssistantContext(
        workspace_root=workspace,
        runs_root=workspace / ".runs",
        environ={},
            gateway_factory=gateway_from_resolved,
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


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


def test_weixin_qr_login_saves_token_to_keyring_not_settings(tmp_path: Path, workspace: Path) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)

    start = asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    assert start.instance_id == "wx-1"
    assert start.qrcode_id == "qr-1"
    assert "bot-secret" not in repr(start)

    first = asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    assert first.status == "wait"
    assert store.get_password(KEYRING_SERVICE_NAME, "channel:weixin:wx-1:bot_token") is None

    done = asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    assert done.status == "confirmed"
    assert done.credential_available is True
    assert "bot-secret" not in repr(done)
    # 登录成功后只展示一次 8 位配对码；磁盘只存 hash。
    assert done.pairing_code is not None
    assert len(done.pairing_code) == 8
    assert done.pairing_code.isalnum()
    assert "bot-secret" not in (done.pairing_code or "")

    username = "channel:weixin:wx-1:bot_token"
    assert store.get_password(KEYRING_SERVICE_NAME, username) == "bot-secret-token"

    settings = load_channel_settings(tmp_path / "channels.json")
    assert len(settings.instances) == 1
    inst = settings.instances[0]
    assert inst.id == "wx-1"
    assert inst.platform == "weixin"
    assert inst.credential_username == username
    assert inst.workspace_root == workspace.resolve()
    assert inst.metadata["ilink_bot_id"] == "bot-abc"
    raw = (tmp_path / "channels.json").read_text(encoding="utf-8")
    assert "bot-secret-token" not in raw
    assert done.pairing_code not in raw

    # 配对码已写入 state，可用 consume 验证；明文不落库。
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    owner = state.consume_pairing_token("wx-1", done.pairing_code, sender_id="owner-wx")
    assert owner == "owner-wx"
    state.close()


def test_list_instances_reports_credential_and_configured_state(
    tmp_path: Path, workspace: Path
) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    items = channels.list_instances()
    assert len(items) == 1
    assert items[0].id == "wx-1"
    assert items[0].credential_available is True
    assert items[0].state == "configured"
    assert items[0].enabled is True
    assert "bot-secret" not in repr(items[0])


def test_delete_instance_removes_settings_keyring_and_state(tmp_path: Path, workspace: Path) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    state.set_owner("wx-1", "owner-1")
    state.upsert_binding(
        binding_key="weixin:wx-1:dm:u1",
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="u1",
        thread_id=None,
        workspace_root=str(workspace),
        session_id="sess-1",
        owner_sender_id="owner-1",
    )
    state.close()

    channels.delete_instance("wx-1")

    assert channels.list_instances() == []
    assert store.get_password(KEYRING_SERVICE_NAME, "channel:weixin:wx-1:bot_token") is None
    state2 = ChannelStateStore(tmp_path / "channels.sqlite3")
    assert state2.get_owner("wx-1") is None
    assert state2.get_binding("weixin:wx-1:dm:u1") is None
    state2.close()


def test_set_enabled_and_test_connection(tmp_path: Path, workspace: Path) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    disabled = channels.set_enabled("wx-1", False)
    assert disabled.enabled is False

    result = asyncio.run(channels.test_connection("wx-1"))
    assert result.ok is True
    assert "bot-secret" not in repr(result)
    assert "bot-secret" not in result.message


def test_connection_uses_lifecycle_probe_without_polling(
    tmp_path: Path, workspace: Path
) -> None:
    username = "channel:weixin:wx-1:bot_token"
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username=username,
                )
            ]
        ),
    )
    calls: list[str] = []

    class _ProbeProtocol:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def notify_start(self) -> None:
            calls.append("notify_start")

        async def notify_stop(self) -> None:
            calls.append("notify_stop")

        async def get_updates(self, *, cursor: str = "") -> None:
            del cursor
            calls.append("get_updates")

        async def aclose(self) -> None:
            calls.append("aclose")

    credentials = FakeCredentialStore({username: "secret-token"})
    channels = AssistantChannels(
        _context(workspace),
        config_dir=tmp_path,
        credential_store=credentials,
        protocol_factory=_ProbeProtocol,
    )

    result = asyncio.run(channels.test_connection("wx-1"))

    assert result.ok is True
    assert calls == ["notify_start", "notify_stop", "aclose"]


def test_connection_refuses_probe_while_gateway_holds_lock(
    tmp_path: Path, workspace: Path
) -> None:
    username = "channel:weixin:wx-1:bot_token"
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username=username,
                )
            ]
        ),
    )
    constructed = 0

    def factory(**kwargs: Any) -> _FakeWeixinProtocol:
        nonlocal constructed
        del kwargs
        constructed += 1
        return _FakeWeixinProtocol()

    channels = AssistantChannels(
        _context(workspace),
        config_dir=tmp_path,
        credential_store=FakeCredentialStore({username: "secret-token"}),
        protocol_factory=factory,
    )
    with GatewayInstanceLock(tmp_path / "gateway.lock"):
        result = asyncio.run(channels.test_connection("wx-1"))

    assert result.ok is False
    assert "already running" in result.message
    assert constructed == 0


def test_issue_pairing_code_for_configured_instance(tmp_path: Path, workspace: Path) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    code = channels.issue_pairing_code("wx-1")
    assert len(code) == 8
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    assert state.consume_pairing_token("wx-1", code, sender_id="u-pair") == "u-pair"
    state.close()


def test_reauth_marks_need_login_when_no_credential(tmp_path: Path, workspace: Path) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    store.delete_password(KEYRING_SERVICE_NAME, "channel:weixin:wx-1:bot_token")

    items = channels.list_instances()
    assert items[0].credential_available is False
    assert items[0].state == "auth_expired"

    start = asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    assert start.instance_id == "wx-1"


def test_same_weixin_identity_relogin_preserves_workspace_owner_cursor_and_permission(
    tmp_path: Path, workspace: Path
) -> None:
    original_workspace = tmp_path / "original-workspace"
    original_workspace.mkdir()
    username = "channel:weixin:wx-1:bot_token"
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=original_workspace,
                    credential_username=username,
                    metadata={"ilink_bot_id": "bot-abc", "ilink_user_id": "user-xyz"},
                    permission_mode="auto_approve",
                )
            ]
        ),
    )
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    state.set_owner("wx-1", "owner-1")
    state.set_cursor("wx-1", "get_updates_buf", "cursor-1")
    state.set_permanent_permission_binding(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root=str(original_workspace.resolve()),
        permission_mode="auto_approve",
    )
    state.close()
    channels = _channels(tmp_path, workspace, FakeCredentialStore())

    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    instance = load_channel_settings(tmp_path / "channels.json").instances[0]
    assert instance.workspace_root == original_workspace.resolve()
    assert instance.permission_mode == "auto_approve"
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    assert state.get_owner("wx-1") == "owner-1"
    assert state.get_cursor("wx-1", "get_updates_buf") == "cursor-1"
    state.close()


def test_changed_weixin_identity_relogin_resets_dynamic_state_and_permission(
    tmp_path: Path, workspace: Path
) -> None:
    username = "channel:weixin:wx-1:bot_token"
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username=username,
                    metadata={"ilink_bot_id": "old-bot", "ilink_user_id": "old-user"},
                    permission_mode="auto_approve",
                )
            ]
        ),
    )
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    state.set_owner("wx-1", "owner-1")
    state.set_cursor("wx-1", "get_updates_buf", "old-cursor")
    state.close()
    channels = _channels(tmp_path, workspace, FakeCredentialStore())

    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    result = asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    assert result.status == "confirmed"
    instance = load_channel_settings(tmp_path / "channels.json").instances[0]
    assert instance.permission_mode == "request_approval"
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    assert state.get_owner("wx-1") is None
    assert state.get_cursor("wx-1", "get_updates_buf") is None
    assert state.get_pairing_status("wx-1")["state"] == "pending"
    state.close()


def test_confirmed_login_does_not_overwrite_invalid_settings_or_leave_token(
    tmp_path: Path, workspace: Path
) -> None:
    raw = '{"version":999,"instances":[]}'
    path = tmp_path / "channels.json"
    path.write_text(raw, encoding="utf-8")
    credentials = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, credentials)
    username = "channel:weixin:wx-1:bot_token"

    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    with pytest.raises(ChannelSettingsError, match="version"):
        asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    assert path.read_text(encoding="utf-8") == raw
    assert credentials.get_password(KEYRING_SERVICE_NAME, username) is None


def test_workspace_change_resets_permanent_and_temporary_auto_approve(
    tmp_path: Path, workspace: Path
) -> None:
    store = FakeCredentialStore()
    channels = _channels(tmp_path, workspace, store)
    asyncio.run(channels.start_weixin_qr_login(workspace_root=workspace, instance_id="wx-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))
    asyncio.run(channels.poll_weixin_qr_login(instance_id="wx-1", qrcode_id="qr-1"))

    settings = load_channel_settings(tmp_path / "channels.json")
    item = settings.instances[0]
    save_channel_settings(
        tmp_path / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id=item.id,
                    platform=item.platform,
                    enabled=item.enabled,
                    workspace_root=item.workspace_root,
                    credential_username=item.credential_username,
                    metadata=item.metadata,
                    permission_mode="auto_approve",
                )
            ]
        ),
    )
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    state.set_temporary_permission(
        instance_id="wx-1",
        owner_sender_id="owner-1",
        workspace_root=str(workspace.resolve()),
        permission_mode="auto_approve",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    state.close()
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    updated = channels.set_workspace_root("wx-1", replacement)

    assert updated.permission_mode == "request_approval"
    state = ChannelStateStore(tmp_path / "channels.sqlite3")
    assert (
        state.get_temporary_permission(
            "wx-1",
            owner_sender_id="owner-1",
            workspace_root=str(workspace.resolve()),
        )
        is None
    )
    state.close()
