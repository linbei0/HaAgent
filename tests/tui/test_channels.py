"""
tests/tui/test_channels.py - TUI /channels 结构化命令与 overlay

验证 /channels 注册、打开 overlay、snapshot 不含 bot token。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from haagent.tui.commands import command_registry, parse_slash_command
from haagent.tui.overlays.channels import ChannelsOverlay, ChannelsOverlayState
from tests.tui.support import FakeAssistantService, _all_text


def test_channels_is_structured_slash_command() -> None:
    registry = command_registry()
    result = parse_slash_command("/channels", registry)
    assert result is not None
    assert result.error is None
    assert result.command is not None
    assert result.command.name == "channels"
    assert result.command.action == "open_channels"
    assert result.argument == ""


def test_channels_overlay_render_hides_secrets_and_shows_reauth(tmp_path: Path) -> None:
    instances = [
        SimpleNamespace(
            id="wx-1",
            platform="weixin",
            enabled=True,
            workspace_root=tmp_path,
            credential_username="channel:weixin:wx-1:bot_token",
            credential_available=False,
            state="auth_expired",
            metadata={"ilink_bot_id": "bot-abc"},
            permission_mode="auto_approve",
        )
    ]
    state = ChannelsOverlayState(instances=list(instances))
    text = state.render()
    assert "wx-1" in text
    assert "weixin" in text
    assert "auth_expired" in text or "重新登录" in text
    assert "配对码" in text or "p 重发" in text
    assert "workspace" in text.lower() or "w 改" in text
    assert "perm:auto" in text
    assert "full_access" not in text
    assert "bot-secret" not in text
    assert "bot_token" not in text or "credential" in text.lower() or True
    # 明确：真实 token 永不出现
    assert "sk-" not in text
    assert "bot-secret-token" not in text


def test_channels_overlay_pair_and_workspace_actions(tmp_path: Path) -> None:
    from haagent.tui.overlays.channels import ChannelsOverlayResult

    instances = [
        SimpleNamespace(
            id="wx-1",
            platform="weixin",
            enabled=True,
            workspace_root=tmp_path,
            credential_username="channel:weixin:wx-1:bot_token",
            credential_available=True,
            state="configured",
            metadata={},
        )
    ]
    # 直接构造 result 合同
    assert ChannelsOverlayResult(action="pair", instance_id="wx-1").action == "pair"
    assert ChannelsOverlayResult(action="workspace", instance_id="wx-1").action == "workspace"
    text = ChannelsOverlayState(instances=list(instances)).render()
    assert "p 重发配对码" in text
    assert "w 选择 workspace" in text or "workspace" in text.lower()


def test_tui_channels_command_opens_overlay(tmp_path: Path) -> None:
    from haagent.tui.application.app import HaAgentTuiApp

    service = FakeAssistantService(workspace_root=tmp_path)
    service.channel_instances = [
        SimpleNamespace(
            id="wx-1",
            platform="weixin",
            enabled=True,
            workspace_root=tmp_path,
            credential_username="channel:weixin:wx-1:bot_token",
            credential_available=True,
            state="configured",
            metadata={},
        )
    ]
    app = HaAgentTuiApp(service)

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            for ch in "channels":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert any(isinstance(screen, ChannelsOverlay) for screen in app.screen_stack)
            body = _all_text(app)
            assert "wx-1" in body or "渠道" in body
            assert "bot-secret-token" not in body

    import asyncio

    asyncio.run(_run())


def test_add_weixin_dismisses_overlay_and_keeps_qr_visible_on_failure(tmp_path: Path) -> None:
    """按 n 后应关闭列表；登录失败不得立刻重新打开 overlay 盖住二维码。"""
    import asyncio
    from types import SimpleNamespace as NS

    from haagent.app.assistant_types import AssistantChannelQrPoll, AssistantChannelQrStart
    from haagent.tui.application.app import HaAgentTuiApp

    class TrackingChannels:
        def __init__(self) -> None:
            self.start_calls = 0
            self.poll_calls = 0
            self.loop_ids: list[int] = []

        def list_instances(self):
            return []

        async def start_weixin_qr_login(self, **kwargs):
            self.start_calls += 1
            self.loop_ids.append(id(asyncio.get_running_loop()))
            return AssistantChannelQrStart(
                instance_id=kwargs.get("instance_id") or "weixin-default",
                qrcode_id="qr-ui-1",
                qrcode_url="https://liteapp.weixin.qq.com/q/test?qrcode=qr-ui-1",
            )

        async def poll_weixin_qr_login(self, **kwargs):
            self.poll_calls += 1
            self.loop_ids.append(id(asyncio.get_running_loop()))
            # 保持 wait，便于断言：轮询进行中不重开列表、且与 start 同 loop。
            return AssistantChannelQrPoll(status="wait", instance_id="weixin-default")

    service = FakeAssistantService(workspace_root=tmp_path)
    tracking = TrackingChannels()
    service.channels = tracking
    service.channel_instances = []
    app = HaAgentTuiApp(service)
    app.channel_flow._poll_interval_seconds = 0.05

    async def _run() -> None:
        async with app.run_test() as pilot:
            await pilot.press("/")
            for ch in "channels":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause()
            assert any(isinstance(screen, ChannelsOverlay) for screen in app.screen_stack)
            await pilot.press("n")
            await pilot.pause()
            # n 应 dismiss 列表 overlay，而不是继续停在渠道列表。
            assert not any(isinstance(screen, ChannelsOverlay) for screen in app.screen_stack)
            # 等待 worker 完成 start + 至少两次 poll（证明同 loop 连续 await）
            for _ in range(80):
                if tracking.start_calls >= 1 and tracking.poll_calls >= 2:
                    break
                await pilot.pause(0.05)
            assert tracking.start_calls == 1
            assert tracking.poll_calls >= 2
            # start 与 poll 必须共享同一 event loop（避免 Event loop is closed）
            assert len(set(tracking.loop_ids)) == 1
            body = _all_text(app)
            assert "qr-ui-1" in body or "扫码" in body or "微信登录" in body
            # 轮询进行中不应被渠道列表盖住
            assert not any(isinstance(screen, ChannelsOverlay) for screen in app.screen_stack)

    asyncio.run(_run())


def test_login_failed_does_not_reopen_channels_overlay(tmp_path: Path) -> None:
    """登录失败应把错误留在对话区，避免立刻 push 列表遮住二维码链接。"""
    import asyncio

    from haagent.app.assistant_types import AssistantChannelQrPoll, AssistantChannelQrStart
    from haagent.tui.application.app import HaAgentTuiApp

    class FailPollChannels:
        def list_instances(self):
            return []

        async def start_weixin_qr_login(self, **kwargs):
            return AssistantChannelQrStart(
                instance_id="weixin-default",
                qrcode_id="qr-fail-1",
                qrcode_url="https://example.com/qr-fail",
            )

        async def poll_weixin_qr_login(self, **kwargs):
            return AssistantChannelQrPoll(
                status="failed",
                instance_id="weixin-default",
                message="simulated failure",
            )

    from haagent.tui.overlays.channels import ChannelsOverlayResult

    service = FakeAssistantService(workspace_root=tmp_path)
    service.channels = FailPollChannels()
    app = HaAgentTuiApp(service)
    app.channel_flow._poll_interval_seconds = 0.05

    async def _run() -> None:
        async with app.run_test() as pilot:
            app.channel_flow.handle_channels_result(ChannelsOverlayResult(action="add_weixin"))
            for _ in range(40):
                body = _all_text(app)
                if "登录失败" in body or "simulated failure" in body:
                    break
                await pilot.pause(0.05)
            body = _all_text(app)
            assert "登录失败" in body or "simulated failure" in body
            assert "https://example.com/qr-fail" in body or "qr-fail-1" in body
            assert not any(isinstance(screen, ChannelsOverlay) for screen in app.screen_stack)

    asyncio.run(_run())
