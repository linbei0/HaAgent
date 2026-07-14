"""
tests/integration/cli/test_cli_gateway.py - 高级 gateway CLI 测试

验证 gateway status/run 的 workspace 要求、无渠道错误与优雅停止。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from haagent import cli
from haagent.channels.runtime import ChannelGatewayRuntime
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, save_channel_settings
from haagent.channels.state import ChannelStateStore
from tests.support.channel_adapter import FakeChannelAdapter as FakeAdapter
from tests.support.model_credentials import FakeCredentialStore


def test_gateway_status_without_instances(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    code = cli.main(["gateway", "status"])
    out = capsys.readouterr().out

    assert code == 0
    assert "instances=0" in out or "no channel" in out.lower() or "渠道" in out


def test_gateway_run_exits_nonzero_when_no_enabled_channels(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "home"
    (home / ".haagent").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    code = cli.main(["gateway", "run", "--workspace-root", str(workspace)])
    err = capsys.readouterr()
    text = err.out + err.err

    assert code != 0
    assert "enabled" in text.lower() or "渠道" in text or "channel" in text.lower()


def test_gateway_run_rejects_second_process_before_preflight_or_adapter_build(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    save_channel_settings(
        config_dir / "channels.json",
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="f1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:f1:bot_token",
                )
            ]
        ),
    )
    from haagent import cli_commands

    preflight_called = False
    adapter_build_called = False

    def _preflight(root: Path) -> tuple[bool, str]:
        nonlocal preflight_called
        preflight_called = True
        return True, "ok"

    def _build_adapters(self):
        nonlocal adapter_build_called
        adapter_build_called = True
        return []

    monkeypatch.setattr(cli_commands.GatewayInstanceLock, "acquire", lambda self: False)
    monkeypatch.setattr(cli_commands, "_gateway_model_preflight", _preflight)
    monkeypatch.setattr(ChannelGatewayRuntime, "build_adapters", _build_adapters)

    code = cli.main(["gateway", "run", "--workspace-root", str(workspace)])
    output = capsys.readouterr().out

    assert code != 0
    assert "already running" in output
    assert preflight_called is False
    assert adapter_build_called is False


def test_gateway_run_starts_fake_adapter_and_stops(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    home = tmp_path / "home"
    config_dir = home / ".haagent"
    config_dir.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)

    store = FakeCredentialStore({"channel:weixin:f1:bot_token": "tok"})
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

    started: list[str] = []
    stopped: list[str] = []

    class _TrackingFake(FakeAdapter):
        async def start(self, on_message):
            started.append(self.instance_id)
            await super().start(on_message)
            # 立即结束一轮，便于 CLI 在测试里用 short-circuit run
            await self.stop()

        async def stop(self):
            stopped.append(self.instance_id)
            await super().stop()

    # 注入：gateway run 使用 fake 适配器工厂与立刻返回的 run loop
    from haagent import cli_commands

    monkeypatch.setattr(
        ChannelGatewayRuntime,
        "_build_one",
        lambda self, cfg, **kwargs: _TrackingFake(instance_id=cfg.id),
    )
    monkeypatch.setattr(cli_commands, "KeyringCredentialStore", lambda: store)
    monkeypatch.setattr(
        cli_commands,
        "_gateway_model_preflight",
        lambda workspace_root: (True, "ok"),
    )

    async def _short_run(runtime, stop_event=None):
        await asyncio.sleep(0.05)
        await runtime.stop()
        return 0

    monkeypatch.setattr(cli_commands, "_run_gateway_until_cancelled", _short_run)

    code = cli.main(["gateway", "run", "--workspace-root", str(workspace)])
    out = capsys.readouterr().out

    assert code == 0
    assert started == ["f1"]
    assert stopped
    assert "f1" in out or "fake" in out or "gateway" in out.lower()


def test_gateway_partial_start_failure_stops_started_adapters_and_closes_state(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = tmp_path / "channels.json"
    save_channel_settings(
        config,
        ChannelSettings(
            instances=[
                ChannelInstanceConfig(
                    id="first",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:first:bot_token",
                ),
                ChannelInstanceConfig(
                    id="second",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:second:bot_token",
                ),
            ]
        ),
    )
    lifecycle: list[str] = []

    class _First(FakeAdapter):
        async def start(self, on_message):
            lifecycle.append("first:start")
            await super().start(on_message)

        async def stop(self):
            lifecycle.append("first:stop")
            await super().stop()

    class _Second(FakeAdapter):
        async def start(self, on_message):
            del on_message
            lifecycle.append("second:start")
            raise RuntimeError("second start failed")

    def factory(cfg, token, cursor, *, on_cursor_persist):
        del token, cursor, on_cursor_persist
        cls = _First if cfg.id == "first" else _Second
        return cls(instance_id=cfg.id)

    from haagent import cli_commands
    from haagent.channels import runtime as runtime_module

    original_runtime = runtime_module.ChannelGatewayRuntime

    class _TrackingRuntime(original_runtime):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _build_one(self, item, *, token, cursor, on_cursor_persist):
            return factory(
                item,
                token,
                cursor,
                on_cursor_persist=on_cursor_persist,
            )

    monkeypatch.setattr(runtime_module, "ChannelGatewayRuntime", _TrackingRuntime)
    monkeypatch.setattr(
        cli_commands,
        "KeyringCredentialStore",
        lambda: FakeCredentialStore(
            {
                "channel:weixin:first:bot_token": "tok-first",
                "channel:weixin:second:bot_token": "tok-second",
            }
        ),
    )
    monkeypatch.setattr(cli_commands, "_gateway_model_preflight", lambda root: (True, "ok"))

    with pytest.raises(RuntimeError, match="second start failed"):
        cli_commands._gateway_run_with_lock(
            workspace_root=workspace,
            config_path=config,
            state_path=tmp_path / "channels.sqlite3",
        )

    assert lifecycle == ["first:start", "second:start", "first:stop"]


def test_gateway_run_reports_reconnecting_adapter_error(capsys) -> None:
    from haagent import cli_commands

    async def _run() -> None:
        stop = asyncio.Event()

        class _Manager:
            def status(self) -> list[dict[str, str]]:
                stop.set()
                return [
                    {
                        "instance_id": "wx-1",
                        "state": "reconnecting",
                        "last_error": "temporary upstream error",
                    }
                ]

        class _Runtime:
            manager = _Manager()

            async def stop(self) -> list[str]:
                return []

        await cli_commands._run_gateway_until_cancelled(_Runtime(), stop_event=stop)

    asyncio.run(_run())
    out = capsys.readouterr().out
    assert "wx-1" in out
    assert "reconnecting" in out
    assert "temporary upstream error" in out


def test_gateway_run_skips_reconnecting_without_error(capsys) -> None:
    from haagent import cli_commands

    async def _run() -> None:
        stop = asyncio.Event()

        class _Manager:
            def status(self) -> list[dict[str, str]]:
                stop.set()
                return [
                    {
                        "instance_id": "wx-1",
                        "state": "reconnecting",
                        "last_error": "",
                    }
                ]

        class _Runtime:
            manager = _Manager()

            async def stop(self) -> list[str]:
                return []

        await cli_commands._run_gateway_until_cancelled(_Runtime(), stop_event=stop)

    asyncio.run(_run())
    out = capsys.readouterr().out
    assert "warning:" not in out
    assert "unknown error" not in out


def test_gateway_status_lists_configured_instances(tmp_path: Path, monkeypatch, capsys) -> None:
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
                    id="wx-1",
                    platform="weixin",
                    enabled=True,
                    workspace_root=workspace,
                    credential_username="channel:weixin:wx-1:bot_token",
                    metadata={"ilink_bot_id": "bot"},
                )
            ],
        ),
    )
    state = ChannelStateStore(config_dir / "channels.sqlite3")
    try:
        state.set_owner("wx-1", "owner-1")
        state.set_cursor("wx-1", "get_updates_buf", "secret-cursor-value")
        state.create_pairing_token("wx-1", "PAIRCODE1", expires_in_seconds=600)
    finally:
        state.close()
    code = cli.main(["gateway", "status"])
    out = capsys.readouterr().out
    assert code == 0
    assert "wx-1" in out
    assert "weixin" in out
    assert "owner=owner-1" in out or "owner-1" in out
    assert "pairing=pending" in out
    assert "cursor=set" in out
    assert "PAIRCODE1" not in out
    assert "secret-cursor-value" not in out
    assert "bot-secret" not in out


def test_root_help_keeps_tui_as_ordinary_entry(capsys) -> None:
    parser = cli.build_parser()
    help_text = parser.format_help()
    assert "ordinary interactive entry" in help_text
    assert "haagent" in help_text
    assert "Textual TUI" in help_text or "TUI" in help_text


def test_gateway_parser_accepts_run_and_status() -> None:
    parser = cli.build_parser()
    run = parser.parse_args(["gateway", "run", "--workspace-root", "."])
    status = parser.parse_args(["gateway", "status"])
    pair = parser.parse_args(["gateway", "pair", "--instance-id", "wx-1"])
    assert run.command == "gateway"
    assert run.gateway_action == "run"
    assert status.gateway_action == "status"
    assert pair.gateway_action == "pair"
    assert pair.instance_id == "wx-1"


def test_gateway_pair_prints_code_once(tmp_path: Path, monkeypatch, capsys) -> None:
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
    store = FakeCredentialStore({"channel:weixin:wx-1:bot_token": "tok"})
    from haagent.app import channel_usecases

    monkeypatch.setattr(channel_usecases, "KeyringCredentialStore", lambda: store)
    code = cli.main(["gateway", "pair", "--instance-id", "wx-1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "pairing_code=" in out
    assert "/pair " in out
    assert "bot-secret" not in out
    assert "tok" not in out
