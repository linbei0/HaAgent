"""
tests/unit/channels/test_settings.py - 渠道配置加载与序列化测试

验证 channels.json 版本、platform、workspace 校验与 secret 不落盘。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from haagent.channels.settings import (
    ChannelSettingsError,
    load_channel_settings,
    save_channel_settings,
)


def test_missing_config_returns_empty_instances(tmp_path: Path) -> None:
    settings = load_channel_settings(tmp_path / "channels.json")
    assert settings.version == 1
    assert settings.instances == []


def test_invalid_version_fails(tmp_path: Path) -> None:
    path = tmp_path / "channels.json"
    path.write_text('{"version": 99, "instances": []}', encoding="utf-8")
    with pytest.raises(ChannelSettingsError, match="version"):
        load_channel_settings(path)


def test_invalid_platform_fails(tmp_path: Path, workspace: Path) -> None:
    path = tmp_path / "channels.json"
    path.write_text(
        (
            '{"version": 1, "instances": [{'
            f'"id": "a", "platform": "unknown", "enabled": true, '
            f'"workspace_root": "{workspace.as_posix()}", '
            '"credential_username": "channel:fake:a:token", "metadata": {}}]}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChannelSettingsError, match="platform"):
        load_channel_settings(path)


def test_missing_workspace_fails(tmp_path: Path) -> None:
    path = tmp_path / "channels.json"
    missing = tmp_path / "no-such-workspace"
    path.write_text(
        (
            '{"version": 1, "instances": [{'
            f'"id": "a", "platform": "weixin", "enabled": true, '
            f'"workspace_root": "{missing.as_posix()}", '
            '"credential_username": "channel:weixin:a:bot_token", "metadata": {}}]}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChannelSettingsError, match="workspace"):
        load_channel_settings(path)


def test_save_and_load_roundtrip_without_secrets(tmp_path: Path, workspace: Path) -> None:
    path = tmp_path / "channels.json"
    from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings

    settings = ChannelSettings(
        version=1,
        instances=[
            ChannelInstanceConfig(
                id="wx-1",
                platform="weixin",
                enabled=True,
                workspace_root=workspace,
                credential_username="channel:weixin:wx-1:bot_token",
                metadata={"ilink_bot_id": "bot-1", "ilink_user_id": "login-user"},
            )
        ],
    )
    save_channel_settings(path, settings)
    loaded = load_channel_settings(path)
    assert len(loaded.instances) == 1
    assert loaded.instances[0].id == "wx-1"
    raw = path.read_text(encoding="utf-8")
    assert "bot_token_value" not in raw
    assert "secret" not in raw.lower() or "credential_username" in raw


def test_permission_mode_defaults_to_safe_and_rejects_full_access(tmp_path: Path, workspace: Path) -> None:
    path = tmp_path / "channels.json"
    path.write_text(
        (
            '{"version": 1, "instances": [{'
            f'"id": "a", "platform": "weixin", "enabled": true, '
            f'"workspace_root": "{workspace.as_posix()}", '
            '"credential_username": "channel:weixin:a:bot_token", "metadata": {}}]}'
        ),
        encoding="utf-8",
    )
    assert load_channel_settings(path).instances[0].permission_mode == "request_approval"

    path.write_text(
        (
            '{"version": 1, "instances": [{'
            f'"id": "a", "platform": "weixin", "enabled": true, '
            f'"workspace_root": "{workspace.as_posix()}", '
            '"credential_username": "channel:weixin:a:bot_token", '
            '"permission_mode": "full_access", "metadata": {}}]}'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChannelSettingsError, match="permission_mode"):
        load_channel_settings(path)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root
