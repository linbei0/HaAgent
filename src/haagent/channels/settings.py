"""
haagent/channels/settings.py - 用户级渠道实例配置

读写 ~/.haagent/channels.json；只保存非敏感元数据，token 走 keyring。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

SUPPORTED_PLATFORMS = frozenset({"weixin"})
SETTINGS_VERSION = 1
ChannelPermissionMode = Literal["request_approval", "auto_approve"]
_CHANNEL_PERMISSION_MODES = frozenset({"request_approval", "auto_approve"})


class ChannelSettingsError(RuntimeError):
    """渠道配置无效时显式失败。"""


@dataclass(frozen=True)
class ChannelInstanceConfig:
    id: str
    platform: str
    enabled: bool
    workspace_root: Path
    credential_username: str
    metadata: Mapping[str, str] = field(default_factory=dict)
    permission_mode: ChannelPermissionMode = "request_approval"


@dataclass(frozen=True)
class ChannelSettings:
    version: int = SETTINGS_VERSION
    instances: list[ChannelInstanceConfig] = field(default_factory=list)


def load_channel_settings(path: Path) -> ChannelSettings:
    if not path.exists():
        return ChannelSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChannelSettingsError(f"failed to read channel settings: {error}") from error
    if not isinstance(raw, dict):
        raise ChannelSettingsError("channel settings must be an object")
    version = raw.get("version")
    if version != SETTINGS_VERSION:
        raise ChannelSettingsError(f"unsupported channel settings version: {version!r}")
    instances_raw = raw.get("instances", [])
    if not isinstance(instances_raw, list):
        raise ChannelSettingsError("instances must be a list")
    seen_ids: set[str] = set()
    instances: list[ChannelInstanceConfig] = []
    for item in instances_raw:
        instances.append(_parse_instance(item, seen_ids))
    return ChannelSettings(version=version, instances=instances)


def save_channel_settings(path: Path, settings: ChannelSettings) -> None:
    payload = {
        "version": settings.version,
        "instances": [
            {
                "id": item.id,
                "platform": item.platform,
                "enabled": item.enabled,
                "workspace_root": str(item.workspace_root),
                "credential_username": item.credential_username,
                "metadata": dict(item.metadata),
                "permission_mode": item.permission_mode,
            }
            for item in settings.instances
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # 绝不写入 bot_token / context_token 等 secret。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _parse_instance(item: Any, seen_ids: set[str]) -> ChannelInstanceConfig:
    if not isinstance(item, dict):
        raise ChannelSettingsError("each instance must be an object")
    instance_id = str(item.get("id") or "").strip()
    if not instance_id:
        raise ChannelSettingsError("instance id is required")
    if instance_id in seen_ids:
        raise ChannelSettingsError(f"duplicate instance id: {instance_id}")
    seen_ids.add(instance_id)
    platform = str(item.get("platform") or "").strip()
    if platform not in SUPPORTED_PLATFORMS:
        raise ChannelSettingsError(f"unsupported platform: {platform!r}")
    workspace_raw = item.get("workspace_root")
    if not workspace_raw:
        raise ChannelSettingsError("workspace_root is required")
    workspace_root = Path(str(workspace_raw)).expanduser()
    if not workspace_root.exists() or not workspace_root.is_dir():
        raise ChannelSettingsError(f"workspace does not exist: {workspace_root}")
    credential_username = str(item.get("credential_username") or "").strip()
    if not credential_username:
        raise ChannelSettingsError("credential_username is required")
    metadata_raw = item.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        raise ChannelSettingsError("metadata must be an object")
    metadata = {str(key): str(value) for key, value in metadata_raw.items()}
    permission_mode = str(item.get("permission_mode", "request_approval") or "").strip()
    if permission_mode not in _CHANNEL_PERMISSION_MODES:
        raise ChannelSettingsError(f"unsupported permission_mode: {permission_mode!r}")
    return ChannelInstanceConfig(
        id=instance_id,
        platform=platform,
        enabled=bool(item.get("enabled", True)),
        workspace_root=workspace_root.resolve(),
        credential_username=credential_username,
        metadata=metadata,
        permission_mode=permission_mode,  # type: ignore[arg-type]
    )
