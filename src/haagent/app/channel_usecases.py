"""
haagent/app/channel_usecases.py - 聊天渠道配置与登录应用 Module

管理 channels.json、keyring token、QR 登录与实例生命周期；不含平台 Agent loop。
"""

from __future__ import annotations

import secrets
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

from haagent.app.assistant_context import AssistantContext
from haagent.app.assistant_types import (
    AssistantChannelInstance,
    AssistantChannelQrPoll,
    AssistantChannelQrStart,
    AssistantChannelTestResult,
    AssistantServiceError,
)
from haagent.channels.adapters.weixin.protocol import WeixinProtocolClient
from haagent.channels.adapters.weixin.types import WeixinAuthenticationExpired, WeixinProtocolError
from haagent.channels.process_lock import GatewayInstanceLock
from haagent.channels.settings import (
    ChannelInstanceConfig,
    ChannelSettings,
    load_channel_settings,
    save_channel_settings,
)
from haagent.channels.state import ChannelStateStore
from haagent.models.credentials import KEYRING_SERVICE_NAME, CredentialStore, KeyringCredentialStore
from haagent.models.model_connections import user_config_dir

ProtocolFactory = Callable[..., Any]


def channel_credential_username(platform: str, instance_id: str) -> str:
    if platform == "weixin":
        return f"channel:weixin:{instance_id}:bot_token"
    return f"channel:{platform}:{instance_id}:token"


class AssistantChannels:
    """TUI/CLI 共用的渠道配置用例；不运行 gateway poll loop。"""

    def __init__(
        self,
        context: AssistantContext,
        *,
        config_dir: Path | None = None,
        credential_store: CredentialStore | None = None,
        protocol_factory: ProtocolFactory | None = None,
    ) -> None:
        self._context = context
        self._config_dir = Path(config_dir) if config_dir is not None else user_config_dir()
        self._credential_store = credential_store or KeyringCredentialStore()
        self._protocol_factory = protocol_factory or WeixinProtocolClient
        self._pending_qr: dict[str, Any] = {}

    @property
    def config_path(self) -> Path:
        return self._config_dir / "channels.json"

    @property
    def state_path(self) -> Path:
        return self._config_dir / "channels.sqlite3"

    def list_instances(self) -> list[AssistantChannelInstance]:
        settings = load_channel_settings(self.config_path)
        return [self._to_view(item) for item in settings.instances]

    def set_enabled(self, instance_id: str, enabled: bool) -> AssistantChannelInstance:
        settings = load_channel_settings(self.config_path)
        updated: list[ChannelInstanceConfig] = []
        found: ChannelInstanceConfig | None = None
        for item in settings.instances:
            if item.id == instance_id:
                found = ChannelInstanceConfig(
                    id=item.id,
                    platform=item.platform,
                    enabled=enabled,
                    workspace_root=item.workspace_root,
                    credential_username=item.credential_username,
                    metadata=dict(item.metadata),
                    permission_mode=item.permission_mode,
                )
                updated.append(found)
            else:
                updated.append(item)
        if found is None:
            raise AssistantServiceError(f"channel instance not found: {instance_id}")
        save_channel_settings(self.config_path, ChannelSettings(version=1, instances=updated))
        return self._to_view(found)

    def delete_instance(self, instance_id: str) -> None:
        settings = load_channel_settings(self.config_path)
        remaining = [item for item in settings.instances if item.id != instance_id]
        if len(remaining) == len(settings.instances):
            raise AssistantServiceError(f"channel instance not found: {instance_id}")
        removed = next(item for item in settings.instances if item.id == instance_id)
        # 先删 keyring，再删动态 state；不触碰 HaAgent session package。
        try:
            self._credential_store.delete_password(KEYRING_SERVICE_NAME, removed.credential_username)
        except Exception as error:
            raise AssistantServiceError(f"failed to delete credential: {error}") from error
        state = ChannelStateStore(self.state_path)
        try:
            state.delete_instance_state(instance_id)
        finally:
            state.close()
        save_channel_settings(self.config_path, ChannelSettings(version=1, instances=remaining))
        self._pending_qr.pop(instance_id, None)

    async def start_weixin_qr_login(
        self,
        *,
        workspace_root: Path | None = None,
        instance_id: str = "weixin-default",
    ) -> AssistantChannelQrStart:
        root = Path(workspace_root or self._context.workspace_root).resolve()
        if not root.exists() or not root.is_dir():
            raise AssistantServiceError(f"workspace does not exist: {root}")
        # 重新开始前先清理旧 client，避免连接泄漏。
        await self.cancel_weixin_qr_login(instance_id)
        client = self._protocol_factory()
        try:
            qr = await client.get_qrcode()
        except WeixinProtocolError as error:
            try:
                await client.aclose()
            except Exception:
                pass
            raise AssistantServiceError(str(error)) from error
        except Exception:
            try:
                await client.aclose()
            except Exception:
                pass
            raise
        self._pending_qr[instance_id] = {
            "client": client,
            "workspace_root": root,
            "qrcode_id": qr.qrcode_id,
        }
        return AssistantChannelQrStart(
            instance_id=instance_id,
            qrcode_id=qr.qrcode_id,
            qrcode_url=qr.qrcode_url,
        )

    async def cancel_weixin_qr_login(self, instance_id: str) -> None:
        """终止 QR 登录并关闭 HTTP client；超时/重开/失败共用。"""
        pending = self._pending_qr.pop(instance_id, None)
        if pending is None:
            return
        client = pending.get("client")
        if client is None:
            return
        try:
            await client.aclose()
        except Exception:
            # 清理路径：关闭失败不阻断后续登录。
            pass

    async def poll_weixin_qr_login(
        self,
        *,
        instance_id: str,
        qrcode_id: str,
    ) -> AssistantChannelQrPoll:
        pending = self._pending_qr.get(instance_id)
        if pending is None or pending.get("qrcode_id") != qrcode_id:
            raise AssistantServiceError("qr login session not found; start login first")
        client = pending["client"]
        try:
            status = await client.poll_qrcode_status(qrcode_id)
        except WeixinProtocolError as error:
            await self.cancel_weixin_qr_login(instance_id)
            return AssistantChannelQrPoll(
                status="failed",
                instance_id=instance_id,
                message=str(error),
            )
        if status.status in {"expired", "failed"}:
            # 终态：清理 client，避免泄漏。
            await self.cancel_weixin_qr_login(instance_id)
            return AssistantChannelQrPoll(
                status=status.status,
                instance_id=instance_id,
                message=getattr(status, "message", "") or f"登录状态：{status.status}",
            )
        if status.status != "confirmed":
            return AssistantChannelQrPoll(status=status.status, instance_id=instance_id)
        if not status.bot_token:
            await self.cancel_weixin_qr_login(instance_id)
            return AssistantChannelQrPoll(
                status="failed",
                instance_id=instance_id,
                message="confirmed without bot_token",
            )
        settings = load_channel_settings(self.config_path)
        existing = next((item for item in settings.instances if item.id == instance_id), None)
        old_bot_id = str(existing.metadata.get("ilink_bot_id") or "") if existing else ""
        old_user_id = str(existing.metadata.get("ilink_user_id") or "") if existing else ""
        identity_changed = bool(
            existing
            and (
                (old_bot_id and status.ilink_bot_id and old_bot_id != status.ilink_bot_id)
                or (old_user_id and status.ilink_user_id and old_user_id != status.ilink_user_id)
            )
        )
        username = channel_credential_username("weixin", instance_id)
        previous_token = self._credential_store.get_password(KEYRING_SERVICE_NAME, username)
        try:
            # token 只进 keyring，永不写入 channels.json / snapshot。
            self._credential_store.set_password(KEYRING_SERVICE_NAME, username, status.bot_token)
        except Exception as error:
            await self.cancel_weixin_qr_login(instance_id)
            raise AssistantServiceError(f"failed to save credential: {error}") from error
        metadata = {
            "ilink_bot_id": status.ilink_bot_id or "",
            "ilink_user_id": status.ilink_user_id or "",
            "base_url": status.base_url or "https://ilinkai.weixin.qq.com",
        }
        config = ChannelInstanceConfig(
                id=instance_id,
                platform="weixin",
                enabled=True,
                workspace_root=(
                    existing.workspace_root if existing is not None else Path(pending["workspace_root"])
                ),
                credential_username=username,
                metadata={k: v for k, v in metadata.items() if v},
                permission_mode=(
                    "request_approval"
                    if identity_changed or existing is None
                    else existing.permission_mode
                ),
            )
        try:
            self._upsert_instance(config)
        except Exception:
            # 配置未提交时撤销本次凭据替换，避免留下无配置可引用的 secret。
            if previous_token is None:
                self._credential_store.delete_password(KEYRING_SERVICE_NAME, username)
            else:
                self._credential_store.set_password(KEYRING_SERVICE_NAME, username, previous_token)
            await self.cancel_weixin_qr_login(instance_id)
            raise
        if identity_changed:
            state = ChannelStateStore(self.state_path)
            try:
                state.reset_instance_identity(instance_id)
            finally:
                state.close()
        # 登录成功后生成一次性 8 位配对码（只展示一次，磁盘只存 hash）。
        pairing_code = self.issue_pairing_code(instance_id)
        await self.cancel_weixin_qr_login(instance_id)
        return AssistantChannelQrPoll(
            status="confirmed",
            instance_id=instance_id,
            credential_available=True,
            message="登录成功",
            pairing_code=pairing_code,
        )

    def issue_pairing_code(self, instance_id: str, *, expires_in_seconds: int = 600) -> str:
        """生成一次性配对码并写入 channels.sqlite3（仅 hash）。"""
        settings = load_channel_settings(self.config_path)
        if not any(item.id == instance_id for item in settings.instances):
            raise AssistantServiceError(f"channel instance not found: {instance_id}")
        alphabet = string.ascii_uppercase + string.digits
        code = "".join(secrets.choice(alphabet) for _ in range(8))
        state = ChannelStateStore(self.state_path)
        try:
            state.create_pairing_token(
                instance_id,
                code,
                expires_in_seconds=expires_in_seconds,
            )
        finally:
            state.close()
        return code

    def set_workspace_root(self, instance_id: str, workspace_root: Path) -> AssistantChannelInstance:
        """更新渠道实例固定 workspace；路径必须存在且为目录。"""
        root = Path(workspace_root).resolve()
        if not root.exists() or not root.is_dir():
            raise AssistantServiceError(f"workspace does not exist: {root}")
        settings = load_channel_settings(self.config_path)
        updated: list[ChannelInstanceConfig] = []
        found: ChannelInstanceConfig | None = None
        for item in settings.instances:
            if item.id == instance_id:
                found = ChannelInstanceConfig(
                    id=item.id,
                    platform=item.platform,
                    enabled=item.enabled,
                    workspace_root=root,
                    credential_username=item.credential_username,
                    metadata=dict(item.metadata),
                    # workspace 改变后，旧路径绑定的自动批准必须失效。
                    permission_mode="request_approval",
                )
                updated.append(found)
            else:
                updated.append(item)
        if found is None:
            raise AssistantServiceError(f"channel instance not found: {instance_id}")
        state = ChannelStateStore(self.state_path)
        try:
            state.clear_instance_permissions(instance_id)
        finally:
            state.close()
        save_channel_settings(self.config_path, ChannelSettings(version=1, instances=updated))
        return self._to_view(found)

    async def test_connection(self, instance_id: str) -> AssistantChannelTestResult:
        settings = load_channel_settings(self.config_path)
        inst = next((item for item in settings.instances if item.id == instance_id), None)
        if inst is None:
            raise AssistantServiceError(f"channel instance not found: {instance_id}")
        if inst.platform != "weixin":
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message=f"platform {inst.platform} not implemented for test",
            )
        token = self._credential_store.get_password(KEYRING_SERVICE_NAME, inst.credential_username)
        if not token:
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message="credential missing; re-login required",
            )
        lock = GatewayInstanceLock(self._config_dir / "gateway.lock")
        if not lock.acquire():
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message="channel gateway is already running; inspect its live status",
            )
        base_url = inst.metadata.get("base_url", "https://ilinkai.weixin.qq.com")
        client = None
        try:
            client = self._protocol_factory(base_url=base_url, bot_token=token)
            await client.notify_start()
        except WeixinAuthenticationExpired as error:
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message=f"auth expired: {error}",
            )
        except WeixinProtocolError as error:
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message=str(error),
            )
        except Exception as error:
            return AssistantChannelTestResult(
                ok=False,
                instance_id=instance_id,
                message=f"connection failed: {error}",
            )
        finally:
            if client is not None:
                try:
                    await client.notify_stop()
                except Exception:
                    # 探测清理失败不得跳过 client 关闭和进程锁释放。
                    pass
                try:
                    await client.aclose()
                except Exception:
                    pass
            lock.release()
        return AssistantChannelTestResult(
            ok=True,
            instance_id=instance_id,
            message="connection ok",
        )

    def _upsert_instance(self, config: ChannelInstanceConfig) -> None:
        settings = load_channel_settings(self.config_path)
        others = [item for item in settings.instances if item.id != config.id]
        others.append(config)
        save_channel_settings(self.config_path, ChannelSettings(version=1, instances=others))

    def _to_view(self, item: ChannelInstanceConfig) -> AssistantChannelInstance:
        available = False
        store_error = False
        try:
            token = self._credential_store.get_password(KEYRING_SERVICE_NAME, item.credential_username)
            available = bool(token)
        except Exception:
            # 凭据库故障 ≠ token 缺失；UI 应提示检查 keyring，而不是 re-login。
            store_error = True
            available = False
        if store_error:
            state = "credential_store_error"
        elif not available:
            state = "auth_expired"
        elif not item.enabled:
            state = "stopped"
        else:
            state = "configured"
        return AssistantChannelInstance(
            id=item.id,
            platform=item.platform,
            enabled=item.enabled,
            workspace_root=item.workspace_root,
            credential_username=item.credential_username,
            credential_available=available,
            state=state,
            metadata=dict(item.metadata),
            permission_mode=item.permission_mode,
        )
