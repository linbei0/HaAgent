"""
haagent/channels/runtime.py - 渠道网关生命周期组合根

装配 settings/state/adapters/manager，供 CLI gateway 与测试复用。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from haagent.channels.adapter import ChannelAdapter
from haagent.channels.manager import ChannelManager, ServiceFactory
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, load_channel_settings
from haagent.channels.state import ChannelStateStore

class ChannelGatewayRuntime:
    """网关组合根：settings/state/manager + 从配置构建 Adapter。"""

    def __init__(
        self,
        *,
        config_path: Path,
        state_path: Path,
        default_workspace_root: Path,
        service_factory: ServiceFactory,
        credential_store: Any | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._state_path = Path(state_path)
        self._default_workspace_root = Path(default_workspace_root)
        self._service_factory = service_factory
        self._credential_store = credential_store
        self._state: ChannelStateStore | None = None
        self.manager: ChannelManager | None = None
        self._settings: ChannelSettings | None = None

    def load(self) -> ChannelSettings:
        self._settings = load_channel_settings(self._config_path)
        self._state = ChannelStateStore(self._state_path)
        # 启动时清理过期 receipt，控制 SQLite 体积。
        self._state.purge_old_receipts(older_than_days=7)
        self.manager = ChannelManager(
            state=self._state,
            default_workspace_root=self._default_workspace_root,
            service_factory=self._service_factory,
            config_path=self._config_path,
        )
        # 为每个实例注册配置中的 workspace_root。
        for item in self._settings.instances:
            if item.enabled:
                self.manager.register_instance_workspace(
                    item.id,
                    item.workspace_root,
                    permission_mode=item.permission_mode,
                )
        return self._settings

    def build_adapters(self) -> list[ChannelAdapter]:
        """
        按 channels.json + keyring 构建已启用平台 Adapter。

        缺 token 时显式失败。
        """
        if self._settings is None or self._state is None or self.manager is None:
            self.load()
        assert self._settings is not None
        assert self._state is not None
        adapters: list[ChannelAdapter] = []
        for item in self._settings.instances:
            if not item.enabled:
                continue
            token = self._resolve_token(item)
            if not token:
                raise RuntimeError(f"missing credential for {item.id}; re-login via /channels")
            cursor = self._state.get_cursor(item.id, "get_updates_buf") or ""
            instance_id = item.id
            state = self._state

            def _persist(value: str, *, _id: str = instance_id) -> None:
                state.set_cursor(_id, "get_updates_buf", value)

            adapter = self._build_one(item, token=token, cursor=cursor, on_cursor_persist=_persist)
            adapters.append(adapter)
        return adapters

    def _resolve_token(self, item: ChannelInstanceConfig) -> str:
        """读取 bot token；凭据库故障显式抛错，不伪装成 missing credential。"""
        if self._credential_store is None:
            return ""
        from haagent.models.config.credentials import KEYRING_SERVICE_NAME, CredentialError

        try:
            return self._credential_store.get_password(KEYRING_SERVICE_NAME, item.credential_username) or ""
        except CredentialError as error:
            # keyring 锁定/损坏：重新登录无效，必须暴露根因。
            raise RuntimeError(f"credential store failed for {item.id}: {error}") from error
        except Exception as error:
            raise RuntimeError(f"credential store failed for {item.id}: {error}") from error

    def _build_one(
        self,
        item: ChannelInstanceConfig,
        *,
        token: str,
        cursor: str,
        on_cursor_persist: Callable[[str], None] | None,
    ) -> ChannelAdapter:
        from haagent.channels.adapters.weixin.adapter import WeixinAdapter
        from haagent.channels.adapters.weixin.protocol import WeixinProtocolClient

        base_url = item.metadata.get("base_url", "https://ilinkai.weixin.qq.com")
        protocol = WeixinProtocolClient(base_url=base_url, bot_token=token)
        return WeixinAdapter(
            instance_id=item.id,
            protocol=protocol,
            initial_cursor=cursor,
            on_cursor_persist=on_cursor_persist,
        )

    async def stop(self) -> list[str]:
        """停止 manager；返回 adapter/actor 关闭错误列表。"""
        errors: list[str] = []
        if self.manager is not None:
            errors.extend(await self.manager.stop())
        if self._state is not None:
            # 关闭前再清一次，覆盖运行期间堆积的旧小票。
            self._state.purge_old_receipts(older_than_days=7)
            self._state.close()
            self._state = None
        return errors
