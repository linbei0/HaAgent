"""
haagent/channels/runtime.py - 渠道网关生命周期组合根

装配 settings/state/adapters/manager，供 CLI gateway 与测试复用。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from haagent.channels.adapter import ChannelAdapter
from haagent.channels.interactions import InteractionBroker
from haagent.channels.manager import ChannelManager, ServiceFactory
from haagent.channels.settings import ChannelInstanceConfig, ChannelSettings, load_channel_settings
from haagent.channels.state import ChannelStateStore

CredentialStoreLike = Any


class AdapterFactory(Protocol):
    """平台 Adapter 工厂的唯一调用合同。"""

    def __call__(
        self,
        config: ChannelInstanceConfig,
        token: str,
        cursor: str,
        *,
        on_cursor_persist: Callable[[str], None] | None,
    ) -> ChannelAdapter:
        raise NotImplementedError


class ChannelGatewayRuntime:
    """网关组合根：settings/state/manager + 从配置构建 Adapter。"""

    def __init__(
        self,
        *,
        config_path: Path,
        state_path: Path,
        default_workspace_root: Path,
        service_factory: ServiceFactory,
        broker: InteractionBroker | None = None,
        credential_store: CredentialStoreLike | None = None,
        adapter_factories: dict[str, AdapterFactory] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.default_workspace_root = Path(default_workspace_root)
        self._service_factory = service_factory
        self._broker = broker or InteractionBroker()
        self._credential_store = credential_store
        self._adapter_factories = dict(adapter_factories or {})
        self._state: ChannelStateStore | None = None
        self.manager: ChannelManager | None = None
        self.settings: ChannelSettings | None = None

    @property
    def state(self) -> ChannelStateStore | None:
        return self._state

    def load(self) -> ChannelSettings:
        self.settings = load_channel_settings(self.config_path)
        self._state = ChannelStateStore(self.state_path)
        # 启动时清理过期 receipt，控制 SQLite 体积。
        self._state.purge_old_receipts(older_than_days=7)
        self.manager = ChannelManager(
            state=self._state,
            default_workspace_root=self.default_workspace_root,
            service_factory=self._service_factory,
            broker=self._broker,
            config_path=self.config_path,
        )
        # 为每个实例注册配置中的 workspace_root。
        for item in self.settings.instances:
            if item.enabled:
                self.manager.register_instance_workspace(
                    item.id,
                    item.workspace_root,
                    permission_mode=item.permission_mode,
                )
        return self.settings

    def build_adapters(self) -> list[Any]:
        """
        按 channels.json + keyring 构建已启用平台 Adapter。

        缺 token（非 fake）时显式失败；未知 platform 跳过并抛错。
        """
        if self.settings is None or self._state is None or self.manager is None:
            self.load()
        assert self.settings is not None
        assert self._state is not None
        adapters: list[Any] = []
        for item in self.settings.instances:
            if not item.enabled:
                continue
            token = self._resolve_token(item)
            if not token and item.platform != "fake":
                raise RuntimeError(f"missing credential for {item.id}; re-login via /channels")
            cursor = self._state.get_cursor(item.id, "get_updates_buf") or ""
            instance_id = item.id

            def _persist(value: str, *, _id: str = instance_id) -> None:
                self.persist_cursor(_id, value)

            adapter = self._build_one(item, token=token or "", cursor=cursor, on_cursor_persist=_persist)
            if adapter is None:
                raise RuntimeError(f"platform not supported for gateway run: {item.platform}")
            adapters.append(adapter)
        return adapters

    def _resolve_token(self, item: ChannelInstanceConfig) -> str:
        """读取 bot token；凭据库故障显式抛错，不伪装成 missing credential。"""
        if self._credential_store is None:
            return ""
        from haagent.models.credentials import KEYRING_SERVICE_NAME, CredentialError

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
    ) -> Any | None:
        if item.platform in self._adapter_factories:
            factory = self._adapter_factories[item.platform]
            return factory(item, token, cursor, on_cursor_persist=on_cursor_persist)
        if item.platform == "fake":
            from haagent.channels.adapters.fake import FakeAdapter

            return FakeAdapter(instance_id=item.id)
        if item.platform == "weixin":
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
        return None

    def persist_cursor(self, instance_id: str, cursor_value: str, *, cursor_name: str = "get_updates_buf") -> None:
        """Adapter 批次成功后持久化 transport cursor。"""
        if self._state is None:
            return
        self._state.set_cursor(instance_id, cursor_name, cursor_value)

    async def stop(self) -> list[str]:
        """停止 manager；返回 adapter/actor 关闭错误列表。"""
        errors: list[str] = []
        if self.manager is not None:
            await self.manager.sync_adapter_states()
            errors.extend(await self.manager.stop())
        if self._state is not None:
            # 关闭前再清一次，覆盖运行期间堆积的旧小票。
            self._state.purge_old_receipts(older_than_days=7)
            self._state.close()
            self._state = None
        return errors
