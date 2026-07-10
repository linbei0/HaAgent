"""
src/haagent/tui/application/model_flow.py - TUI 模型连接与目录流程

从主 App 迁出模型目录加载、连接中心、连接创建向导、模型切换和连接测试逻辑。
所有模型调用仍经 AssistantService 转发，UI 只负责交互与展示，不直接触碰 ModelGateway。
"""

from __future__ import annotations

from typing import Any

from haagent.models.gateway_registry import catalog_provider_capability
from haagent.tui.overlays.connections import (
    ConnectionCenterOverlay,
    ConnectionCenterResult,
    ConnectionSetupResult,
    ConnectionSetupWizard,
)
from haagent.tui.overlays.modals import ConfirmModal
from haagent.tui.overlays.models import (
    ModelCatalogLoadingOverlay,
    ModelSwitchOverlay,
    ModelSwitchResult,
)


class ModelFlow:
    """封装模型目录、连接管理和模型切换的全部交互流程。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.providers_cache: list[object] | None = None

    # ── 入口动作 ─────────────────────────────────────────────────────────
    def open_models(self) -> None:
        if self._app._prompt_has_pending_text():
            return
        self._app.push_screen(ModelCatalogLoadingOverlay())
        self._app._load_model_switch_catalog()

    def open_connections(self) -> None:
        if self._app._prompt_has_pending_text():
            return
        connections = self._app.service.list_model_connections()
        if connections:
            self._app.push_screen(
                ConnectionCenterOverlay(connections),
                self.handle_connection_center_result,
            )
            return
        self._app.push_screen(ModelCatalogLoadingOverlay())
        self._app._refresh_model_catalog_and_open_connection_setup()

    # ── 连接中心结果分发 ─────────────────────────────────────────────────
    def handle_connection_center_result(self, result: ConnectionCenterResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        try:
            if result.action == "delete_connection" and result.connection_id is not None:
                self._confirm_delete_connection(result.connection_id)
                return
            if result.action == "new_connection":
                self._app.push_screen(ModelCatalogLoadingOverlay())
                self._app._refresh_model_catalog_and_open_connection_setup()
                return
            if result.action == "refresh_catalog":
                self._app._refresh_model_catalog_only()
                return
            if result.action == "test_connection" and result.connection_id is not None:
                self._app._run_model_connection_test(result.connection_id)
                return
        except Exception as error:
            self._app._conversation.append_block("Model warning", f"连接操作失败：{error}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def _confirm_delete_connection(self, connection_id: str) -> None:
        self._app.push_screen(
            ConfirmModal(
                f"删除模型连接：{connection_id}",
                "删除后会影响该连接下的默认/会话模型选择。确认删除？",
            ),
            lambda confirmed, cid=connection_id: self.handle_delete_connection_result(cid, confirmed),
        )

    def handle_delete_connection_result(self, connection_id: str, confirmed: bool | None) -> None:
        if not confirmed:
            self.open_connections()
            return
        try:
            self._app.service.delete_model_connection(connection_id)
        except Exception as error:
            self._app._conversation.append_block("Model warning", f"模型删除失败：{error}")
        else:
            self._app._conversation.append_line(f"模型连接已删除：{connection_id}")
        self._app._refresh()
        self.open_connections()

    def handle_connection_setup_result(self, result: ConnectionSetupResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        try:
            self._app.service.configure_model_connection(result.connection)
            self._app._run_model_connection_test(result.connection.id, result.test_model)
        except Exception as error:
            self._app._conversation.append_block("Model warning", f"连接配置失败：{error}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    # ── 后台目录加载（在 worker 线程内执行）─────────────────────────────
    def refresh_catalog_and_open_setup(self) -> None:
        if self.providers_cache is not None:
            self._app.call_from_thread(self.open_setup_wizard, list(self.providers_cache))
            return
        try:
            catalog = self._app.service.get_model_catalog()
            providers = list(catalog.providers)
            if not configurable_catalog_providers(providers):
                catalog = self._app.service.refresh_model_catalog()
                providers = list(catalog.providers)
        except Exception as error:
            self._app.call_from_thread(self.handle_catalog_error, error)
            return
        if configurable_catalog_providers(providers):
            self.providers_cache = providers
        self._app.call_from_thread(self.open_setup_wizard, providers)

    def load_switch_catalog(self) -> None:
        if self.providers_cache is not None:
            self._app.call_from_thread(self.open_switch_overlay, list(self.providers_cache))
            return
        try:
            catalog = self._app.service.get_model_catalog()
            providers = list(catalog.providers)
            if not providers:
                catalog = self._app.service.refresh_model_catalog()
                providers = list(catalog.providers)
        except Exception as error:
            self._app.call_from_thread(self.handle_catalog_error, error)
            return
        self.providers_cache = providers
        self._app.call_from_thread(self.open_switch_overlay, providers)

    def refresh_catalog_only(self) -> None:
        try:
            catalog = self._app.service.refresh_model_catalog()
        except Exception as error:
            self._app.call_from_thread(self.handle_catalog_error, error)
            return
        providers = list(catalog.providers)
        self.providers_cache = providers
        self._app.call_from_thread(self.handle_catalog_success, providers)

    def run_connection_test(self, connection_id: str, model: str | None = None) -> None:
        try:
            result = self._app.service.test_model_connection(connection_id, model=model)
        except Exception as error:
            self._app.call_from_thread(self.handle_catalog_error, error)
            return
        self._app.call_from_thread(self.handle_connection_test_result, result)

    # ── UI 线程回调 ──────────────────────────────────────────────────────
    def open_setup_wizard(self, providers: list[object]) -> None:
        configurable_providers = configurable_catalog_providers(providers)
        if not configurable_providers:
            self.dismiss_loading_overlay()
            self._app._conversation.append_block(
                "Model warning",
                "模型目录没有可配置模型。\n请刷新目录或检查网络；如果使用缓存，请删除损坏的 models_catalog_cache.json 后重试。",
            )
            self._app._refresh()
            self._app._defer_prompt_focus()
            return
        self.dismiss_loading_overlay()
        self._app.push_screen(
            ConnectionSetupWizard(configurable_providers),
            self.handle_connection_setup_result,
        )

    def open_switch_overlay(self, providers: list[object]) -> None:
        self.dismiss_loading_overlay()
        self._app.push_screen(
            ModelSwitchOverlay(self._app.service.list_model_connections(), providers),
            self.handle_switch_result,
        )

    def handle_switch_result(self, result: ModelSwitchResult | None) -> None:
        if result is None:
            self._app._defer_prompt_focus()
            return
        try:
            if result.action == "set_default":
                self._app.service.set_default_model_selection(result.selection)
                self._app._conversation.append_line(f"默认模型：{result.selection.model}")
            else:
                status = self._app.service.switch_current_session_model_selection(result.selection)
                model_name = status.model or result.selection.model
                self._app._conversation.append_line(f"当前会话：{model_name}")
        except Exception as error:
            self._app._conversation.append_block("Model warning", f"模型切换失败：{error}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def handle_catalog_success(self, providers: list[object]) -> None:
        self._app._conversation.append_line(f"模型目录已刷新：{len(providers)} 个 provider")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def handle_connection_test_result(self, result: object) -> None:
        status = "OK" if bool(getattr(result, "ok", False)) else "失败"
        message = str(getattr(result, "message", ""))
        self._app._conversation.append_line(f"模型连接测试 {status}: {message}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def handle_catalog_error(self, error: Exception) -> None:
        self.dismiss_loading_overlay()
        self._app._conversation.append_block("Model warning", f"模型操作失败：{error}")
        self._app._refresh()
        self._app._defer_prompt_focus()

    def dismiss_loading_overlay(self) -> None:
        if isinstance(self._app.screen, ModelCatalogLoadingOverlay):
            self._app.screen.dismiss(None)


def configurable_catalog_providers(providers: list[object]) -> list[object]:
    """只保留 runnable 且有可用模型的 provider，供连接向导展示。"""
    return [
        provider
        for provider in providers
        if getattr(catalog_provider_capability(provider), "status", None) == "runnable"
        and list(getattr(provider, "models", []) or [])
    ]
