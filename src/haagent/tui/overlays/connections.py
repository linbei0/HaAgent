"""
haagent/tui/overlays/connections.py - 供应商连接配置弹窗

提供 /connect 使用的供应商连接管理、连接创建和测试模型选择交互。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from textual import events
from textual.app import ComposeResult
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from haagent.app.assistant_service import AssistantModelConnection, ModelConnectionConfigureRequest
from haagent.models.gateway_registry import catalog_provider_capability
from haagent.tui.design.utils import safe_summary

ConnectionCenterAction = Literal[
    "delete_connection",
    "new_connection",
    "test_connection",
    "refresh_catalog",
]


@dataclass(frozen=True)
class ConnectionCenterResult:
    action: ConnectionCenterAction
    connection_id: str | None = None


@dataclass(frozen=True)
class ConnectionSetupResult:
    connection: ModelConnectionConfigureRequest
    test_model: str


@dataclass(frozen=True)
class ConnectionCenterState:
    connections: list[AssistantModelConnection]
    query: str = ""
    selected_index: int = 0

    @property
    def visible_connections(self) -> list[AssistantModelConnection]:
        needle = self.query.casefold()
        if not needle:
            return self.connections
        return [
            connection
            for connection in self.connections
            if needle in connection.id.casefold()
            or needle in connection.name.casefold()
            or needle in connection.provider_name.casefold()
            or needle in connection.gateway_provider.casefold()
        ]

    @property
    def selected_connection(self) -> AssistantModelConnection | None:
        visible = self.visible_connections
        if not visible:
            return None
        return visible[min(max(self.selected_index, 0), len(visible) - 1)]

    def with_query(self, query: str) -> ConnectionCenterState:
        return replace(self, query=query, selected_index=0)

    def move(self, delta: int) -> ConnectionCenterState:
        visible = self.visible_connections
        if not visible:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        return replace(self, selected_index=next_index)

    def render(self) -> str:
        lines = ["供应商连接", f"搜索: {self.query or '-'}", ""]
        visible = self.visible_connections
        if not visible:
            lines.append("无匹配连接")
        for index, connection in enumerate(visible):
            selected = ">" if index == min(self.selected_index, len(visible) - 1) else " "
            credential = "key:ok" if connection.credential_available else "key:missing"
            provider = safe_summary(connection.provider_name, 24)
            lines.append(
                f"{selected} {connection.name:<16} {provider:<24} "
                f"{connection.gateway_provider:<12} {credential}"
            )
        lines.extend(["", "输入过滤  ↑/↓ 移动  d 删除连接  n 新建连接  r 刷新  t 测试  Esc 关闭"])
        return "\n".join(lines)


def _connection_option(connection: AssistantModelConnection) -> Option:
    credential = "key:ok" if connection.credential_available else "key:missing"
    provider = safe_summary(connection.provider_name, 24)
    return Option(
        f"{connection.name:<16} {provider:<24} {connection.gateway_provider:<12} {credential}",
        id=connection.id,
    )


class ConnectionCenterOverlay(ModalScreen[ConnectionCenterResult | None]):
    def __init__(self, connections: list[AssistantModelConnection]) -> None:
        super().__init__()
        self.state = ConnectionCenterState(connections=connections)

    def compose(self) -> ComposeResult:
        yield Static("", id="connection-center-dialog")
        yield OptionList(id="connection-center-list")

    def on_mount(self) -> None:
        self._set_state(self.state)
        self.query_one(OptionList).focus()

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if key == "backspace":
            event.stop()
            self._set_state(self.state.with_query(self.state.query[:-1]))
            return
        if key == "d":
            event.stop()
            self._sync_selected_from_option_list()
            selected = self.state.selected_connection
            if selected is not None:
                self.dismiss(ConnectionCenterResult(action="delete_connection", connection_id=selected.id))
            return
        if key == "n":
            event.stop()
            self.dismiss(ConnectionCenterResult(action="new_connection"))
            return
        if key == "r":
            event.stop()
            self.dismiss(ConnectionCenterResult(action="refresh_catalog"))
            return
        if key == "t":
            event.stop()
            self._sync_selected_from_option_list()
            selected = self.state.selected_connection
            if selected is not None:
                self.dismiss(ConnectionCenterResult(action="test_connection", connection_id=selected.id))
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.with_query(self.state.query + event.character))

    def _set_state(self, state: ConnectionCenterState) -> None:
        self.state = state
        try:
            body = self.query_one("#connection-center-dialog", Static)
            option_list = self.query_one(OptionList)
        except NoMatches:
            return
        body.update("\n".join(["供应商连接", f"搜索: {state.query or '-'}", ""]))
        options = [_connection_option(connection) for connection in state.visible_connections]
        if not options:
            options = [Option("无匹配连接", id="empty", disabled=True)]
        option_list.set_options(options)
        option_list.highlighted = state.selected_index if state.visible_connections else None

    def _sync_selected_from_option_list(self) -> None:
        if not self.is_mounted:
            return
        index = self.query_one(OptionList).highlighted
        if index is None:
            return
        self.state = replace(self.state, selected_index=index)


class ConnectionSetupWizard(ModalScreen[ConnectionSetupResult | None]):
    def __init__(self, providers: list[object]) -> None:
        super().__init__()
        self.providers = providers
        self.provider_index = 0
        self.model_index = 0
        self.provider_query = ""
        self.model_query = ""
        self.connection_name = ""
        self.input_error = ""
        self.step = "provider"

    @property
    def selected_provider(self):
        providers = self.visible_providers
        return providers[self.provider_index] if providers else None

    @property
    def selected_model(self):
        models = self.visible_models
        return models[self.model_index] if models else None

    @property
    def visible_providers(self) -> list[object]:
        if not self.provider_query:
            return self.providers
        needle = self.provider_query.casefold()
        return [
            provider
            for provider in self.providers
            if needle in str(getattr(provider, "id", "")).casefold()
            or needle in str(getattr(provider, "name", "")).casefold()
        ]

    @property
    def visible_models(self) -> list[object]:
        provider = self.selected_provider
        models = list(getattr(provider, "models", []) if provider is not None else [])
        if not self.model_query:
            return models
        needle = self.model_query.casefold()
        return [
            model
            for model in models
            if needle in str(getattr(model, "id", "")).casefold()
            or needle in str(getattr(model, "name", "")).casefold()
            or needle in str(getattr(model, "family", "")).casefold()
        ]

    def compose(self) -> ComposeResult:
        # provider/model 选择用 OptionList；secret 单独用 password Input，
        # 真实 API key 只进安全输入，绝不进入 OptionList 或状态文本。
        yield Static("", id="connection-setup-dialog")
        yield OptionList(id="connection-setup-list")
        yield Input(password=True, id="connection-secret", placeholder="API key")

    def on_mount(self) -> None:
        self._set_step_ui()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            if self.step == "model":
                self.step = "provider"
                self.model_query = ""
                self.model_index = 0
                self._set_step_ui()
            else:
                self.dismiss(None)
            return
        if self.step not in {"provider", "model"}:
            return
        if event.key in {"up", "down"}:
            event.stop()
            self._move_selection(-1 if event.key == "up" else 1)
            return
        if event.key == "backspace":
            event.stop()
            self._trim_query()
            return
        if event.key == "enter":
            event.stop()
            self._advance_from_list()
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._extend_query(event.character)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # 鼠标点击选项等价于回车确认当前 step。
        if self.step not in {"provider", "model"}:
            return
        event.stop()
        self._sync_index_from_list()
        self._advance_from_list()

    def _move_selection(self, delta: int) -> None:
        if self.step == "provider":
            self._move_provider(delta)
        else:
            self._move_model(delta)
        self._set_step_ui()

    def _trim_query(self) -> None:
        if self.step == "provider":
            self.provider_query = self.provider_query[:-1]
            self.provider_index = 0
        else:
            self.model_query = self.model_query[:-1]
            self.model_index = 0
        self._set_step_ui()

    def _extend_query(self, character: str) -> None:
        if self.step == "provider":
            self.provider_query += character
            self.provider_index = 0
        else:
            self.model_query += character
            self.model_index = 0
        self._set_step_ui()

    def _advance_from_list(self) -> None:
        if self.step == "provider" and self.selected_provider is not None:
            self.step = "model"
            self.model_index = 0
            self.model_query = ""
        elif self.step == "model" and self.selected_model is not None:
            self.step = "connection_name"
        self._set_step_ui()

    def _sync_index_from_list(self) -> None:
        try:
            index = self.query_one("#connection-setup-list", OptionList).highlighted
        except NoMatches:
            return
        if index is None:
            return
        if self.step == "provider":
            self.provider_index = index
        else:
            self.model_index = index

    def _set_step_ui(self) -> None:
        try:
            body = self.query_one("#connection-setup-dialog", Static)
            option_list = self.query_one("#connection-setup-list", OptionList)
            secret_input = self.query_one("#connection-secret", Input)
        except NoMatches:
            return
        body.update(self._header_text())
        list_active = self.step in {"provider", "model"}
        option_list.display = list_active
        secret_input.display = not list_active
        secret_input.disabled = list_active
        if list_active:
            self._populate_option_list(option_list)
            option_list.focus()
            return
        secret_input.password = self.step == "api_key"
        secret_input.placeholder = "连接名" if self.step == "connection_name" else "API key"
        secret_input.value = ""
        secret_input.focus()

    def _populate_option_list(self, option_list: OptionList) -> None:
        if self.step == "provider":
            items = self.visible_providers
            selected_index = self.provider_index
            options = [_provider_option(provider) for provider in items]
        else:
            items = self.visible_models
            selected_index = self.model_index
            options = [_model_option(model) for model in items]
        if not options:
            empty = "无匹配 provider" if self.step == "provider" else "无匹配 model"
            option_list.set_options([Option(empty, id="empty", disabled=True)])
            option_list.highlighted = None
            return
        option_list.set_options(options)
        option_list.highlighted = min(max(selected_index, 0), len(items) - 1)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self.step == "connection_name":
            if self._accept_connection_name(event.value):
                self.step = "api_key"
            self._set_step_ui()
            return
        if self.step != "api_key":
            return
        result = self._new_connection_result(event.value)
        if result is not None:
            self.dismiss(result)
            return
        self.input_error = "API key 不能为空"
        self._set_step_ui()

    def _header_text(self) -> str:
        if self.step == "provider":
            visible = self.visible_providers
            selected_provider = self.selected_provider
            selected_name = _entity_name(selected_provider)
            return "\n".join(
                [
                    "连接配置",
                    "",
                    f"provider: {selected_name}",
                    f"provider 搜索: {self.provider_query or '-'}",
                    f"Provider {min(self.provider_index + 1, len(visible)) if visible else 0}/{len(visible)}  total:{len(self.providers)}",
                    "",
                    "输入搜索  ↑/↓ 移动  Enter 选择 provider  Backspace 删除  Esc 关闭",
                ]
            )
        if self.step == "model":
            provider = self.selected_provider
            models = self.visible_models
            all_models = list(getattr(provider, "models", []) or []) if provider is not None else []
            return "\n".join(
                [
                    "连接配置",
                    "",
                    f"provider: {_entity_name(provider)}",
                    f"test_model: {_entity_name(self.selected_model)}",
                    f"model 搜索: {self.model_query or '-'}",
                    f"模型 {min(self.model_index + 1, len(models)) if models else 0}/{len(models)}  total:{len(all_models)}",
                    "",
                    "输入搜索  ↑/↓ 移动  Enter 选择测试模型  Backspace 删除  Esc 返回",
                ]
            )
        provider = self.selected_provider
        model = self.selected_model
        if provider is None or model is None:
            return "连接配置\n\n没有可配置的目录模型\n\nEsc 关闭"
        env_names = list(getattr(provider, "env_names", []) or [])
        env_name = env_names[0] if env_names else _default_env_name(str(getattr(provider, "id", "MODEL")))
        action = "输入连接名后按 Enter" if self.step == "connection_name" else "输入 API key 后按 Enter"
        lines = [
            "连接配置",
            "",
            f"provider: {_entity_name(provider)}",
            f"test_model: {_entity_name(model)}",
            f"connection: {self.connection_name or '-'}",
            f"api_key_env: {env_name}",
            "",
        ]
        if self.input_error:
            lines.extend([self.input_error, ""])
        lines.append(f"{action}；Esc 关闭")
        return "\n".join(lines)

    def _new_connection_result(self, api_key: str) -> ConnectionSetupResult | None:
        provider = self.selected_provider
        model = self.selected_model
        if provider is None or model is None:
            return None
        api_key = api_key.strip()
        if not api_key:
            return None
        provider_id = str(getattr(provider, "id", "provider"))
        env_names = list(getattr(provider, "env_names", []) or [])
        capability = catalog_provider_capability(provider)
        gateway_provider = capability.gateway_provider
        if not isinstance(gateway_provider, str) or not gateway_provider:
            return None
        connection_id = _connection_id(provider_id, self.connection_name)
        return ConnectionSetupResult(
            connection=ModelConnectionConfigureRequest(
                id=connection_id,
                name=self.connection_name,
                provider_id=provider_id,
                provider_name=str(getattr(provider, "name", provider_id)),
                gateway_provider=gateway_provider,
                base_url=str(getattr(provider, "api_base_url", "")),
                api_key_env=env_names[0] if env_names else _default_env_name(provider_id),
                credential_source="keyring",
                api_key=api_key,
            ),
            test_model=str(getattr(model, "id", "model")),
        )

    def _accept_connection_name(self, value: str) -> bool:
        name = value.strip()
        self.input_error = ""
        if not name:
            return False
        if _looks_like_secret(name):
            self.input_error = "连接名不能是 API key，请输入 personal、work 等名称"
            return False
        self.connection_name = name
        return True

    def _move_provider(self, delta: int) -> None:
        providers = self.visible_providers
        if not providers:
            self.provider_index = 0
            self.model_index = 0
            return
        next_index = min(max(self.provider_index + delta, 0), len(providers) - 1)
        if next_index != self.provider_index:
            self.provider_index = next_index
            self.model_index = 0

    def _move_model(self, delta: int) -> None:
        models = self.visible_models
        if not models:
            self.model_index = 0
            return
        self.model_index = min(max(self.model_index + delta, 0), len(models) - 1)


def _connection_id(provider_id: str, connection_name: str) -> str:
    raw = f"{provider_id}-{connection_name}"
    result = []
    previous_dash = False
    for character in raw.lower():
        if character.isalnum():
            result.append(character)
            previous_dash = False
        elif not previous_dash:
            result.append("-")
            previous_dash = True
    return "".join(result).strip("-") or "provider-connection"


def _entity_name(entity: object) -> str:
    if entity is None:
        return "-"
    return str(getattr(entity, "name", getattr(entity, "id", "-")))


def _provider_option(provider: object) -> Option:
    name = safe_summary(_entity_name(provider), 24)
    provider_id = safe_summary(str(getattr(provider, "id", "-")), 22)
    model_count = len(list(getattr(provider, "models", []) or []))
    capability = getattr(catalog_provider_capability(provider), "gateway_provider", "-")
    return Option(
        f"{name:<24} {provider_id:<22} models:{model_count:<4} {capability}",
        id=str(getattr(provider, "id", "-")),
    )


def _model_option(model: object) -> Option:
    name = safe_summary(_entity_name(model), 32)
    model_id = safe_summary(str(getattr(model, "id", "-")), 40)
    return Option(f"{name:<32} {model_id}", id=str(getattr(model, "id", "-")))


def _default_env_name(provider_id: str) -> str:
    return f"{provider_id.upper().replace('-', '_')}_API_KEY"


def _looks_like_secret(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized.startswith(("sk-", "sk_", "sess-", "key-", "api-"))
        or "api_key" in normalized
        or len(normalized) >= 32
    )
