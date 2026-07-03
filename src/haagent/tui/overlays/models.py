"""
haagent/tui/models.py - 模型中心弹窗

提供当前会话模型切换和默认 profile 设置的 TUI 交互。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from haagent.app.assistant_service import AssistantModelProfile, ModelProfileConfigureRequest
from haagent.models.gateway_registry import catalog_provider_capability
from haagent.tui.design.utils import safe_summary

ModelCenterAction = Literal[
    "switch_session",
    "set_default",
    "delete_profile",
    "new_profile",
    "manual_profile",
    "test_profile",
    "refresh_catalog",
]


@dataclass(frozen=True)
class ModelCenterResult:
    action: ModelCenterAction
    profile_name: str | None = None


@dataclass(frozen=True)
class ModelCenterState:
    profiles: list[AssistantModelProfile]
    query: str = ""
    selected_index: int = 0

    @property
    def visible_profiles(self) -> list[AssistantModelProfile]:
        needle = self.query.casefold()
        if not needle:
            return self.profiles
        return [
            profile
            for profile in self.profiles
            if needle in profile.name.casefold()
            or needle in profile.provider.casefold()
            or needle in profile.model.casefold()
        ]

    @property
    def selected_profile(self) -> AssistantModelProfile | None:
        visible = self.visible_profiles
        if not visible:
            return None
        return visible[min(max(self.selected_index, 0), len(visible) - 1)]

    def with_query(self, query: str) -> ModelCenterState:
        return replace(self, query=query, selected_index=0)

    def move(self, delta: int) -> ModelCenterState:
        visible = self.visible_profiles
        if not visible:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        return replace(self, selected_index=next_index)

    def render(self) -> str:
        lines = ["模型中心", f"搜索: {self.query or '-'}", ""]
        visible = self.visible_profiles
        if not visible:
            lines.append("无匹配模型 profile")
        for index, profile in enumerate(visible):
            selected = ">" if index == min(self.selected_index, len(visible) - 1) else " "
            active = "*" if profile.active else " "
            credential = "key:ok" if profile.credential_available else "key:missing"
            capability = getattr(profile.capability, "status", "-")
            model = safe_summary(profile.model, 36)
            lines.append(f"{selected}{active} {profile.name:<16} {profile.provider:<12} {model} {credential} {capability}")
        lines.extend(["", "输入过滤  ↑/↓ 移动  Enter 切当前会话  p 默认  d 删除  n 目录新建  m 手动  r 刷新  t 测试  Esc 关闭"])
        return "\n".join(lines)


class ModelCenterOverlay(ModalScreen[ModelCenterResult | None]):
    def __init__(self, profiles: list[AssistantModelProfile]) -> None:
        super().__init__()
        self.state = ModelCenterState(profiles=profiles)

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="model-center-dialog")

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
        if key == "enter":
            event.stop()
            selected = self.state.selected_profile
            if selected is not None:
                self.dismiss(ModelCenterResult(action="switch_session", profile_name=selected.name))
            return
        if key == "p":
            event.stop()
            selected = self.state.selected_profile
            if selected is not None:
                self.dismiss(ModelCenterResult(action="set_default", profile_name=selected.name))
            return
        if key == "d":
            event.stop()
            selected = self.state.selected_profile
            if selected is not None:
                self.dismiss(ModelCenterResult(action="delete_profile", profile_name=selected.name))
            return
        if key == "n":
            event.stop()
            self.dismiss(ModelCenterResult(action="new_profile"))
            return
        if key == "m":
            event.stop()
            self.dismiss(ModelCenterResult(action="manual_profile"))
            return
        if key == "r":
            event.stop()
            self.dismiss(ModelCenterResult(action="refresh_catalog"))
            return
        if key == "t":
            event.stop()
            selected = self.state.selected_profile
            if selected is not None:
                self.dismiss(ModelCenterResult(action="test_profile", profile_name=selected.name))
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.with_query(self.state.query + event.character))

    def _set_state(self, state: ModelCenterState) -> None:
        self.state = state
        self.query_one("#model-center-dialog", Static).update(state.render())


class ModelCatalogLoadingOverlay(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("模型中心\n\n正在读取模型目录...\n\n请稍候", id="model-catalog-loading-dialog")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()


class ManualModelSetupWizard(ModalScreen[ModelProfileConfigureRequest | None]):
    fields = [
        ("name", "profile name", "例如 deepseek"),
        ("provider", "provider", "openai 或 openai-chat"),
        ("base_url", "base_url", "兼容 endpoint 基础地址"),
        ("model", "model", "模型名称"),
        ("api_key_env", "api_key_env", "例如 OPENAI_API_KEY"),
        ("credential_source", "credential_source", "keyring/env/insecure_file"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}
        self.field_index = 0
        self.awaiting_insecure_confirmation = False
        self.awaiting_api_key = False

    def compose(self) -> ComposeResult:
        yield Static(self._body_text(), id="manual-model-setup-dialog")
        yield Input(
            password=False,
            id="manual-model-input",
            placeholder=self._placeholder(),
        )

    def on_mount(self) -> None:
        self.query_one("#manual-model-input", Input).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        field_name = self._current_field_name()
        if self.awaiting_insecure_confirmation:
            if text == "YES":
                self.awaiting_insecure_confirmation = False
                self.awaiting_api_key = True
            self._reset_input()
            return
        if self.awaiting_api_key:
            request = self._request(api_key=text)
            if request is not None:
                self.dismiss(request)
            return
        if field_name is None:
            return
        self.values[field_name] = text
        self.field_index += 1
        if self.field_index < len(self.fields):
            self._reset_input()
            return
        credential_source = self.values["credential_source"]
        if credential_source == "env":
            request = self._request(api_key=None)
            if request is not None:
                self.dismiss(request)
            return
        if credential_source == "insecure_file":
            self.awaiting_insecure_confirmation = True
            self._reset_input()
            return
        self.awaiting_api_key = True
        self._reset_input()

    def _current_field_name(self) -> str | None:
        if self.field_index >= len(self.fields):
            return None
        return self.fields[self.field_index][0]

    def _placeholder(self) -> str:
        if self.awaiting_insecure_confirmation:
            return "输入 YES 继续"
        if self.awaiting_api_key:
            return "API key"
        _, label, placeholder = self.fields[self.field_index]
        return f"{label}: {placeholder}"

    def _reset_input(self) -> None:
        self.query_one("#manual-model-setup-dialog", Static).update(self._body_text())
        input_widget = self.query_one("#manual-model-input", Input)
        input_widget.password = self.awaiting_api_key
        input_widget.placeholder = self._placeholder()
        input_widget.value = ""
        input_widget.focus()

    def _body_text(self) -> str:
        lines = ["手动模型配置", ""]
        for name, label, _placeholder in self.fields:
            value = self.values.get(name, "-")
            lines.append(f"{label}: {value}")
        lines.append("")
        if self.awaiting_insecure_confirmation:
            lines.append("insecure_file 会把 API key 写入明文用户文件；必须输入 YES 才会继续。")
        elif self.awaiting_api_key:
            lines.append("输入 API key；内容会被遮蔽，不会显示在界面文本中。")
        else:
            _name, label, _placeholder = self.fields[self.field_index]
            lines.append(f"请输入 {label} 后按 Enter；Esc 关闭。")
        return "\n".join(lines)

    def _request(self, api_key: str | None) -> ModelProfileConfigureRequest | None:
        provider = self.values.get("provider", "")
        credential_source = self.values.get("credential_source", "")
        if provider not in {"openai", "openai-chat"}:
            return None
        if credential_source not in {"keyring", "env", "insecure_file"}:
            return None
        return ModelProfileConfigureRequest(
            name=self.values.get("name", ""),
            provider=provider,
            base_url=self.values.get("base_url", ""),
            model=self.values.get("model", ""),
            api_key_env=self.values.get("api_key_env", ""),
            credential_source=credential_source,
            api_key=api_key or None,
        )


class ModelSetupWizard(ModalScreen[ModelProfileConfigureRequest | None]):
    def __init__(self, providers: list[object]) -> None:
        super().__init__()
        self.providers = providers
        self.provider_index = 0
        self.model_index = 0
        self.provider_query = ""
        self.model_query = ""
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
        yield Static(self._body_text(), id="model-setup-dialog")
        yield Input(password=True, id="model-api-key", placeholder="API key")

    def on_mount(self) -> None:
        self._sync_input_state()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            if self.step == "model":
                self.step = "provider"
                self.model_query = ""
                self.model_index = 0
                self._set_body()
                self._sync_input_state()
            else:
                self.dismiss(None)
            return
        if event.key in {"up", "down"} and self.step in {"provider", "model"}:
            event.stop()
            delta = -1 if event.key == "up" else 1
            if self.step == "provider":
                self._move_provider(delta)
            else:
                self._move_model(delta)
            self._set_body()
            return
        if event.key == "backspace" and self.step in {"provider", "model"}:
            event.stop()
            if self.step == "provider":
                self.provider_query = self.provider_query[:-1]
                self.provider_index = 0
            else:
                self.model_query = self.model_query[:-1]
                self.model_index = 0
            self._set_body()
            return
        if event.key == "enter" and self.step in {"provider", "model"}:
            event.stop()
            if self.step == "provider" and self.selected_provider is None:
                return
            if self.step == "model" and self.selected_model is None:
                return
            if self.step == "provider":
                self.step = "model"
                self.model_index = 0
            else:
                self.step = "api_key"
            self._set_body()
            self._sync_input_state()
            return
        if event.character and event.character.isprintable() and self.step in {"provider", "model"}:
            event.stop()
            if self.step == "provider":
                self.provider_query += event.character
                self.provider_index = 0
            else:
                self.model_query += event.character
                self.model_index = 0
            self._set_body()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self.step != "api_key":
            return
        request = self._request(event.value)
        if request is not None:
            self.dismiss(request)

    def _body_text(self) -> str:
        provider = self.selected_provider
        model = self.selected_model
        if self.step == "provider":
            return self._provider_list_text()
        if self.step == "model":
            return self._model_list_text()
        if provider is None or model is None:
            return "模型设置\n\n没有可配置的目录模型\n\nEsc 关闭"
        env_names = list(getattr(provider, "env_names", []) or [])
        env_name = env_names[0] if env_names else _default_env_name(str(getattr(provider, "id", "MODEL")))
        if self.step == "provider":
            action = "Enter 选择 provider"
        elif self.step == "model":
            action = "Enter 选择 model"
        else:
            action = "输入 API key 后按 Enter 保存到 keyring"
        return "\n".join(
            [
                "模型设置",
                "",
                f"provider: {getattr(provider, 'name', getattr(provider, 'id', '-'))}",
                f"model: {getattr(model, 'name', getattr(model, 'id', '-'))}",
                f"api_key_env: {env_name}",
                "",
                f"{action}；Esc 关闭",
            ]
        )

    def _provider_list_text(self) -> str:
        visible = self.visible_providers
        selected_provider = self.selected_provider
        selected_name = (
            str(getattr(selected_provider, "name", getattr(selected_provider, "id", "-")))
            if selected_provider is not None
            else "-"
        )
        lines = [
            "模型设置",
            "",
            f"provider: {selected_name}",
            f"provider 搜索: {self.provider_query or '-'}",
            f"Provider {min(self.provider_index + 1, len(visible)) if visible else 0}/{len(visible)}  total:{len(self.providers)}",
            "",
        ]
        if not visible:
            lines.append("无匹配 provider")
        start, shown = _visible_window(visible, self.provider_index)
        for offset, provider in enumerate(shown):
            index = start + offset
            selected = ">" if index == self.provider_index else " "
            name = safe_summary(str(getattr(provider, "name", getattr(provider, "id", "-"))), 24)
            provider_id = safe_summary(str(getattr(provider, "id", "-")), 22)
            model_count = len(list(getattr(provider, "models", []) or []))
            capability = getattr(catalog_provider_capability(provider), "gateway_provider", "-")
            lines.append(f"{selected} {name:<24} {provider_id:<22} models:{model_count:<4} {capability}")
        lines.extend(["", "输入搜索  ↑/↓ 移动  Enter 选择 provider  Backspace 删除  Esc 关闭"])
        return "\n".join(lines)

    def _model_list_text(self) -> str:
        provider = self.selected_provider
        if provider is None:
            return "模型设置\n\n无匹配 provider\n\nEsc 返回"
        models = self.visible_models
        all_models = list(getattr(provider, "models", []) or [])
        provider_name = str(getattr(provider, "name", getattr(provider, "id", "-")))
        selected_model = self.selected_model
        selected_model_name = (
            str(getattr(selected_model, "name", getattr(selected_model, "id", "-")))
            if selected_model is not None
            else "-"
        )
        lines = [
            "模型设置",
            "",
            f"provider: {provider_name}",
            f"model: {selected_model_name}",
            f"model 搜索: {self.model_query or '-'}",
            f"模型 {min(self.model_index + 1, len(models)) if models else 0}/{len(models)}  total:{len(all_models)}",
            "",
        ]
        if not models:
            lines.append("无匹配 model")
        start, shown = _visible_window(models, self.model_index)
        for offset, model in enumerate(shown):
            index = start + offset
            selected = ">" if index == self.model_index else " "
            name = safe_summary(str(getattr(model, "name", getattr(model, "id", "-"))), 32)
            model_id = safe_summary(str(getattr(model, "id", "-")), 40)
            lines.append(f"{selected} {name:<32} {model_id}")
        lines.extend(["", "输入搜索  ↑/↓ 移动  Enter 选择 model  Backspace 删除  Esc 返回"])
        return "\n".join(lines)

    def _set_body(self) -> None:
        self.query_one("#model-setup-dialog", Static).update(self._body_text())

    def _sync_input_state(self) -> None:
        api_key_input = self.query_one("#model-api-key", Input)
        api_key_input.display = self.step == "api_key"
        api_key_input.disabled = self.step != "api_key"
        if self.step == "api_key":
            api_key_input.focus()

    def _request(self, api_key: str) -> ModelProfileConfigureRequest | None:
        provider = self.selected_provider
        model = self.selected_model
        if provider is None or model is None:
            return None
        provider_id = str(getattr(provider, "id", "provider"))
        model_id = str(getattr(model, "id", "model"))
        env_names = list(getattr(provider, "env_names", []) or [])
        capability = catalog_provider_capability(provider)
        gateway_provider = capability.gateway_provider
        if not isinstance(gateway_provider, str) or not gateway_provider:
            return None
        return ModelProfileConfigureRequest(
            name=_profile_name(provider_id, model_id),
            provider=gateway_provider,
            base_url=str(getattr(provider, "api_base_url", "")),
            model=model_id,
            api_key_env=env_names[0] if env_names else _default_env_name(provider_id),
            credential_source="keyring",
            api_key=api_key,
        )

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


def _profile_name(provider_id: str, model_id: str) -> str:
    raw = f"{provider_id}-{model_id}"
    result = []
    previous_dash = False
    for character in raw.lower():
        if character.isalnum():
            result.append(character)
            previous_dash = False
        elif not previous_dash:
            result.append("-")
            previous_dash = True
    return "".join(result).strip("-")


def _visible_window(items: list[object], selected_index: int, *, page_size: int = 12) -> tuple[int, list[object]]:
    if not items:
        return 0, []
    clamped = min(max(selected_index, 0), len(items) - 1)
    start = min(max(clamped - page_size + 1, 0), max(len(items) - page_size, 0))
    return start, items[start : start + page_size]


def _default_env_name(provider_id: str) -> str:
    return provider_id.upper().replace("-", "_") + "_API_KEY"
