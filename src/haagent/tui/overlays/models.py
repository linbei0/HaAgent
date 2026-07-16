"""
haagent/tui/overlays/models.py - 模型切换弹窗

提供 /model 使用的已配置连接与目录模型组合选择交互。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Callable, Literal

from textual import events
from textual.app import ComposeResult, ScreenStackError
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.app.assistant_types import AssistantModelConnection, ModelSelectionRequest
from haagent.tui.design.utils import safe_summary
from haagent.models.local_runtime import LocalRuntimeDiscovery, LocalRuntimeModel

ModelSwitchAction = Literal[
    "switch_current",
    "set_default",
    "set_fallback",
    "set_fallback_cloud",
    "scan_local",
]
MODEL_SWITCH_PAGE_SIZE = 15


@dataclass(frozen=True)
class ModelSwitchRow:
    connection_id: str
    connection_name: str
    provider_id: str
    provider_name: str
    model: str
    model_name: str


@dataclass(frozen=True)
class ModelSwitchResult:
    action: ModelSwitchAction
    selection: ModelSelectionRequest | None = None


@dataclass(frozen=True)
class ModelSwitchState:
    connections: list[AssistantModelConnection]
    providers: list[object]
    query: str = ""
    selected_index: int = 0
    # None：模型列表；非 None：variant 子步骤（已选模型 + 可选 variants）
    variant_step: ModelSwitchRow | None = None
    # None 表示 TUI 中的「默认」，即不传 variant。
    variant_options: tuple[str | None, ...] = ()
    variant_selected_index: int = 0
    variant_error: str | None = None

    @cached_property
    def rows(self) -> list[ModelSwitchRow]:
        providers_by_id = {str(getattr(provider, "id", "")): provider for provider in self.providers}
        rows: list[ModelSwitchRow] = []
        for connection in self.connections:
            provider = providers_by_id.get(connection.provider_id)
            if provider is None:
                continue
            for model in list(getattr(provider, "models", []) or []):
                model_id = str(getattr(model, "id", ""))
                if not model_id:
                    continue
                rows.append(
                    ModelSwitchRow(
                        connection_id=connection.id,
                        connection_name=connection.name,
                        provider_id=connection.provider_id,
                        provider_name=connection.provider_name,
                        model=model_id,
                        model_name=str(getattr(model, "name", model_id)),
                    )
                )
        return rows

    @cached_property
    def visible_rows(self) -> list[ModelSwitchRow]:
        needle = self.query.casefold()
        if not needle:
            return self.rows
        return [
            row
            for row in self.rows
            if needle in row.provider_name.casefold()
            or needle in row.connection_name.casefold()
            or needle in row.connection_id.casefold()
            or needle in row.model.casefold()
            or needle in row.model_name.casefold()
        ]

    @property
    def selected_row(self) -> ModelSwitchRow | None:
        visible = self.visible_rows
        if not visible:
            return None
        return visible[min(max(self.selected_index, 0), len(visible) - 1)]

    @property
    def visible_window(self) -> tuple[int, list[ModelSwitchRow]]:
        visible = self.visible_rows
        if not visible:
            return 0, []
        selected = min(max(self.selected_index, 0), len(visible) - 1)
        start = max(0, min(selected - MODEL_SWITCH_PAGE_SIZE + 1, len(visible) - MODEL_SWITCH_PAGE_SIZE))
        return start, visible[start : start + MODEL_SWITCH_PAGE_SIZE]

    def with_query(self, query: str) -> ModelSwitchState:
        return replace(self, query=query, selected_index=0, variant_step=None, variant_options=(), variant_selected_index=0)

    def move(self, delta: int) -> ModelSwitchState:
        if self.variant_step is not None:
            options = self.variant_options
            if not options:
                return replace(self, variant_selected_index=0)
            next_index = min(max(self.variant_selected_index + delta, 0), len(options) - 1)
            return replace(self, variant_selected_index=next_index)
        visible = self.visible_rows
        if not visible:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        return replace(self, selected_index=next_index)

    def enter_variant_step(self, row: ModelSwitchRow, variant_names: list[str]) -> ModelSwitchState:
        return replace(
            self,
            variant_step=row,
            variant_options=(None, *variant_names),
            variant_selected_index=0,
        )

    def exit_variant_step(self) -> ModelSwitchState:
        return replace(self, variant_step=None, variant_options=(), variant_selected_index=0)

    @property
    def selected_variant(self) -> str | None:
        assert self.variant_step is not None and self.variant_options
        index = min(max(self.variant_selected_index, 0), len(self.variant_options) - 1)
        return self.variant_options[index]

    def render(self) -> str:
        if self.variant_step is not None:
            return self._render_variant_step()
        lines = ["模型切换", f"搜索: {self.query or '-'}", ""]
        if not self.connections:
            lines.extend(["请先 /connect 配置供应商连接", "", "Esc 关闭"])
            return "\n".join(lines)
        visible = self.visible_rows
        if not visible:
            lines.append("没有可切换的模型；请先 /connect 配置连接或刷新模型目录")
        else:
            selected_index = min(max(self.selected_index, 0), len(visible) - 1)
            lines.append(f"模型 {selected_index + 1}/{len(visible)}  total:{len(self.rows)}")
        start, shown = self.visible_window
        for offset, row in enumerate(shown):
            index = start + offset
            selected = ">" if index == min(max(self.selected_index, 0), len(visible) - 1) else " "
            provider_connection = safe_summary(f"{row.provider_name} / {row.connection_name}", 34)
            model = safe_summary(row.model, 48)
            lines.append(f"{selected} {provider_connection:<34} {model}")
        lines.extend(["", "输入过滤  ↑/↓ 移动  Enter 当前会话  p 默认  b 备用  c 云端备用  l 扫描本机  Esc 关闭"])
        if self.variant_error:
            lines.extend(["", f"模型参数配置错误：{self.variant_error}"])
        return "\n".join(lines)

    def _render_variant_step(self) -> str:
        row = self.variant_step
        assert row is not None
        lines = [
            "模型参数变体",
            f"{row.provider_name} / {row.connection_name} · {row.model}",
            "",
        ]
        for index, option in enumerate(self.variant_options):
            selected = ">" if index == self.variant_selected_index else " "
            lines.append(f"{selected} {option or '默认'}")
        lines.extend(["", "↑/↓ 移动  Enter 当前会话  p 默认  b 备用  c 云端备用  Esc 返回"])
        return "\n".join(lines)


class ModelSwitchOverlay(ModalScreen[ModelSwitchResult | None]):
    def __init__(
        self,
        connections: list[AssistantModelConnection],
        providers: list[object],
        *,
        variant_lookup: Callable[[str, str], list[str]],
    ) -> None:
        super().__init__()
        self.state = ModelSwitchState(connections=connections, providers=providers)
        self._variant_lookup = variant_lookup
        self._pending_action: ModelSwitchAction = "switch_current"

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="model-switch-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key == "escape":
            event.stop()
            if self.state.variant_step is not None:
                self._set_state(self.state.exit_variant_step())
                return
            self._safe_dismiss(None)
            return
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return
        if self.state.variant_step is not None:
            if key == "enter":
                event.stop()
                # Enter 完成进入子步骤时的原始动作（当前会话/默认/备用）。
                self._dismiss_selection(self._pending_action)
                return
            if key == "p":
                event.stop()
                self._dismiss_selection("set_default")
                return
            if key == "b":
                event.stop()
                self._dismiss_selection("set_fallback")
                return
            if key == "c":
                event.stop()
                self._dismiss_selection("set_fallback_cloud")
                return
            return
        if key == "backspace":
            event.stop()
            self._set_state(self.state.with_query(self.state.query[:-1]))
            return
        if key == "enter":
            event.stop()
            self._maybe_enter_variant_or_dismiss("switch_current")
            return
        if key == "p":
            event.stop()
            self._maybe_enter_variant_or_dismiss("set_default")
            return
        if key == "b":
            event.stop()
            self._maybe_enter_variant_or_dismiss("set_fallback")
            return
        if key == "c":
            event.stop()
            self._maybe_enter_variant_or_dismiss("set_fallback_cloud")
            return
        if key == "l":
            event.stop()
            self._safe_dismiss(ModelSwitchResult(action="scan_local"))
            return
        if event.character and event.character.isprintable():
            event.stop()
            self._set_state(self.state.with_query(self.state.query + event.character))

    def _maybe_enter_variant_or_dismiss(self, action: ModelSwitchAction) -> None:
        row = self.state.selected_row
        if row is None:
            return
        try:
            variants = self._variant_lookup(row.connection_id, row.model)
        except Exception as error:
            # 配置错误必须留在模型选择界面，不能当作无 variant 继续发送。
            self._set_state(replace(self.state, variant_error=str(error)))
            return
        # 仅当配置了 variants 时进入子步骤；首项始终为「默认」。
        if variants:
            self._set_state(self.state.enter_variant_step(row, variants))
            self._pending_action = action
            return
        self._dismiss_selection(action, row=row, variant=None)

    def _dismiss_selection(
        self,
        action: ModelSwitchAction,
        *,
        row: ModelSwitchRow | None = None,
        variant: str | None = None,
    ) -> None:
        if self.state.variant_step is not None:
            row = self.state.variant_step
            variant = self.state.selected_variant
        if row is None:
            row = self.state.selected_row
        if row is None:
            return
        self._safe_dismiss(
            ModelSwitchResult(
                action=action,
                selection=ModelSelectionRequest(
                    connection_id=row.connection_id,
                    model=row.model,
                    variant=variant,
                ),
            )
        )

    def _set_state(self, state: ModelSwitchState) -> None:
        self.state = state
        self.query_one("#model-switch-dialog", Static).update(state.render())

    def _safe_dismiss(self, result: ModelSwitchResult | None) -> None:
        try:
            if self.app.screen is not self:
                return
            self.dismiss(result)
        except ScreenStackError:
            return


class ModelCatalogLoadingOverlay(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("模型目录\n\n正在读取模型目录...\n\n请稍候", id="model-catalog-loading-dialog")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()


@dataclass(frozen=True)
class LocalModelSelection:
    discovery: LocalRuntimeDiscovery
    model: LocalRuntimeModel


class LocalRuntimeOverlay(ModalScreen[LocalModelSelection | None]):
    def __init__(self, discoveries: tuple[LocalRuntimeDiscovery, ...]) -> None:
        super().__init__()
        self.discoveries = discoveries
        self.rows = [
            LocalModelSelection(discovery, model)
            for discovery in discoveries
            if discovery.status == "available"
            for model in discovery.models
        ]
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield Static(self._render_text(), id="local-runtime-dialog")

    def _render_text(self) -> str:
        lines = ["本地模型运行时", ""]
        for discovery in self.discoveries:
            if discovery.status != "available":
                lines.append(f"{discovery.runtime_kind}: {discovery.status} - {discovery.reason or ''}")
        for index, row in enumerate(self.rows):
            caps = row.model.capabilities
            selected = ">" if index == self.selected_index else " "
            lines.append(
                f"{selected} {row.discovery.runtime_kind} / {row.model.id}  "
                f"loaded={'yes' if row.model.loaded else 'no'}  context={caps.context_window_tokens or '?'}  "
                f"tools={caps.tools_mode}  vision={caps.vision}  reasoning={caps.reasoning}"
            )
        if not self.rows:
            lines.append("未发现可用的本地聊天模型")
        lines.extend(["", "↑/↓ 移动  Enter 保存并切换  Esc 关闭"])
        return "\n".join(lines)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
        elif event.key in {"up", "down"} and self.rows:
            event.stop()
            delta = -1 if event.key == "up" else 1
            self.selected_index = min(max(self.selected_index + delta, 0), len(self.rows) - 1)
            self.query_one("#local-runtime-dialog", Static).update(self._render_text())
        elif event.key == "enter" and self.rows:
            event.stop()
            self.dismiss(self.rows[self.selected_index])
