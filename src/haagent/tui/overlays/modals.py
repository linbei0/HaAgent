"""
haagent/tui/modals.py - TUI 弹窗组件

封装帮助和工具审批弹窗，保持 ModalScreen 行为独立于主 App 编排。
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.runtime.execution.path_policy import PermissionMode
from haagent.tui.design.copy import MODAL_TITLES
from haagent.tui.design.keys import APPROVAL_BINDINGS, EDIT_DIFF_BINDINGS, HELP_DISMISS_BINDINGS, help_body
from haagent.tui.design.renderers import approval_body, edit_diff_body


class HelpModal(ModalScreen[None]):
    BINDINGS = HELP_DISMISS_BINDINGS

    def __init__(self, context: str) -> None:
        super().__init__()
        self.context = context

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(MODAL_TITLES["help"], id="help-title")
            yield Static(help_body(self.context), id="help-body")
            yield Static("[Esc]关闭")

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


class ToolApprovalModal(ModalScreen[bool]):
    BINDINGS = APPROVAL_BINDINGS

    def __init__(self, request: HumanInteractionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(MODAL_TITLES["approval"], id="approval-title")
            yield Static(Text(approval_body(self.request)), id="approval-body")
            with Horizontal(id="approval-buttons"):
                yield Button("允许 y", id="approval-allow", variant="success", classes="action-success")
                yield Button("拒绝 n", id="approval-deny", variant="error", classes="action-danger")

    def on_mount(self) -> None:
        self.query_one("#approval-deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approval-allow")

    def on_key(self, event: events.Key) -> None:
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            self.action_help()

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def action_help(self) -> None:
        self.app.push_screen(HelpModal("approval"))


class EditDiffModal(ModalScreen[str]):
    BINDINGS = EDIT_DIFF_BINDINGS

    def __init__(self, request: HumanInteractionRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Static(MODAL_TITLES["edit_diff"], id="approval-title")
            yield Static(Text(edit_diff_body(self.request)), id="approval-body")
            with Horizontal(id="approval-buttons"):
                yield Button("允许 y", id="edit-allow-once", variant="success", classes="action-success")
                yield Button("始终 a", id="edit-allow-always", variant="primary")
                yield Button("拒绝 n", id="edit-deny", variant="error", classes="action-danger")

    def on_mount(self) -> None:
        self.query_one("#edit-deny", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "edit-allow-once":
            self.dismiss("once")
        elif event.button.id == "edit-allow-always":
            self.dismiss("always")
        else:
            self.dismiss("deny")

    def on_key(self, event: events.Key) -> None:
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            self.action_help()

    def action_allow_once(self) -> None:
        self.dismiss("once")

    def action_allow_always(self) -> None:
        self.dismiss("always")

    def action_deny(self) -> None:
        self.dismiss("deny")

    def action_help(self) -> None:
        self.app.push_screen(HelpModal("edit_diff"))


class ConfirmModal(ModalScreen[bool]):
    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static(self.title, id="confirm-title")
            yield Static(self.body, id="confirm-body")
            with Horizontal(id="confirm-buttons"):
                yield Button("确认 y", id="confirm-yes", variant="error", classes="action-danger")
                yield Button("取消 n", id="confirm-no", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#confirm-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" or event.character == "n":
            event.stop()
            self.dismiss(False)
            return
        if event.character == "y":
            event.stop()
            self.dismiss(True)


class ExternalDirectoryDecisionModal(ModalScreen[str | None]):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def compose(self) -> ComposeResult:
        with Vertical(id="external-directory-dialog"):
            yield Static("检测到工作区外目录", id="external-directory-title")
            yield Static(str(self.path), id="external-directory-body")
            with Horizontal(id="external-directory-buttons"):
                yield Button("只读参考", id="external-read", variant="primary")
                yield Button("切换工作区", id="external-switch")
                yield Button("完全信任", id="external-full", variant="warning")
                yield Button("取消", id="external-cancel", variant="error")

    def on_mount(self) -> None:
        self.query_one("#external-read", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mapping = {
            "external-read": "read",
            "external-switch": "switch",
            "external-full": "full",
            "external-cancel": None,
        }
        self.dismiss(mapping.get(event.button.id))

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape" or event.character == "c":
            event.stop()
            self.dismiss(None)
        elif event.character in {"r", "o"}:
            event.stop()
            self.dismiss("read")
        elif event.character == "s":
            event.stop()
            self.dismiss("switch")
        elif event.character == "f":
            event.stop()
            self.dismiss("full")


PERMISSION_MODE_ORDER: list[PermissionMode] = ["request_approval", "auto_approve", "full_access"]
PERMISSION_MODE_LABELS: dict[PermissionMode, str] = {
    "request_approval": "请求批准",
    "auto_approve": "自动批准",
    "full_access": "完全访问权限",
}


class PermissionsModal(ModalScreen[dict[str, object] | None]):
    def __init__(
        self,
        project_root: Path,
        external_roots: list[dict[str, str]],
        permission_mode: PermissionMode = "request_approval",
    ) -> None:
        super().__init__()
        self.project_root = project_root
        self.external_roots = list(external_roots)
        self.selected = 0
        self.mode_index = PERMISSION_MODE_ORDER.index(permission_mode) if permission_mode in PERMISSION_MODE_ORDER else 0

    def compose(self) -> ComposeResult:
        with Vertical(id="permissions-dialog"):
            yield Static("权限设置", id="permissions-title")
            yield Static(self._body(), id="permissions-body")
            yield Static("←/→ 模式  Enter 应用  ↑/↓ 目录  o 只读  f 完全信任  r 移除  c 清空  Esc 关闭")

    def on_mount(self) -> None:
        self.can_focus = True
        self.focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            self.dismiss(None)
            return
        if event.key == "down":
            event.stop()
            self._move(1)
            return
        if event.key == "up":
            event.stop()
            self._move(-1)
            return
        if event.key in {"right", "left"}:
            event.stop()
            self._move_mode(1 if event.key == "right" else -1)
            return
        if event.key == "enter":
            event.stop()
            self.dismiss({"action": "set_mode", "mode": PERMISSION_MODE_ORDER[self.mode_index]})
            return
        if event.character == "c":
            event.stop()
            self.dismiss({"action": "clear"})
            return
        if not self.external_roots:
            return
        selected = self.external_roots[self.selected]
        if event.character == "o":
            event.stop()
            self.dismiss({"action": "set_access", "path": selected["path"], "access": "read"})
        elif event.character == "f":
            event.stop()
            self.dismiss({"action": "set_access", "path": selected["path"], "access": "full"})
        elif event.character == "r":
            event.stop()
            self.dismiss({"action": "remove", "path": selected["path"]})

    def _move(self, delta: int) -> None:
        if not self.external_roots:
            return
        self.selected = min(max(self.selected + delta, 0), len(self.external_roots) - 1)
        self.query_one("#permissions-body", Static).update(self._body())

    def _move_mode(self, delta: int) -> None:
        self.mode_index = (self.mode_index + delta) % len(PERMISSION_MODE_ORDER)
        self.query_one("#permissions-body", Static).update(self._body())

    def _body(self) -> str:
        mode_parts = []
        for index, mode in enumerate(PERMISSION_MODE_ORDER):
            label = PERMISSION_MODE_LABELS[mode]
            mode_parts.append(f"[{label}]" if index == self.mode_index else f" {label} ")
        lines = [
            "权限模式：",
            "  " + "  ".join(mode_parts),
            "",
            "项目根：",
            f"  {self.project_root}",
            "",
            "外部目录：",
        ]
        if not self.external_roots:
            lines.append("  无")
        for index, root in enumerate(self.external_roots):
            marker = ">" if index == self.selected else " "
            access = root.get("access", "")
            label = "只读参考" if access == "read" else "完全信任" if access == "full" else access or "-"
            lines.append(f"{marker} {root.get('path', '-')}  {label}")
        return "\n".join(lines)
