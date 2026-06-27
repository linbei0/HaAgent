"""
haagent/tui/modals.py - TUI 弹窗组件

封装帮助和工具审批弹窗，保持 ModalScreen 行为独立于主 App 编排。
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from haagent.runtime.human_interaction import HumanInteractionRequest
from haagent.tui.keys import APPROVAL_BINDINGS, HELP_DISMISS_BINDINGS, help_body
from haagent.tui.renderers import approval_body


class HelpModal(ModalScreen[None]):
    BINDINGS = HELP_DISMISS_BINDINGS

    def __init__(self, context: str) -> None:
        super().__init__()
        self.context = context

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static("HaAgent Help", id="help-title")
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
            yield Static("Tool Approval", id="approval-title")
            yield Static(Text(approval_body(self.request)), id="approval-body")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow y", id="approval-allow", variant="success")
                yield Button("Deny n", id="approval-deny", variant="error")

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
