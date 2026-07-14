"""
haagent/tui/overlays/channels.py - 渠道配置 overlay

列表展示渠道实例状态，支持新增微信、重新登录、启用/停用、删除与连接测试动作。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from textual import events
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from haagent.tui.design.utils import safe_summary

ChannelsOverlayAction = Literal[
    "add_weixin",
    "relogin",
    "enable",
    "disable",
    "delete",
    "test",
    "pair",
    "workspace",
]


@dataclass(frozen=True)
class ChannelsOverlayResult:
    action: ChannelsOverlayAction
    instance_id: str | None = None
    workspace_root: str | None = None


@dataclass(frozen=True)
class ChannelsOverlayState:
    instances: list[Any]
    selected_index: int = 0

    @property
    def selected_instance(self) -> Any | None:
        if not self.instances:
            return None
        index = min(max(self.selected_index, 0), len(self.instances) - 1)
        return self.instances[index]

    def move(self, delta: int) -> ChannelsOverlayState:
        if not self.instances:
            return replace(self, selected_index=0)
        next_index = min(max(self.selected_index + delta, 0), len(self.instances) - 1)
        return replace(self, selected_index=next_index)

    def render(self) -> str:
        lines = [
            "渠道配置",
            "配置微信等聊天渠道；token 仅存 keyring，不在此显示。",
            "",
        ]
        if not self.instances:
            lines.append("（暂无渠道实例）")
        for index, item in enumerate(self.instances):
            marker = ">" if index == min(self.selected_index, max(len(self.instances) - 1, 0)) else " "
            state = str(getattr(item, "state", "unknown"))
            enabled = "on" if getattr(item, "enabled", False) else "off"
            cred = "key:ok" if getattr(item, "credential_available", False) else "key:missing"
            permission = "perm:auto" if getattr(item, "permission_mode", "request_approval") == "auto_approve" else "perm:safe"
            workspace = safe_summary(str(getattr(item, "workspace_root", "")), 36)
            reauth = "  [重新登录]" if state == "auth_expired" or not getattr(item, "credential_available", True) else ""
            lines.append(
                f"{marker} {getattr(item, 'id', '?'):<16} {getattr(item, 'platform', '?'):<8} "
                f"{state:<14} {enabled:<3} {cred} {permission}  {workspace}{reauth}"
            )
        lines.extend(
            [
                "",
                "↑/↓ 移动  n 新增微信  r 重新登录  e 启用  d 停用",
                "p 重发配对码  w 选择 workspace  t 连接测试  x 删除  Esc 关闭",
            ]
        )
        return "\n".join(lines)


class ChannelsOverlay(ModalScreen[ChannelsOverlayResult | None]):
    def __init__(self, instances: list[Any]) -> None:
        super().__init__()
        self.state = ChannelsOverlayState(instances=list(instances))

    def compose(self) -> ComposeResult:
        yield Static(self.state.render(), id="channels-dialog")

    def on_key(self, event: events.Key) -> None:
        key = event.key
        if key in {"escape"}:
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
        if key == "n":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="add_weixin"))
            return
        selected = self.state.selected_instance
        if selected is None:
            return
        instance_id = str(getattr(selected, "id", ""))
        if key == "r":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="relogin", instance_id=instance_id))
            return
        if key == "e":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="enable", instance_id=instance_id))
            return
        if key == "d":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="disable", instance_id=instance_id))
            return
        if key == "t":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="test", instance_id=instance_id))
            return
        if key == "p":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="pair", instance_id=instance_id))
            return
        if key == "w":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="workspace", instance_id=instance_id))
            return
        if key == "x":
            event.stop()
            self.dismiss(ChannelsOverlayResult(action="delete", instance_id=instance_id))
            return

    def _set_state(self, state: ChannelsOverlayState) -> None:
        self.state = state
        self.query_one("#channels-dialog", Static).update(state.render())
