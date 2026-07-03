"""
haagent/tui/command_suggestions.py - slash command 建议面板

基于结构化命令注册表提供过滤、选择和执行回调，不把命令文本发送给模型。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from textual import events
from textual.widgets import Static

from haagent.tui.commands import SlashCommand
from haagent.tui.design.copy import EMPTY_LABELS, MODAL_TITLES


VISIBLE_COMMAND_COUNT = 4


@dataclass(frozen=True)
class CommandSuggestionState:
    commands: list[SlashCommand]
    query: str = ""
    selected_index: int = 0
    scroll_offset: int = 0

    @property
    def visible_commands(self) -> list[SlashCommand]:
        needle = self.query.removeprefix("/").casefold()
        if not needle:
            return self.commands
        return [
            command
            for command in self.commands
            if needle in command.name.casefold() or needle in command.description.casefold()
        ]

    @property
    def selected_command(self) -> SlashCommand | None:
        visible = self.visible_commands
        if not visible:
            return None
        return visible[min(max(self.selected_index, 0), len(visible) - 1)]

    def with_query(self, query: str) -> CommandSuggestionState:
        return replace(self, query=query, selected_index=0, scroll_offset=0)

    def move(self, delta: int) -> CommandSuggestionState:
        visible = self.visible_commands
        if not visible:
            return replace(self, selected_index=0, scroll_offset=0)
        next_index = min(max(self.selected_index + delta, 0), len(visible) - 1)
        scroll_offset = self.scroll_offset
        if next_index < scroll_offset:
            scroll_offset = next_index
        elif next_index >= scroll_offset + VISIBLE_COMMAND_COUNT:
            scroll_offset = next_index - VISIBLE_COMMAND_COUNT + 1
        return replace(self, selected_index=next_index, scroll_offset=scroll_offset)

    def render(self) -> str:
        lines = [MODAL_TITLES["commands"], f"过滤: /{self.query.removeprefix('/') or ''}", ""]
        visible = self.visible_commands
        if not visible:
            lines.append(EMPTY_LABELS["no_matching_commands"])
        scroll_offset = min(self.scroll_offset, max(len(visible) - VISIBLE_COMMAND_COUNT, 0))
        visible_window = visible[scroll_offset : scroll_offset + VISIBLE_COMMAND_COUNT]
        for offset, command in enumerate(visible_window):
            index = scroll_offset + offset
            marker = ">" if index == min(self.selected_index, len(visible) - 1) else " "
            lines.append(f"{marker} {command.token:<12} {command.description}")
        lines.extend(["", "输入过滤  ↑/↓ 移动  Enter 执行  Esc 关闭"])
        return "\n".join(lines)


class CommandSuggestionOverlay(Static):
    def __init__(self, commands: list[SlashCommand]) -> None:
        super().__init__("", id="command-suggestions-dialog")
        self.state = CommandSuggestionState(commands=commands)

    def on_mount(self) -> None:
        self.update(self.state.render())

    def handle_key(self, event: events.Key) -> SlashCommand | str | None:
        key = event.key
        if key == "escape":
            event.stop()
            return ""
        if key == "up":
            event.stop()
            self._set_state(self.state.move(-1))
            return None
        if key == "down":
            event.stop()
            self._set_state(self.state.move(1))
            return None
        if key == "enter":
            event.stop()
            command = self.state.selected_command
            if command is not None:
                return command
            return ""
        return None

    def update_query(self, query: str) -> None:
        self._set_state(self.state.with_query(query))

    def _set_state(self, state: CommandSuggestionState) -> None:
        self.state = state
        if self.is_mounted:
            self.update(state.render())
