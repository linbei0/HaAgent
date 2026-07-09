"""
src/haagent/tui/widgets/input_dock.py - TUI 输入停靠区组件

统一管理输入区 overlay、prompt 读写和焦点恢复，避免 App 直接操控布局细节。
"""

from __future__ import annotations

from pathlib import Path

from textual.widget import Widget
from textual.containers import Vertical

from haagent.tui.commands import command_registry
from haagent.tui.commands.suggestions import CommandSuggestionOverlay
from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import FileReferenceIndex
from haagent.tui.widgets.timeline import PromptInput, _end_location


class InputDock(Vertical):
    """输入区容器：负责 prompt 与补全 overlay 的 DOM 生命周期。"""

    COLLAPSED_HEIGHT = 5
    EXPANDED_HEIGHT = 14

    def __init__(
        self,
        *children: Widget,
        workspace_root: Path | None = None,
        file_reference_index: FileReferenceIndex | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*children, **kwargs)
        self.workspace_root = workspace_root or Path.cwd()
        self.file_reference_index = file_reference_index
        self.command_overlay: CommandSuggestionOverlay | None = None
        self.file_ref_overlay: FileReferenceOverlay | None = None

    def prompt_value(self) -> str:
        return self._prompt().text

    def set_prompt_value(self, value: str) -> None:
        prompt = self._prompt()
        prompt.text = value
        prompt.cursor_location = _end_location(value)
        prompt.focus()

    def open_command_suggestions(self, query: str) -> CommandSuggestionOverlay:
        self.close_file_refs()
        prompt = self._prompt()
        if self.command_overlay is None:
            self.command_overlay = CommandSuggestionOverlay(command_registry().commands())
            self.mount(self.command_overlay, before=prompt)
        self.styles.height = self.EXPANDED_HEIGHT
        self.command_overlay.update_query(query)
        self.call_after_refresh(prompt.focus)
        return self.command_overlay

    def open_file_refs(self, query: str) -> FileReferenceOverlay:
        self.close_command_suggestions()
        prompt = self._prompt()
        if self.file_ref_overlay is None:
            self.file_ref_overlay = FileReferenceOverlay(
                self.workspace_root,
                query,
                index=self.file_reference_index,
            )
            self.mount(self.file_ref_overlay, before=prompt)
        self.styles.height = self.EXPANDED_HEIGHT
        self.file_ref_overlay.update_query(query)
        self.call_after_refresh(prompt.focus)
        return self.file_ref_overlay

    def close_overlays(self) -> None:
        self.close_command_suggestions()
        self.close_file_refs()

    def close_command_suggestions(self) -> None:
        overlay = self.command_overlay
        self.command_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self._collapse_if_idle()

    def close_file_refs(self) -> None:
        overlay = self.file_ref_overlay
        self.file_ref_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self._collapse_if_idle()

    def selected_command(self):
        if self.command_overlay is None:
            return None
        return self.command_overlay.selected_command()

    def selected_file_token(self) -> str | None:
        if self.file_ref_overlay is None:
            return None
        return self.file_ref_overlay.selected_token()

    def _prompt(self) -> PromptInput:
        return self.query_one("#prompt-input", PromptInput)

    def _collapse_if_idle(self) -> None:
        if self.command_overlay is None and self.file_ref_overlay is None:
            self.styles.height = self.COLLAPSED_HEIGHT
            if self.is_mounted:
                self._prompt().focus()
