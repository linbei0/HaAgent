"""
src/haagent/tui/application/completion_flow.py - TUI 输入补全流程

统一管理命令建议 overlay 与文件引用 overlay 的打开、过滤、导航、确认和关闭。
两类补全都基于 InputDock 内的 OptionList，App 只保留 Textual binding 入口。
"""

from __future__ import annotations

from typing import Any

from textual import events

from haagent.tui.commands import is_prompt_mode_command, parse_slash_command
from haagent.tui.files.refs import query_after_at, replace_at_query


class CompletionFlow:
    """封装命令建议与文件引用两类输入补全 overlay 的交互。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.command_overlay = None
        self.file_ref_overlay = None

    # ── 命令建议 ─────────────────────────────────────────────────────────
    def open_command_suggestions(self) -> None:
        prompt_input = self._app._prompt_input()
        value = self._app._prompt_value(prompt_input)
        if prompt_input.has_focus and value:
            if value.startswith("/") and " " not in value:
                self._open_command_overlay(value.removeprefix("/"))
            else:
                prompt_input.insert("/")
            return
        if not value:
            prompt_input.insert("/")
        self._open_command_overlay(self._app._prompt_value(prompt_input).removeprefix("/"))

    def _open_command_overlay(self, query: str) -> None:
        self.file_ref_overlay = None
        self.command_overlay = self._app._input_dock().open_command_suggestions(query)

    def handle_command_key(self, event: events.Key) -> None:
        overlay = self.command_overlay
        if overlay is None:
            return
        result = overlay.handle_navigation_key(event)
        if result is None:
            return
        if result == "":
            self.close_command_suggestions()
            return
        self._accept_command_token(result.token)

    def accept_command_suggestion(self) -> None:
        overlay = self.command_overlay
        if overlay is None:
            return
        command = overlay.selected_command()
        if command is None:
            self.close_command_suggestions()
            return
        self._accept_command_token(command.token)

    def _accept_command_token(self, token: str) -> None:
        self.close_command_suggestions()
        prompt_input = self._app._prompt_input()
        if is_prompt_mode_command(token):
            self._app._set_prompt_value(prompt_input, f"{token} ")
            return
        self._app._set_prompt_value(prompt_input, "")
        self._app._handle_slash_command(parse_slash_command(token, self._app._commands))

    def sync_command_suggestions(self, text: str) -> None:
        if self.command_overlay is None:
            return
        if not text.startswith("/") or " " in text:
            self.close_command_suggestions()
            return
        self.command_overlay.update_query(text.removeprefix("/"))

    def close_command_suggestions(self) -> None:
        self.command_overlay = None
        self._app._input_dock().close_command_suggestions()

    def command_suggestions_is_open(self) -> bool:
        return self.command_overlay is not None

    # ── 文件引用 ─────────────────────────────────────────────────────────
    def open_file_refs(self) -> None:
        prompt_input = self._app._prompt_input()
        query = query_after_at(self._app._prompt_value(prompt_input))
        if query is None:
            return
        status = self._app.service.get_workspace_status()
        input_dock = self._app._input_dock()
        input_dock.workspace_root = status.workspace_root
        input_dock.file_reference_index = self._app._file_ref_index
        self.file_ref_overlay = input_dock.open_file_refs(query)

    def handle_file_ref_key(self, event: events.Key) -> None:
        overlay = self.file_ref_overlay
        if overlay is None:
            return
        token = overlay.handle_navigation_key(event)
        if token is None:
            return
        if token:
            self._apply_file_reference(token)
        self.close_file_refs()

    def accept_file_ref(self) -> None:
        overlay = self.file_ref_overlay
        if overlay is None:
            return
        token = overlay.selected_token()
        if token:
            self._apply_file_reference(token)
            self.close_file_refs()

    def sync_file_refs(self, text: str) -> None:
        if self.file_ref_overlay is None:
            return
        query = query_after_at(text)
        if query is None:
            self.close_file_refs()
        else:
            self.file_ref_overlay.update_query(query)

    def _apply_file_reference(self, token: str | None) -> None:
        prompt_input = self._app._prompt_input()
        if token is not None:
            self._app._set_prompt_value(prompt_input, replace_at_query(self._app._prompt_value(prompt_input), token))
        prompt_input.focus()

    def close_file_refs(self) -> None:
        self.file_ref_overlay = None
        self._app._input_dock().close_file_refs()

    def file_reference_is_open(self) -> bool:
        return self.file_ref_overlay is not None
