"""
tests/unit/tui/test_commands.py - TUI slash command 测试

验证命令注册、解析与 turns 命令分发保持稳定。
"""

from __future__ import annotations

from types import SimpleNamespace

from haagent.app.assistant_service import AssistantServiceError
from haagent.prompts.packs import iter_prompt_modes
from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.commands import command_registry, parse_slash_command


def test_turns_command_is_registered_and_parsed() -> None:
    registry = command_registry()

    result = parse_slash_command("/turns set 80", registry)

    assert result is not None
    assert result.command is not None
    assert result.command.action == "turns"
    assert result.argument == "set 80"


def test_prompt_pack_commands_are_forwarded_as_chat_prompts() -> None:
    registry = command_registry()

    prompt_mode_commands = {mode.command for mode in iter_prompt_modes()}
    names = {command.name for command in registry.commands()}
    assert prompt_mode_commands <= names
    for command in prompt_mode_commands:
        assert parse_slash_command(f"/{command} 看看改动", registry) is None


def test_turns_command_show_reports_current_and_configured_limits() -> None:
    app = _FakeTurnsApp(
        status=SimpleNamespace(current_max_turns=None, configured_interactive_max_turns=200),
    )

    HaAgentTuiApp._handle_turns_command(app, "")

    assert app.blocks == [
        (
            "Command",
            "当前 session turn 限制：unlimited\n已保存交互默认值：200\n用法：/turns [show|unlimited|COUNT]",
        )
    ]
    assert app.refreshes == 1


def test_turns_command_count_saves_and_applies_interactive_limit() -> None:
    app = _FakeTurnsApp(
        status=SimpleNamespace(current_max_turns=80, configured_interactive_max_turns=80),
    )

    HaAgentTuiApp._handle_turns_command(app, "set 80")

    assert app.service.saved_limits == [80]
    assert app.blocks == [("Command", "已保存交互默认 turn 限制：80；当前 session 已同步。")]


def test_turns_command_unlimited_is_current_session_only() -> None:
    app = _FakeTurnsApp(
        status=SimpleNamespace(current_max_turns=None, configured_interactive_max_turns=200),
    )

    HaAgentTuiApp._handle_turns_command(app, "unlimited")

    assert app.service.unlimited_calls == 1
    assert app.blocks == [("Command", "当前 session turn 限制已设为 unlimited；不会写入全局配置。")]


def test_turns_command_reports_missing_session_for_unlimited() -> None:
    app = _FakeTurnsApp(
        status=SimpleNamespace(current_max_turns=200, configured_interactive_max_turns=200),
        unlimited_error=AssistantServiceError("当前没有 session；先发送一条消息再使用 /turns unlimited。"),
    )

    HaAgentTuiApp._handle_turns_command(app, "unlimited")

    assert app.blocks == [("Command", "当前没有 session；先发送一条消息再使用 /turns unlimited。")]


def test_turns_command_rejects_invalid_arguments() -> None:
    app = _FakeTurnsApp(
        status=SimpleNamespace(current_max_turns=200, configured_interactive_max_turns=200),
    )

    HaAgentTuiApp._handle_turns_command(app, "0")

    assert app.blocks == [("Command", "用法：/turns [show|unlimited|COUNT]")]


class _FakeTurnsService:
    def __init__(self, status, unlimited_error: Exception | None = None) -> None:
        self._status = status
        self._unlimited_error = unlimited_error
        self.saved_limits: list[int] = []
        self.unlimited_calls = 0

    def get_turn_limit_status(self):
        return self._status

    def set_interactive_max_turns(self, max_turns: int):
        self.saved_limits.append(max_turns)
        return self._status

    def set_current_turns_unlimited(self):
        self.unlimited_calls += 1
        if self._unlimited_error is not None:
            raise self._unlimited_error
        return self._status


class _FakeTurnsApp:
    def __init__(self, status, unlimited_error: Exception | None = None) -> None:
        self.service = _FakeTurnsService(status, unlimited_error)
        self.blocks: list[tuple[str, str]] = []
        self.refreshes = 0

    def _append_block(self, title: str, body: str) -> None:
        self.blocks.append((title, body))

    def _refresh(self) -> None:
        self.refreshes += 1
