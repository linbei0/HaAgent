"""
tests/unit/tui/test_application_components.py - TUI 应用组件化测试

验证 App 外围协调器、输入停靠区和列表型 overlay 的可组合边界。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.geometry import Offset
from textual.screen import Screen
from textual.widgets import OptionList, Static

from haagent.tui.application.app import HaAgentScreen, HaAgentTuiApp
from haagent.tui.application.commands import CommandDispatcher
from haagent.tui.application.conversation import ConversationController
from haagent.tui.commands import SlashCommand, SlashCommandResult
from haagent.tui.commands.suggestions import CommandSuggestionOverlay
from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch
from haagent.tui.widgets import PromptInput, ProgressStatusLine
from haagent.tui.widgets.input_dock import InputDock
from tests.tui.support import FakeAssistantService


def test_default_screen_ignores_detached_cached_text_selection_target(monkeypatch) -> None:
    stale_widget = Static("旧 Markdown 段落")
    monkeypatch.setattr(
        Screen,
        "get_widget_and_offset_at",
        lambda self, x, y: (stale_widget, Offset(0, 0)),
    )

    screen = HaAgentScreen()

    assert screen.get_widget_and_offset_at(10, 5) == (None, None)


def test_command_dispatcher_routes_registered_action_and_refreshes_errors() -> None:
    app = _DispatchApp()
    dispatcher = CommandDispatcher(app)

    handled = dispatcher.dispatch(
        SlashCommandResult(SlashCommand("help", "帮助", "help")),
    )
    error_handled = dispatcher.dispatch(SlashCommandResult(command=None, error="未知命令：/x"))

    assert handled is True
    assert error_handled is True
    assert app.calls == ["help"]
    assert app.blocks == [("Command", "未知命令：/x")]
    assert app.refreshes == 1


def test_command_dispatcher_returns_false_for_missing_command() -> None:
    app = _DispatchApp()
    dispatcher = CommandDispatcher(app)

    assert dispatcher.dispatch(SlashCommandResult(command=None)) is False
    assert app.calls == []


def test_conversation_controller_wraps_timeline_public_operations() -> None:
    app = _ConversationApp()
    controller = ConversationController(app)

    controller.append_block("You", "你好", turn_index=1)
    controller.start_assistant(turn_index=1)
    controller.merge_assistant_delta(1, 1, "完整审查报告")
    controller.finalize_intermediate_message(1, 1, "完整审查报告")
    controller.merge_assistant_delta(1, 2, "最终")
    controller.finalize_assistant_message(1, 2, "最终总结")
    controller.record_tool_diagnostic(1, "shell", "诊断")
    controller.set_tool_details(True)

    assert app.timeline.calls == [
        ("user", "你好", 1),
        ("start", "", 1),
        ("delta", "完整审查报告", 1),
        ("intermediate", "完整审查报告", 1),
        ("delta", "最终", 1),
        ("final", "最终总结", 1),
        ("diagnostic", "shell:诊断", 1),
        ("details", "True", 0),
    ]


def test_input_dock_opens_one_overlay_and_preserves_prompt_value() -> None:
    async def run() -> None:
        app = _InputDockApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)
            dock = app.query_one(InputDock)
            prompt = app.query_one(PromptInput)

            prompt.text = "/he"
            prompt.focus()
            dock.open_command_suggestions("he")
            await pilot.pause(0.1)
            dock.open_file_refs("read")
            await pilot.pause(0.1)

            assert prompt.text == "/he"
            assert len(app.query(OptionList)) == 1
            assert app.query_one(PromptInput).has_focus

    asyncio.run(run())


def test_command_suggestion_overlay_uses_option_list_selection() -> None:
    async def run() -> None:
        overlay = CommandSuggestionOverlay([SlashCommand("help", "帮助", "help")])
        app = _OverlayApp(overlay)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)

            assert app.query_one(OptionList).option_count == 1
            await pilot.press("enter")

            assert app.selected == SlashCommand("help", "帮助", "help")

    asyncio.run(run())


def test_file_reference_overlay_uses_option_list_without_rescanning_preloaded_index(tmp_path: Path) -> None:
    async def run() -> None:
        index = FileReferenceIndex(
            root=tmp_path.resolve(),
            files=(
                FileReferenceMatch(path=tmp_path / "README.md", display_path="README.md"),
                FileReferenceMatch(path=tmp_path / "src" / "app.py", display_path="src/app.py"),
            ),
        )
        overlay = FileReferenceOverlay(tmp_path, "read", index=index)
        app = _OverlayApp(overlay)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause(0.1)

            option_list = app.query_one(OptionList)
            assert option_list.option_count == 1
            await pilot.press("enter")

            assert app.selected == '@file("README.md")'

    asyncio.run(run())


def test_file_reference_index_callback_ignores_unmounted_input_dock(tmp_path: Path) -> None:
    index = FileReferenceIndex(root=tmp_path.resolve(), files=())
    app = _FileReferenceCallbackApp()

    HaAgentTuiApp._set_file_reference_index(app, index)

    assert app._file_ref_index is index


def test_app_exclusive_workers_use_independent_responsibility_groups(tmp_path: Path) -> None:
    app = HaAgentTuiApp(FakeAssistantService(workspace_root=tmp_path))
    groups: list[str] = []

    def capture_worker(_callback, **kwargs):
        groups.append(kwargs["group"])
        return SimpleNamespace()

    app.run_worker = capture_worker  # type: ignore[method-assign]

    app._run_prompt("test")
    app._warm_file_reference_index()
    app._search_skill_marketplace_worker("textual", 1)
    app._run_initial_session_restore_worker(None, True)
    app._refresh_model_catalog_only()
    app._run_model_connection_test("connection")
    app._run_memory_action_worker("confirm", "candidate")

    assert groups == [
        "prompt",
        "file-reference-index",
        "skills-marketplace",
        "session-ops",
        "model-ops",
        "model-connection-test",
        "memory-ops",
    ]


class _DispatchConversation:
    def __init__(self, app: "_DispatchApp") -> None:
        self._app = app

    def append_block(self, title: str, body: str, *, turn_index: int | None = None) -> None:
        self._app.blocks.append((title, body))


class _DispatchApp:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.refreshes = 0
        self._active_turn_index = 0
        self._conversation = _DispatchConversation(self)

    def action_help(self) -> None:
        self.calls.append("help")

    def _refresh(self) -> None:
        self.refreshes += 1


class _ConversationTimeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def add_user(self, content: str, *, turn_index: int) -> None:
        self.calls.append(("user", content, turn_index))

    def start_assistant_response(self, *, turn_index: int) -> None:
        self.calls.append(("start", "", turn_index))

    def update_assistant_delta(self, turn_index: int, delta: str) -> None:
        self.calls.append(("delta", delta, turn_index))

    def finalize_assistant(self, turn_index: int, content: str) -> None:
        self.calls.append(("final", content, turn_index))

    def finalize_intermediate(self, turn_index: int, model_turn: int | None, content: str) -> None:
        del model_turn
        self.calls.append(("intermediate", content, turn_index))

    def add_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        self.calls.append(("diagnostic", f"{tool_name}:{message}", turn_index))

    def set_tool_details(self, enabled: bool) -> None:
        self.calls.append(("details", str(enabled), 0))


class _ConversationApp:
    def __init__(self) -> None:
        self.timeline = _ConversationTimeline()

    def query_one(self, selector: str, widget_type):
        assert selector == "#conversation"
        return self.timeline


class _FileReferenceCallbackApp:
    """模拟 App 尚在退出、但输入区已卸载的生命周期窗口。"""

    is_mounted = True
    _file_ref_index = None
    _input_dock_widget = None

    def _input_dock(self):
        raise AssertionError("已卸载输入区不应被再次查询")


class _InputDockApp(App[None]):
    def compose(self) -> ComposeResult:
        with InputDock(id="input-panel"):
            yield ProgressStatusLine("", id="progress-status")
            yield PromptInput(placeholder="输入任务", id="prompt-input", show_line_numbers=False)


class _OverlayApp(App[None]):
    def __init__(self, overlay) -> None:
        super().__init__()
        self.overlay = overlay
        self.selected = None

    def compose(self) -> ComposeResult:
        with Vertical(id="host"):
            yield self.overlay

    def on_command_suggestion_overlay_selected(self, event) -> None:
        self.selected = event.command

    def on_file_reference_overlay_selected(self, event) -> None:
        self.selected = event.token
