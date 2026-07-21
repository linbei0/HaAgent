"""
tests/unit/tui/test_screen_helpers.py - Modal dismiss 与列表窗口辅助测试

覆盖 safe_dismiss 防 ScreenStackError，以及方向键路径禁止全量 set_options。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from textual.app import App, ScreenStackError
from textual.widgets import OptionList

from haagent.app.assistant_types import AssistantModelConnection
from haagent.tui.commands import SlashCommand
from haagent.tui.commands.suggestions import CommandSuggestionOverlay
from haagent.tui.design.screen_helpers import safe_dismiss, visible_window_start
from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import FileReferenceIndex, FileReferenceMatch
from haagent.tui.overlays.connections import ConnectionCenterOverlay
from haagent.tui.overlays.sessions import SessionOverlayState


def test_visible_window_start_keeps_selection_in_page() -> None:
    assert visible_window_start(total=0, selected=0, page_size=15) == 0
    assert visible_window_start(total=10, selected=3, page_size=15) == 0
    assert visible_window_start(total=40, selected=20, page_size=15) == 6
    assert visible_window_start(total=40, selected=39, page_size=15) == 25


def test_safe_dismiss_ignores_when_screen_already_left_stack() -> None:
    screen = MagicMock()
    screen.app.screen = MagicMock()  # 不是自己
    screen.dismiss = MagicMock()

    safe_dismiss(screen, result="x")

    screen.dismiss.assert_not_called()


def test_safe_dismiss_swallows_screen_stack_error() -> None:
    screen = MagicMock()
    screen.app.screen = screen
    screen.dismiss = MagicMock(side_effect=ScreenStackError("Can't pop screen"))

    safe_dismiss(screen, result=None)

    screen.dismiss.assert_called_once_with(None)


def test_connection_center_move_does_not_rebuild_option_list() -> None:
    async def run() -> None:
        connections = [
            _connection(f"c-{index:03d}", f"name-{index}", "requesty") for index in range(40)
        ]
        overlay = ConnectionCenterOverlay(connections)
        app = App()
        async with app.run_test(size=(120, 40)) as pilot:
            await app.push_screen(overlay)
            await pilot.pause(0.05)
            option_list = overlay.query_one(OptionList)
            assert option_list.option_count == 40
            calls = {"set_options": 0}
            original = option_list.set_options

            def counting(*args, **kwargs):
                calls["set_options"] += 1
                return original(*args, **kwargs)

            option_list.set_options = counting  # type: ignore[method-assign]

            overlay.on_key(_FakeKeyEvent("down"))
            await pilot.pause(0.05)

            assert overlay.state.selected_index == 1
            assert option_list.highlighted == 1
            assert calls["set_options"] == 0

    asyncio.run(run())


def test_file_reference_move_does_not_rebuild_option_list(tmp_path: Path) -> None:
    async def run() -> None:
        index = FileReferenceIndex(
            root=tmp_path.resolve(),
            files=tuple(
                FileReferenceMatch(
                    path=tmp_path / f"file-{index:02}.txt",
                    display_path=f"file-{index:02}.txt",
                )
                for index in range(30)
            ),
        )
        overlay = FileReferenceOverlay(tmp_path, "", index)
        app = App()
        async with app.run_test(size=(100, 30)) as pilot:
            await app.mount(overlay)
            await pilot.pause(0.05)
            option_list = overlay.query_one(OptionList)
            calls = {"set_options": 0}
            original = option_list.set_options

            def counting(*args, **kwargs):
                calls["set_options"] += 1
                return original(*args, **kwargs)

            option_list.set_options = counting  # type: ignore[method-assign]

            overlay.handle_navigation_key(_FakeKeyEvent("down"))
            await pilot.pause(0.05)

            assert overlay.selected_index == 1
            assert option_list.highlighted == 1
            assert calls["set_options"] == 0

    asyncio.run(run())


def test_command_suggestions_move_does_not_rebuild_option_list() -> None:
    async def run() -> None:
        commands = [SlashCommand(f"cmd{i}", f"desc{i}", f"cmd{i}") for i in range(20)]
        overlay = CommandSuggestionOverlay(commands)
        app = App()
        async with app.run_test(size=(100, 30)) as pilot:
            await app.mount(overlay)
            await pilot.pause(0.05)
            option_list = overlay.query_one(OptionList)
            calls = {"set_options": 0}
            original = option_list.set_options

            def counting(*args, **kwargs):
                calls["set_options"] += 1
                return original(*args, **kwargs)

            option_list.set_options = counting  # type: ignore[method-assign]

            overlay.handle_navigation_key(_FakeKeyEvent("down"))
            await pilot.pause(0.05)

            assert overlay.state.selected_index == 1
            assert option_list.highlighted == 1
            assert calls["set_options"] == 0

    asyncio.run(run())


def test_session_overlay_render_uses_window() -> None:
    sessions = [
        _session(f"session-{index:03d}", f"request {index}") for index in range(40)
    ]
    state = SessionOverlayState(sessions=sessions, selected_index=25)
    rendered = state.render()
    assert "session-025" in rendered
    assert "session-000" not in rendered
    assert "session-039" not in rendered


class _FakeKeyEvent:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _connection(connection_id: str, name: str, provider_id: str) -> AssistantModelConnection:
    return AssistantModelConnection(
        id=connection_id,
        name=name,
        provider_id=provider_id,
        provider_name="Requesty",
        gateway_provider="openai-chat",
        base_url="https://router.requesty.ai/v1",
        api_key_env="REQUESTY_API_KEY",
        credential_source="keyring",
        credential_available=True,
        credential_source_used="keyring",
        model_config_diagnostics=(),
    )


def _session(session_id: str, first_request: str):
    from haagent.app.assistant_types import AssistantSessionSummary

    return AssistantSessionSummary(
        session_id=session_id,
        created_at="2026-06-27T00:00:00+00:00",
        updated_at="2026-06-27T01:00:00+00:00",
        workspace_root=Path("."),
        turn_count=1,
        first_request=first_request,
        session_path=Path(".") / ".runs" / "sessions" / session_id,
    )
