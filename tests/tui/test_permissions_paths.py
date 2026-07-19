"""
tests/tui/test_permissions_paths.py - HaAgent TUI permissions 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.design.renderers import status_line

from tests.tui.support import FakeAssistantService, _all_text, _text

def test_status_line_hides_permission_mode(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path, permission_mode="auto_approve")
    status = service.workspace.status()

    line = status_line(status, ui_state="idle", width=120)
    assert "工作区" in line
    assert "perm:" not in line

def test_tui_permissions_command_shows_current_external_roots(tmp_path: Path) -> None:
    external = tmp_path / "external"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        external_roots=[{"path": str(external), "access": "full", "source": "user"}],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            modal_text = _all_text(app)
            assert "权限设置" in modal_text
            assert "请求批准" in modal_text
            assert "自动批准" in modal_text
            assert "完全访问权限" in modal_text
            assert "external" in modal_text
            assert "完全信任" in modal_text

    asyncio.run(run())

def test_tui_ctrl_p_opens_permissions_modal(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()

            assert "权限设置" in _all_text(app)

    asyncio.run(run())

def test_tui_permissions_modal_changes_permission_modes(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path)

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()

            assert service.permission_mode == "auto_approve"
            assert "perm:" not in _text(app, "#status-bar")

            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()
            assert "完全访问权限" in _all_text(app)

            await pilot.press("y")
            await pilot.pause()
            assert service.permission_mode == "full_access"
            assert "perm:" not in _text(app, "#status-bar")

    asyncio.run(run())

def test_tui_permissions_modal_changes_access_removes_and_clears_roots(tmp_path: Path) -> None:
    external_a = tmp_path / "external-a"
    external_b = tmp_path / "external-b"
    service = FakeAssistantService(
        workspace_root=tmp_path,
        external_roots=[
            {"path": str(external_a), "access": "read", "source": "user"},
            {"path": str(external_b), "access": "full", "source": "user"},
        ],
    )

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("f")
            assert service.external_roots[0]["access"] == "full"

            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("r")
            assert service.external_roots == [{"path": str(external_a.resolve()), "access": "full", "source": "user"}]

            input_widget.value = "/permissions"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("c")
            await pilot.press("y")
            assert service.external_roots == []

    asyncio.run(run())

def test_tui_submits_prompt_with_external_path_without_preflight_authorization(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project", permission_mode="auto_approve")
    service.workspace_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            prompt = f'介绍 "{external}"'
            input_widget = app.query_one("#prompt-input")
            input_widget.value = prompt
            await pilot.press("enter")
            await pilot.pause()

            assert "检测到工作区外目录" not in _all_text(app)
            assert service.prompts == [prompt]

    asyncio.run(run())


def test_tui_submits_url_without_preflight_directory_authorization(tmp_path: Path) -> None:
    service = FakeAssistantService(workspace_root=tmp_path / "project", permission_mode="auto_approve")
    service.workspace_root.mkdir()
    prompt = "联网搜索 https://velagrow.com/"

    async def run() -> None:
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = prompt
            await pilot.press("enter")
            await pilot.pause()

            assert "检测到工作区外目录" not in _all_text(app)
            assert service.prompts == [prompt]

    asyncio.run(run())


