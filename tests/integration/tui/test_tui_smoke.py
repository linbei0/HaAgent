"""
tests/integration/tui/test_tui_smoke.py - 默认 TUI 主路径冒烟测试

验证普通入口至少可以完成启动、提交一条消息并正常退出。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.widgets import PromptInput

from tests.tui.support import FakeAssistantService


def test_tui_starts_submits_and_exits(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)

        async with app.run_test(size=(80, 24)) as pilot:
            prompt = app.query_one("#prompt-input", PromptInput)
            prompt.value = "整理当前目录"
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert service.prompts == ["整理当前目录"]
            await pilot.press("ctrl+q")

    asyncio.run(run())
