"""
tests/tui/test_steering_ui.py - 运行中排队与立即引导的 TUI 集成测试

验证运行中 Enter 排队 + 本轮结束自动投递、Ctrl+G 立即引导及其回落路径。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp

from tests.tui.support import FakeAssistantService, _text


def test_tui_enter_during_run_queues_and_dispatches_after_finish(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            await pilot.pause(0.1)

            input_widget.value = "queued follow up"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert input_widget.value == ""
            assert service.prompts == ["Long task"]
            assert app._steering_queue == ["queued follow up"]
            assert "You (已排队)" in _text(app, "#conversation")

            service.release.set()
            for _ in range(20):
                await pilot.pause(0.1)
                if len(service.prompts) >= 2:
                    break
            assert service.prompts == ["Long task", "queued follow up"]
            assert app._steering_queue == []

    asyncio.run(run())


def test_tui_ctrl_g_steers_running_task(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            await pilot.pause(0.1)

            input_widget.value = "改成只分析不修改"
            await pilot.press("ctrl+g")
            await pilot.pause(0.1)

            assert service.steered_texts == ["改成只分析不修改"]
            assert input_widget.value == ""
            assert app._steering_queue == []
            assert "You (引导)" in _text(app, "#conversation")

            service.release.set()
            await pilot.pause(0.3)
            # 引导不追加新请求；本轮结束后队列为空、不自动重发。
            assert service.prompts == ["Long task"]

    asyncio.run(run())


def test_tui_ctrl_g_falls_back_to_queue_when_run_already_finished(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        service.steer_accepts = False
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "Long task"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)
            await pilot.pause(0.1)

            input_widget.value = "来不及引导的消息"
            await pilot.press("ctrl+g")
            await pilot.pause(0.1)

            assert service.steered_texts == []
            assert app._steering_queue == ["来不及引导的消息"]
            assert "You (已排队)" in _text(app, "#conversation")

            service.release.set()
            for _ in range(20):
                await pilot.pause(0.1)
                if len(service.prompts) >= 2:
                    break
            assert service.prompts == ["Long task", "来不及引导的消息"]

    asyncio.run(run())


def test_tui_ctrl_g_is_noop_when_idle(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "尚未发送的输入"
            await pilot.press("ctrl+g")
            await pilot.pause(0.1)

            assert service.steered_texts == []
            assert app._steering_queue == []
            assert input_widget.value == "尚未发送的输入"

    asyncio.run(run())
