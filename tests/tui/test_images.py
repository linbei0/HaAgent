"""
tests/tui/test_images.py - HaAgent TUI images 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp

from tests.tui.support import FakeAssistantService, _text

def test_tui_ctrl_v_queues_image_token_in_prompt_and_next_prompt_sends_then_clears(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, assistant_content="收到图片。")
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            await pilot.press("ctrl+v")
            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            conversation = _text(app, "#conversation")
            assert "已添加图片附件" not in conversation
            assert "待发送附件" not in conversation
            assert input_widget.value == "[image 1] [image 2]"

            input_widget.value = f"{input_widget.value} 描述这张图"
            await pilot.press("enter")
            await pilot.pause(0.2)

            assert service.prompts == ["描述这张图"]
            assert service.prompt_attachments == [[service.clipboard_attachments[0], service.clipboard_attachments[1]]]
            assert app._attachments.pending == []
            assert input_widget.value == ""
            conversation_after_send = _text(app, "#conversation")
            assert "[image 1] [image 2] 描述这张图" in conversation_after_send
            assert "待发送附件" not in conversation_after_send

    asyncio.run(run())

def test_tui_ctrl_v_image_paste_is_rejected_while_running(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(workspace_root=tmp_path, block_until_released=True)
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "长任务"
            await pilot.press("enter")
            await asyncio.to_thread(service.started.wait, 2)

            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            assert service.clipboard_attachments == []
            assert "运行中不能修改待发送附件" in _text(app, "#conversation")
            service.release.set()
            await pilot.pause(0.2)

    asyncio.run(run())

def test_tui_image_prompt_requires_vision_capable_model(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            model="deepseek-chat",
            image_input_supported=False,
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            await pilot.press("ctrl+v")
            await pilot.pause(0.1)

            input_widget.value = f"{input_widget.value} 厉害吗"
            await pilot.press("enter")
            await pilot.pause(0.1)

            assert service.prompts == []
            assert service.prompt_attachments == []
            assert input_widget.value == "[image 1] 厉害吗"
            conversation = _text(app, "#conversation")
            assert "当前模型不支持图片输入" in conversation
            assert "deepseek-chat" in conversation

    asyncio.run(run())

