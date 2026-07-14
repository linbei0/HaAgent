"""
tests/tui/test_app_misc.py - HaAgent TUI misc 集成测试

从 test_app.py 按领域拆分；共享 Fake 与 helpers 见 support.py。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from haagent.tui.application.app import HaAgentTuiApp
from haagent.tui.widgets import ConversationTimeline

from tests.tui.support import FakeAssistantService, _tool_event

def test_tui_timeline_uses_distinct_message_widgets_for_visual_hierarchy(tmp_path: Path) -> None:
    async def run() -> None:
        service = FakeAssistantService(
            workspace_root=tmp_path,
            extra_events=[
                _tool_event("tool_started", 1, "web_search"),
                _tool_event("tool_finished", 1, "web_search"),
            ],
            assistant_content="已经完成搜索。",
        )
        app = HaAgentTuiApp(service)
        async with app.run_test(size=(120, 40)) as pilot:
            input_widget = app.query_one("#prompt-input")
            input_widget.value = "搜索资料"
            await pilot.press("enter")
            await pilot.pause(0.2)

            conversation = app.query_one("#conversation", ConversationTimeline)
            assert conversation.has_class("timeline-ready")
            assert list(conversation.query(".timeline-item"))
            assert list(conversation.query(".timeline-user"))
            assert list(conversation.query(".timeline-assistant"))
            assert list(conversation.query(".timeline-tools"))
            assert list(conversation.query(".timeline-body"))

    asyncio.run(run())

