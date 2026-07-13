"""
tests/unit/tui/test_markdown_stream_finalize.py - 流式 Markdown 收尾不重复尾字

复现：streaming 末尾 delta 尚在 MarkdownStream 队列时 finalize，
若 stop 未与 update 串行，stop 会再 append 一次尾字，UI 出现「。。」。
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Markdown

from haagent.tui.widgets.timeline_block import TimelineBlock
from haagent.tui.widgets.timeline_models import TimelineItem


class _MarkdownFinalizeApp(App[None]):
    def compose(self) -> ComposeResult:
        item = TimelineItem(
            item_id=1,
            role="assistant",
            turn_index=1,
            content="",
            status="streaming",
            title="HaAgent",
        )
        yield TimelineBlock(item, show_tool_details=False)


def test_finalize_after_pending_stream_write_does_not_duplicate_trailing_period() -> None:
    """末尾句号只应出现一次；不得因 stream.stop 尾刷 + body.update 叠成两个。"""

    async def run() -> None:
        app = _MarkdownFinalizeApp()
        async with app.run_test() as pilot:
            block = app.query_one(TimelineBlock)
            body = app.query_one(Markdown)

            streaming = TimelineItem(
                item_id=1,
                role="assistant",
                turn_index=1,
                content="都可以直接告诉我",
                status="streaming",
                title="HaAgent",
            )
            block.update_item(streaming, show_tool_details=False)
            await pilot.pause()
            for _ in range(30):
                if body.source == "都可以直接告诉我":
                    break
                await asyncio.sleep(0.02)
            assert body.source == "都可以直接告诉我"

            with_period = TimelineItem(
                item_id=1,
                role="assistant",
                turn_index=1,
                content="都可以直接告诉我。",
                status="streaming",
                title="HaAgent",
            )
            block.update_item(with_period, show_tool_details=False)

            finalized = TimelineItem(
                item_id=1,
                role="assistant",
                turn_index=1,
                content="都可以直接告诉我。",
                status="done",
                title="HaAgent",
            )
            block.update_item(finalized, show_tool_details=False)

            for _ in range(50):
                if body.source.endswith("。。") or body.source == "都可以直接告诉我。":
                    await asyncio.sleep(0.05)
                    break
                await asyncio.sleep(0.02)

            assert body.source == "都可以直接告诉我。"
            assert not body.source.endswith("。。")

    asyncio.run(run())
