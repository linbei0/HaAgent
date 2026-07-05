"""
tests/unit/tui/test_tool_detail_budget.py - 工具详情原地预算测试

确保 /details 仍在主 timeline 中展示，但不会无限展开造成选中和滚动卡顿。
"""

from __future__ import annotations

import haagent.tui.widgets.timeline as timeline_module
from haagent.tui.widgets.timeline import ToolActivity
from textual._cells import cell_len
from textual.app import App, ComposeResult
from textual.widgets import Log


def test_tool_detail_renderer_keeps_recent_items_and_reports_collapsed_count() -> None:
    tools = [
        ToolActivity(
            tool_name=f"tool_{index}",
            status="done",
            summary=f"summary_{index}",
            turn_index=1,
            diagnostics=[f"diagnostic_{index}"],
        )
        for index in range(12)
    ]

    lines = timeline_module._render_tool_summary(tools, show_details=True)
    text = "\n".join(lines)

    assert "已折叠 4 条较早工具详情" in text
    assert "tool_0" not in text
    assert "diagnostic_0" not in text
    assert "tool_11" in text
    assert "diagnostic_11" in text


def test_tool_activity_log_uses_textual_log_with_budgeted_lines() -> None:
    log_cls = getattr(timeline_module, "ToolActivityLog", None)

    assert log_cls is not None
    log = log_cls()
    assert isinstance(log, Log)
    assert log.max_lines == 32
    assert log.auto_scroll is False
    assert log.show_horizontal_scrollbar is False
    assert log.show_vertical_scrollbar is False

    tools = [
        ToolActivity(
            tool_name=f"tool_{index}",
            status="done",
            summary=f"summary_{index}",
            turn_index=1,
            diagnostics=[f"diagnostic_{index}"],
        )
        for index in range(12)
    ]
    log.render_tools(tools, show_details=True)
    text = log.plain_text

    assert "已折叠 4 条较早工具详情" in text
    assert "tool_0" not in text
    assert "tool_11" in text


def test_tool_activity_log_does_not_paint_trailing_empty_cells() -> None:
    class ToolLogApp(App[None]):
        def compose(self) -> ComposeResult:
            yield timeline_module.ToolActivityLog()

    async def run() -> None:
        app = ToolLogApp()
        async with app.run_test(size=(80, 20)):
            log = app.query_one(timeline_module.ToolActivityLog)
            line = "工具 web_search ok"
            log.write_line(line, scroll_end=False)
            strip = log._render_line(0, 0, 80)

            assert strip.cell_length == cell_len(line)

    import asyncio

    asyncio.run(run())
