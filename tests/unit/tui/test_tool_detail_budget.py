"""
tests/unit/tui/test_tool_detail_budget.py - 工具详情原地预算测试

确保 /details 仍在主 timeline 中展示，但不会无限展开造成选中和滚动卡顿。
"""

from __future__ import annotations

from haagent.tui.widgets.timeline_block import ToolActivityLog
from haagent.tui.widgets.timeline_models import ToolActivity
from haagent.tui.widgets.timeline_rendering import render_tool_summary
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

    lines = render_tool_summary(tools, show_details=True)
    text = "\n".join(lines)

    assert "已折叠 4 条较早工具详情" in text
    assert "tool_0" not in text
    assert "diagnostic_0" not in text
    assert "tool_11" in text
    assert "diagnostic_11" in text


def test_tool_summary_uses_chinese_names_and_collapses_completed_items() -> None:
    tools = [
        ToolActivity("file_read", "done", "读取 pyproject.toml", 1),
        ToolActivity("shell", "done", "运行 pytest", 1),
    ]

    compact = "\n".join(render_tool_summary(tools, show_details=False))
    process_list = "\n".join(render_tool_summary(tools, show_details=False, list_names=True))
    details = "\n".join(render_tool_summary(tools, show_details=True))

    assert compact == "  已完成 2 项 ›"
    assert process_list == "  读取文件 · 运行命令 ›"
    assert "file_read" not in compact
    assert "读取文件（file_read）" in details
    assert "运行命令（shell）" in details


def test_running_and_failed_tool_summaries_remain_visible() -> None:
    running = "\n".join(
        render_tool_summary(
            [ToolActivity("file_read", "running", "正在读取", 1)],
            show_details=False,
        ),
    )
    failed = "\n".join(
        render_tool_summary(
            [ToolActivity("web_search", "failed", "请求超时", 1)],
            show_details=False,
        ),
    )

    assert running == "  正在读取文件 · 1 项"
    assert failed == "  联网搜索失败"
    assert "web_search" not in failed


def test_tool_activity_log_uses_textual_log_with_budgeted_lines() -> None:
    log_cls = ToolActivityLog

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
            yield ToolActivityLog()

    async def run() -> None:
        app = ToolLogApp()
        async with app.run_test(size=(80, 20)):
            log = app.query_one(ToolActivityLog)
            line = "工具 web_search ok"
            log.write_line(line, scroll_end=False)
            strip = log._render_line(0, 0, 80)

            assert strip.cell_length == cell_len(line)

    import asyncio

    asyncio.run(run())
