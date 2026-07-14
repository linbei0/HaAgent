"""
src/haagent/tui/widgets/timeline_block.py - 对话时间线条目渲染块

封装 TimelineBlock（单条时间线条目的 Vertical 容器）、AssistantMarkdown（禁用悬浮表格）
和 ToolActivityLog（工具详情追加日志）。
"""

from __future__ import annotations

from functools import partial
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.await_complete import AwaitComplete
from textual.containers import Vertical
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import Log, Markdown, Static

from haagent.tui.presentation.progress import TimelinePresentationItem
from haagent.tui.widgets.timeline_models import (
    PROCESS_GROUP_ID_BASE,
    TimelineItem,
    TimelineRole,
    ToolActivity,
)
from haagent.tui.widgets.timeline_rendering import (
    render_tool_summary as _render_tool_summary,
    role_label as _role_label,
    timeline_item_body as _timeline_item_body,
)


class ToolActivityLog(Log):
    """Timeline 内嵌工具详情日志，避免整段 Static 文本反复重绘。"""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_lines", 32)
        kwargs.setdefault("auto_scroll", False)
        kwargs.setdefault("highlight", False)
        super().__init__(*args, **kwargs)
        self._rendered_text = ""

    @property
    def plain_text(self) -> str:
        return self._rendered_text

    def _render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        if y >= len(self._lines):
            return Strip([], 0)
        line = self._render_line_strip(y, self.rich_style)
        line = line.crop(scroll_x, scroll_x + width)
        return line.apply_offsets(scroll_x, y)

    def render_tools(self, tools: list[ToolActivity], *, show_details: bool) -> None:
        lines = _render_tool_summary(tools, show_details=show_details)
        self._rendered_text = "\n".join(lines)
        if not self.is_attached:
            return
        self.clear()
        if lines:
            self.write_lines(lines, scroll_end=False)


class AssistantMarkdown(Markdown):
    """对话区 Markdown：保留渲染，禁用表格悬停浮窗。"""

    def update(self, markdown: str) -> AwaitComplete:
        return AwaitComplete(self._clear_table_tooltips_after(super().update(markdown)))

    def append(self, markdown: str) -> AwaitComplete:
        return AwaitComplete(self._clear_table_tooltips_after(super().append(markdown)))

    async def _clear_table_tooltips_after(self, update: AwaitComplete) -> None:
        await update
        self.clear_table_tooltips()

    def clear_table_tooltips(self) -> None:
        for widget in self.walk_children():
            if widget.has_class("cell") or widget.has_class("header"):
                widget.tooltip = None


class TimelineBlock(Vertical):
    _STREAMING_INDICATOR_FRAMES = ("|", "/", "-", "\\")

    def __init__(self, item: TimelineItem, *, show_tool_details: bool) -> None:
        self._item = item
        self._show_tool_details = show_tool_details
        self._header_widget: Static | None = None
        self._body_widget: Markdown | Static | None = None
        self._active_widget: Static | None = None
        self._tools_widget: ToolActivityLog | None = None
        self._markdown_stream: Any | None = None
        self._markdown_stream_content = ""
        self._markdown_update_version = 0
        self._streaming_indicator_timer: Timer | None = None
        self._streaming_indicator_index = 0
        super().__init__(classes=_timeline_item_classes(item))

    def compose(self) -> ComposeResult:
        self._header_widget = Static("", classes="timeline-header")
        if self._item.role == "assistant":
            self._body_widget = AssistantMarkdown("", classes="timeline-body timeline-answer", open_links=False)
        else:
            self._body_widget = Static("", classes="timeline-body")
        self._active_widget = Static("", classes="timeline-active")
        self._tools_widget = ToolActivityLog(classes="timeline-tools")
        yield self._header_widget
        yield self._tools_widget
        yield self._body_widget
        yield self._active_widget

    def on_mount(self) -> None:
        self.update_item(self._item, show_tool_details=self._show_tool_details)

    async def on_unmount(self) -> None:
        self._stop_streaming_indicator()
        stream = self._markdown_stream
        self._markdown_stream = None
        self._markdown_stream_content = ""
        if stream is not None:
            await stream.stop()

    def on_click(self, event: events.Click) -> None:
        if not _is_clickable_item(self._item):
            return
        activate = getattr(self.parent, "activate_item", None)
        if not callable(activate):
            return
        event.stop()
        activate(self._item.item_id)

    def update_item(self, item: TimelineItem, *, show_tool_details: bool) -> None:
        self._item = item
        self._show_tool_details = show_tool_details
        header = self._header_widget
        body = self._body_widget
        active = self._active_widget
        tools = self._tools_widget
        if header is None or body is None or active is None or tools is None:
            return
        label = item.title or _role_label(item.role)
        header.update(f"[{label}]")
        header.display = True
        if isinstance(body, Markdown):
            self._update_markdown_body(body, item)
        else:
            body.update(_timeline_item_body(item))
        body.display = bool(_timeline_item_body(item))
        if item.role == "assistant" and item.status == "streaming":
            self._start_streaming_indicator()
            active_text = self._streaming_indicator_text()
        else:
            self._stop_streaming_indicator()
            active_text = ""
        active.update(active_text)
        active.display = bool(active_text)
        tools.render_tools(item.tools, show_details=self._show_tool_details)
        tools.display = bool(item.tools)
        # 清理所有 role/status class 后重新赋值，避免残留旧 class
        self.set_class(item.role == "assistant" and item.status == "streaming", "timeline-streaming")
        self.set_class(item.role == "failure" or item.status == "failed", "timeline-failed")
        self.set_class(item.role == "user", "timeline-user")
        self.set_class(item.role == "assistant", "timeline-assistant")
        self.set_class(item.role == "system", "timeline-system")
        self.set_class(item.role == "activity", "timeline-activity")
        self.set_class(item.role == "notice", "timeline-notice")
        self.set_class(item.role == "effect", "timeline-effect")
        self.set_class(item.role == "process", "timeline-process")

    def _update_markdown_body(self, body: Markdown, item: TimelineItem) -> None:
        content = item.content or ""
        if item.status == "streaming":
            if self._markdown_stream is None or not content.startswith(self._markdown_stream_content):
                self._stop_markdown_stream()
                self._queue_markdown_update(body, "")
                self._markdown_stream = Markdown.get_stream(body)
                self._markdown_stream_content = ""
            delta = content[len(self._markdown_stream_content) :]
            if delta:
                # write 不可 exclusive：否则新 delta 会取消未完成 write，中间字丢失。
                stream = self._markdown_stream
                assert stream is not None
                self.run_worker(
                    partial(stream.write, delta),
                    group="markdown-stream",
                    exit_on_error=False,
                )
                self._markdown_stream_content = content
                self.call_after_refresh(self._scroll_timeline_to_end)
            return
        # finalize exclusive：取消未完成 write，await stop 后再 update 权威全文，避免「。。」。
        stream = self._markdown_stream
        self._markdown_stream = None
        self._markdown_stream_content = ""
        self._markdown_update_version += 1
        version = self._markdown_update_version
        self.run_worker(
            partial(self._finalize_markdown_content, body, content, version, stream),
            group="markdown-stream",
            exclusive=True,
            exit_on_error=False,
        )

    def _queue_markdown_update(self, body: Markdown, content: str) -> None:
        self._markdown_update_version += 1
        version = self._markdown_update_version
        self.run_worker(
            partial(self._update_markdown_content, body, content, version),
            group="markdown-update",
            exit_on_error=False,
        )

    async def _update_markdown_content(self, body: Markdown, content: str, version: int) -> None:
        if version != self._markdown_update_version:
            return
        await body.update(content)
        self._scroll_timeline_to_end()
        self.call_after_refresh(self._scroll_timeline_to_end)
        self.call_after_refresh(lambda: self.call_after_refresh(self._scroll_timeline_to_end))

    async def _finalize_markdown_content(
        self,
        body: Markdown,
        content: str,
        version: int,
        stream: Any | None,
    ) -> None:
        # Textual MarkdownStream.stop 取消时仍会 append 未刷完的 pending；
        # 必须先等 stop 完成，再用 update 覆盖成权威全文，否则末尾会重复一字。
        if stream is not None:
            await stream.stop()
        if version != self._markdown_update_version:
            return
        # 即使 source 看似相同也 update：stop 尾刷可能刚叠出重复尾字。
        await body.update(content)
        self._scroll_timeline_to_end()
        self.call_after_refresh(self._scroll_timeline_to_end)
        self.call_after_refresh(lambda: self.call_after_refresh(self._scroll_timeline_to_end))

    def _stop_markdown_stream(self) -> None:
        if self._markdown_stream is None:
            return
        stream = self._markdown_stream
        self._markdown_stream = None
        self._markdown_stream_content = ""
        self.run_worker(partial(stream.stop), group="markdown-stream", exit_on_error=False)

    def _start_streaming_indicator(self) -> None:
        if self._streaming_indicator_timer is not None:
            return
        self._streaming_indicator_index = 0
        self._streaming_indicator_timer = self.set_interval(
            0.12,
            self._advance_streaming_indicator,
            name="streaming-indicator",
        )

    def _stop_streaming_indicator(self) -> None:
        if self._streaming_indicator_timer is None:
            return
        self._streaming_indicator_timer.stop()
        self._streaming_indicator_timer = None

    def _advance_streaming_indicator(self) -> None:
        if self._item.role != "assistant" or self._item.status != "streaming":
            self._stop_streaming_indicator()
            return
        active = self._active_widget
        if active is None:
            return
        self._streaming_indicator_index = (self._streaming_indicator_index + 1) % len(
            self._STREAMING_INDICATOR_FRAMES
        )
        active.update(self._streaming_indicator_text())

    def _streaming_indicator_text(self) -> str:
        frame = self._STREAMING_INDICATOR_FRAMES[self._streaming_indicator_index]
        if self.screen.has_class("theme-monochrome"):
            return f"{frame} 生成中"
        return frame

    def _scroll_timeline_to_end(self) -> None:
        # 延迟导入避免循环
        from haagent.tui.widgets.conversation_timeline import ConversationTimeline
        parent = self.parent
        if isinstance(parent, ConversationTimeline):
            parent.scroll_end_if_sticky()


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _timeline_item_classes(item: TimelineItem) -> str:
    role_class = f"timeline-{item.role}"
    classes = f"timeline-item {role_class}"
    if item.role == "assistant" and item.status == "streaming":
        classes = f"{classes} timeline-streaming"
    if item.role == "failure" or item.status == "failed":
        classes = f"{classes} timeline-failed"
    return classes


def _is_process_item(item: TimelineItem) -> bool:
    return item.role in {"activity", "effect", "process"}


def _is_clickable_item(item: TimelineItem) -> bool:
    return item.role == "process" or bool(item.detail_lines)


def _process_group_id(turn_index: int) -> int:
    return PROCESS_GROUP_ID_BASE - turn_index


def _is_process_group_id(item_id: int) -> bool:
    return item_id <= PROCESS_GROUP_ID_BASE


def _process_group_turn_index(item_id: int) -> int:
    return PROCESS_GROUP_ID_BASE - item_id


def _process_group_item(turn_index: int, count: int, *, expanded: bool) -> TimelineItem:
    arrow = "v" if expanded else ">"
    return TimelineItem(
        item_id=_process_group_id(turn_index),
        role="process",
        turn_index=turn_index,
        title=f"已处理 {count} 项 {arrow}",
        content="",
    )


def _presentation_role(item: TimelinePresentationItem) -> TimelineRole:
    if item.kind == "effect":
        return "effect"
    if item.kind == "activity":
        return "activity"
    return "notice"
