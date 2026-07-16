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
from textual.containers import Horizontal, Vertical
from textual.strip import Strip
from textual.timer import Timer
from textual.widgets import Button, Label, Log, Markdown, Static
from textual.widgets.markdown import MarkdownBlock, MarkdownFence

from haagent.tui.presentation.progress import TimelinePresentationItem
from haagent.tui.widgets.timeline_models import (
    PROCESS_GROUP_ID_BASE,
    TimelineItem,
    TimelineRole,
    ToolActivity,
)
from haagent.tui.widgets.timeline_rendering import (
    process_group_title as _process_group_title,
    render_tool_summary as _render_tool_summary,
    timeline_item_body as _timeline_item_body,
    timeline_item_label as _timeline_item_label,
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

    def render_tools(self, tools: list[ToolActivity], *, show_details: bool, list_names: bool = False) -> None:
        lines = _render_tool_summary(tools, show_details=show_details, list_names=list_names)
        self._rendered_text = "\n".join(lines)
        if not self.is_attached:
            return
        self.clear()
        if lines:
            self.write_lines(lines, scroll_end=False)


class CopyButton(Button):
    """复制操作按钮；成功反馈留在按钮内，不污染对话流。"""

    def __init__(self, label: str, *, classes: str) -> None:
        super().__init__(label, classes=f"copy-button {classes}", compact=True)
        self._default_label = label
        self._feedback_timer: Timer | None = None

    def show_copied(self) -> None:
        if self._feedback_timer is not None:
            self._feedback_timer.stop()
        self.label = "已复制"
        self._feedback_timer = self.set_timer(1.5, self._restore_label)

    def _restore_label(self) -> None:
        self._feedback_timer = None
        self.label = self._default_label


class CopyableMarkdownFence(MarkdownFence):
    """保留 Textual 代码高亮，并提供当前代码块的独立复制按钮。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._copy_enabled = False
        self._copy_button: CopyButton | None = None
        self._language_widget: Static | None = None
        self.add_class("copyable-code-block")

    def compose(self) -> ComposeResult:
        markdown = self._markdown_ref()
        self._copy_enabled = bool(getattr(markdown, "copy_enabled", False))
        self._copy_button = CopyButton("复制代码", classes="code-copy-button")
        self._copy_button.display = self._copy_enabled
        self._language_widget = Static((self.lexer or "代码").strip(), classes="code-block-language")
        toolbar = Horizontal(classes="code-block-toolbar")
        toolbar.styles.height = 1
        with toolbar:
            yield self._language_widget
            yield self._copy_button
        yield Label(self._highlighted_code, id="code-content", expand=True)

    async def _update_from_block(self, block: MarkdownBlock) -> None:
        await super()._update_from_block(block)
        if self._language_widget is not None:
            self._language_widget.update((self.lexer or "代码").strip())

    def set_copy_enabled(self, enabled: bool) -> None:
        self._copy_enabled = enabled
        if self._copy_button is not None:
            self._copy_button.display = enabled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is not self._copy_button or not self._copy_enabled:
            return
        event.stop()
        self.app.copy_to_clipboard(self.code)
        self._copy_button.show_copied()


class AssistantMarkdown(Markdown):
    """对话区 Markdown：保留渲染，禁用表格悬停浮窗。"""

    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": CopyableMarkdownFence,
        "code_block": CopyableMarkdownFence,
    }

    def __init__(self, *args, **kwargs) -> None:
        self._copy_enabled = False
        super().__init__(*args, **kwargs)

    @property
    def copy_enabled(self) -> bool:
        return self._copy_enabled

    def set_copy_enabled(self, enabled: bool) -> None:
        self._copy_enabled = enabled
        for fence in self.query(CopyableMarkdownFence):
            fence.set_copy_enabled(enabled)

    def update(self, markdown: str) -> AwaitComplete:
        return AwaitComplete(self._clear_table_tooltips_after(super().update(markdown)))

    def append(self, markdown: str) -> AwaitComplete:
        return AwaitComplete(self._clear_table_tooltips_after(super().append(markdown)))

    async def _clear_table_tooltips_after(self, update: AwaitComplete) -> None:
        await update
        self.clear_table_tooltips()
        for fence in self.query(CopyableMarkdownFence):
            fence.set_copy_enabled(self._copy_enabled)

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
        self._answer_actions_widget: Horizontal | None = None
        self._answer_copy_button: CopyButton | None = None
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
            self._answer_copy_button = CopyButton("复制回答", classes="answer-copy-button")
            self._answer_actions_widget = Horizontal(
                self._answer_copy_button,
                classes="answer-copy-actions",
            )
            # 组件也会在不加载主 TCSS 的测试/嵌入 App 中使用，必须自行约束伸展高度。
            self._answer_actions_widget.styles.height = 1
        else:
            self._body_widget = Static("", classes="timeline-body")
        self._active_widget = Static("", classes="timeline-active")
        self._tools_widget = ToolActivityLog(classes="timeline-tools")
        yield self._header_widget
        yield self._tools_widget
        yield self._body_widget
        if self._answer_actions_widget is not None:
            yield self._answer_actions_widget
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

    def on_mouse_down(self, event: events.MouseDown) -> None:
        # WezTerm/Windows 下 MouseDown/MouseUp 可到达但 Click 不一定被合成；
        # 在可交互摘要上直接响应左键，Shift+左键继续交给终端文本选择。
        if event.button != 1 or event.shift or not _is_clickable_item(self._item):
            return
        activate = getattr(self.parent, "activate_item", None)
        if not callable(activate):
            return
        event.stop()
        activate(self._item.item_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is not self._answer_copy_button:
            return
        if self._item.role != "assistant" or self._item.status != "done" or not self._item.content:
            return
        event.stop()
        self.app.copy_to_clipboard(self._item.content)
        self._answer_copy_button.show_copied()

    def update_item(self, item: TimelineItem, *, show_tool_details: bool) -> None:
        self._item = item
        self._show_tool_details = show_tool_details
        header = self._header_widget
        body = self._body_widget
        active = self._active_widget
        tools = self._tools_widget
        if header is None or body is None or active is None or tools is None:
            return
        # 过程子项不显示空泛「过程」标签；仅分组头「已完成 N 项」等有标题时展示 header。
        label = _timeline_item_label(item)
        header.update(label)
        header.display = bool(label)
        if isinstance(body, Markdown):
            self._update_markdown_body(body, item)
            if isinstance(body, AssistantMarkdown):
                body.set_copy_enabled(item.status == "done" and bool(item.content))
        else:
            body.update(_timeline_item_body(item))
        body.display = bool(_timeline_item_body(item))
        copy_answer = item.role == "assistant" and item.status == "done" and bool(item.content)
        if self._answer_actions_widget is not None:
            self._answer_actions_widget.display = copy_answer
        if self._answer_copy_button is not None:
            self._answer_copy_button.display = copy_answer
        if item.role == "assistant" and item.status == "streaming":
            self._start_streaming_indicator()
            active_text = self._streaming_indicator_text()
        else:
            self._stop_streaming_indicator()
            active_text = ""
        active.update(active_text)
        active.display = bool(active_text)
        tools.render_tools(
            item.tools,
            show_details=self._show_tool_details,
            list_names=item.role == "process",
        )
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
    # 失败与任务受阻通知同样属于本轮过程：运行时展开供用户处理，
    # 最终回答到达后由 ConversationTimeline 统一折叠进过程组。
    if item.detail_id and item.detail_id.startswith(("approval:", "input:")):
        return False
    return item.role in {"activity", "effect", "process", "notice", "failure"}


def _is_clickable_item(item: TimelineItem) -> bool:
    return _is_process_group_id(item.item_id) or bool(item.detail_lines)


def _process_group_id(turn_index: int) -> int:
    return PROCESS_GROUP_ID_BASE - turn_index


def _is_process_group_id(item_id: int) -> bool:
    return item_id <= PROCESS_GROUP_ID_BASE


def _process_group_turn_index(item_id: int) -> int:
    return PROCESS_GROUP_ID_BASE - item_id


def _process_group_item(
    turn_index: int,
    process_items: list[TimelineItem],
    *,
    expanded: bool,
    elapsed_seconds: float | None,
) -> TimelineItem:
    return TimelineItem(
        item_id=_process_group_id(turn_index),
        role="process",
        turn_index=turn_index,
        title=_process_group_title(process_items, expanded=expanded, elapsed_seconds=elapsed_seconds),
        content="",
    )


def _presentation_role(item: TimelinePresentationItem) -> TimelineRole:
    if item.kind == "effect":
        return "effect"
    if item.kind == "activity":
        return "activity"
    return "notice"
