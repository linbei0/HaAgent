"""
haagent/tui/widgets.py - TUI 基础组件

封装输入、状态栏、结构化对话时间线、footer 和尺寸提示等稳定 UI 区域。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Any, Literal

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Markdown, Static, TextArea


class PromptInput(TextArea):
    BINDINGS = [
        Binding("enter", "submit_from_input", "发送", priority=True),
        Binding("shift+enter", "insert_newline_from_input", "换行", priority=True),
        Binding("ctrl+f", "open_search_from_input", "搜索", priority=True),
        Binding("ctrl+x", "cancel_current_task_from_input", "取消任务", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, input: PromptInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(text)
        self.move_cursor(_end_location(text))

    def on_key(self, event: events.Key) -> None:
        app = self.app
        if getattr(app, "command_suggestions_is_open", lambda: False)() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_command_suggestion_key(event)
            return
        if getattr(app, "file_reference_is_open", lambda: False)() and event.key in {"escape", "up", "down", "enter"}:
            event.prevent_default()
            app.action_handle_file_ref_key(event)
            return
        if getattr(app, "_memory_mode", False) and getattr(app, "_pending_interaction", None) is None:
            handled = False
            if event.key == "enter":
                app.action_memory_enter()
                handled = True
            elif event.key in {"a", "y"}:
                app.action_confirm_memory()
                handled = True
            elif event.key == "r":
                app.action_reject_memory()
                handled = True
            else:
                handled = app._handle_memory_key(event.key)
            if handled:
                event.stop()
                event.prevent_default()
                return
        if event.key == "@" or event.character == "@":
            event.stop()
            event.prevent_default()
            self.insert("@")
            app.action_open_file_refs()
            return
        if event.key in {"/", "slash"} or event.character == "/":
            event.stop()
            event.prevent_default()
            self.insert("/")
            if self.value == "/":
                app.action_open_command_suggestions()
            return
        if self.value:
            return
        if event.key in {"?", "question_mark"} or event.character == "?":
            event.stop()
            app.action_help()
            return

    def action_open_search_from_input(self) -> None:
        self.app.action_open_search()

    def action_cancel_current_task_from_input(self) -> None:
        self.app.action_cancel_current_task()

    def action_submit_from_input(self) -> None:
        if getattr(self.app, "command_suggestions_is_open", lambda: False)():
            self.app.action_accept_command_suggestion()
            return
        if getattr(self.app, "file_reference_is_open", lambda: False)():
            self.app.action_accept_file_ref()
            return
        if getattr(self.app, "_memory_mode", False):
            self.app.action_memory_enter()
            return
        self.app.action_submit_prompt()

    def action_insert_newline_from_input(self) -> None:
        self.insert("\n")


class StatusBar(Static):
    def update_status(self, text: str) -> None:
        self.update(text)


TimelineRole = Literal["user", "assistant", "system", "failure"]
TimelineStatus = Literal["streaming", "done", "failed"]
ToolStatus = Literal["running", "approval", "done", "failed"]


@dataclass
class ToolActivity:
    tool_name: str
    status: ToolStatus
    summary: str
    turn_index: int
    diagnostics: list[str] = field(default_factory=list)


@dataclass
class TimelineItem:
    item_id: int
    role: TimelineRole
    turn_index: int
    content: str
    status: TimelineStatus = "done"
    title: str | None = None
    tools: list[ToolActivity] = field(default_factory=list)


class ConversationTimeline(VerticalScroll):
    _ids = count(1)

    def __init__(self, *args, **kwargs) -> None:
        kwargs.pop("wrap", None)
        kwargs.pop("auto_scroll", None)
        super().__init__(*args, **kwargs)
        self._items: list[TimelineItem] = []
        self._tool_details_enabled = False
        self._plain_text = ""
        self._blocks: dict[int, TimelineBlock] = {}
        self._placeholder_widget: Static | None = None
        self._memory_widget: Static | None = None

    def show_placeholder(self) -> None:
        self._items.clear()
        self._plain_text = "Ready. 输入消息后按 Enter 发送；Shift+Enter 换行；Ctrl+Q 退出。"
        self._show_singleton("placeholder", "Ready. 输入消息后按 Enter 发送；Shift+Enter 换行；Ctrl+Q 退出。", "timeline-placeholder")

    def show_memory(self, text: str) -> None:
        self._items.clear()
        self._plain_text = text
        self._show_singleton("memory", text, "timeline-memory")

    def append_lines(self, lines: list[str], *, start: int) -> None:
        for line in lines[start:]:
            self.add_system("系统", line)

    @property
    def plain_text(self) -> str:
        return self._plain_text

    def clear_timeline(self) -> None:
        self._items.clear()
        self._plain_text = ""
        self._remove_singletons()
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()

    def set_tool_details(self, enabled: bool) -> None:
        self._tool_details_enabled = enabled
        self._render_timeline()

    def add_user(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="user", turn_index=turn_index, content=content, title="你"))
        self._render_timeline()

    def add_system(self, title: str, content: str, *, turn_index: int = 0) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="system", turn_index=turn_index, content=content, title=title))
        self._render_timeline()

    def add_failure(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="failure", turn_index=turn_index, content=content, status="failed", title="失败"))
        self._render_timeline()

    def add_assistant_message(self, content: str, *, turn_index: int) -> None:
        self._items.append(TimelineItem(item_id=next(self._ids), role="assistant", turn_index=turn_index, content=content, status="done", title="HaAgent"))
        self._render_timeline()

    def update_assistant_delta(self, turn_index: int, delta: str) -> None:
        if not delta:
            return
        item = self._assistant_item(turn_index)
        item.content += delta
        item.status = "streaming"
        self._sync_block(item)
        self._sync_plain_text()

    def finalize_assistant(self, turn_index: int, content: str) -> None:
        item = self._assistant_item(turn_index)
        if content:
            item.content = content
        item.status = "done"
        self._sync_block(item)
        self._sync_plain_text()

    def add_tool_activity(self, activity: ToolActivity) -> None:
        item = self._assistant_item(activity.turn_index)
        merge_tool_activity(item.tools, activity)
        self._sync_block(item)
        self._sync_plain_text()

    def add_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        item = self._assistant_item(turn_index)
        activity = _matching_latest_tool_activity(item.tools, tool_name)
        if activity is None:
            activity = ToolActivity(tool_name=tool_name, status="done", summary="诊断", turn_index=turn_index)
            item.tools.append(activity)
        if message not in activity.diagnostics:
            activity.diagnostics.append(message)
        self._sync_block(item)
        self._sync_plain_text()

    def _assistant_item(self, turn_index: int) -> TimelineItem:
        for item in self._items:
            if item.role == "assistant" and item.turn_index == turn_index:
                return item
        item = TimelineItem(item_id=next(self._ids), role="assistant", turn_index=turn_index, content="", status="streaming", title="HaAgent")
        self._items.append(item)
        return item

    def _render_timeline(self) -> None:
        self._sync_blocks()
        self._sync_plain_text()

    def _sync_block(self, item: TimelineItem) -> None:
        block = self._blocks.get(item.item_id)
        if block is None:
            self._render_timeline()
            return
        block.update_item(item, show_tool_details=self._tool_details_enabled)
        self.set_class(bool(self._items), "timeline-ready")
        self.call_after_refresh(self.scroll_end, animate=False)

    def _sync_blocks(self) -> None:
        self._remove_singletons()
        seen: set[int] = set()
        anchor: Vertical | Static | None = None
        for item in self._items:
            block = self._blocks.get(item.item_id)
            if block is None:
                block = TimelineBlock(item, show_tool_details=self._tool_details_enabled)
                self._blocks[item.item_id] = block
                if anchor is None:
                    self.mount(block)
                else:
                    self.mount(block, after=anchor)
            else:
                block.update_item(item, show_tool_details=self._tool_details_enabled)
                if block.parent is None:
                    if anchor is None:
                        self.mount(block)
                    else:
                        self.mount(block, after=anchor)
            seen.add(item.item_id)
            anchor = block
        for turn_index in list(self._blocks):
            if turn_index not in seen:
                block = self._blocks.pop(turn_index)
                block.remove()
        self.set_class(bool(self._items), "timeline-ready")
        self.call_after_refresh(self.scroll_end, animate=False)

    def _sync_plain_text(self) -> None:
        self._plain_text = "\n\n".join(_render_timeline_item(item, show_tool_details=self._tool_details_enabled) for item in self._items)

    def _show_singleton(self, kind: str, text: str, classes: str) -> None:
        self._remove_singletons()
        for block in list(self._blocks.values()):
            block.remove()
        self._blocks.clear()
        existing = Static(text, classes=classes)
        self.mount(existing)
        if kind == "placeholder":
            self._placeholder_widget = existing
        else:
            self._memory_widget = existing
        self.set_class(False, "timeline-ready")
        self.call_after_refresh(self.scroll_end, animate=False)

    def _remove_singletons(self) -> None:
        for widget in (self._placeholder_widget, self._memory_widget):
            if widget is not None:
                widget.remove()
        self._placeholder_widget = None
        self._memory_widget = None


ConversationView = ConversationTimeline


def _render_timeline_item(item: TimelineItem, *, show_tool_details: bool) -> str:
    label = item.title or _role_label(item.role)
    marker = _role_marker(item.role, item.status)
    lines = [f"{marker} [{label}]"]
    if item.content:
        lines.extend(item.content.splitlines())
    if item.role == "assistant" and item.status == "streaming":
        lines.append("  生成中 · HaAgent")
    if item.tools:
        lines.extend(_render_tool_summary(item.tools, show_details=show_tool_details))
    return "\n".join(lines)


class TimelineBlock(Vertical):
    _STREAMING_INDICATOR_FRAMES = ("|", "/", "-", "\\")

    def __init__(self, item: TimelineItem, *, show_tool_details: bool) -> None:
        self._item = item
        self._show_tool_details = show_tool_details
        self._header_widget: Static | None = None
        self._body_widget: Markdown | Static | None = None
        self._active_widget: Static | None = None
        self._tools_widget: Static | None = None
        self._markdown_stream: Any | None = None
        self._markdown_stream_content = ""
        self._streaming_indicator_timer: Timer | None = None
        self._streaming_indicator_index = 0
        super().__init__(classes=_timeline_item_classes(item))

    def compose(self) -> ComposeResult:
        self._header_widget = Static("", classes="timeline-header")
        if self._item.role == "assistant":
            self._body_widget = Markdown("", classes="timeline-body timeline-answer", open_links=False)
        else:
            self._body_widget = Static("", classes="timeline-body")
        self._active_widget = Static("", classes="timeline-active")
        self._tools_widget = Static("", classes="timeline-tools")
        yield self._header_widget
        yield self._body_widget
        yield self._active_widget
        yield self._tools_widget

    def on_mount(self) -> None:
        self.update_item(self._item, show_tool_details=self._show_tool_details)

    def on_unmount(self) -> None:
        self._stop_streaming_indicator()

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
            body.update(item.content or "")
        body.display = bool(item.content)
        if item.role == "assistant" and item.status == "streaming":
            self._start_streaming_indicator()
            active_text = self._streaming_indicator_text()
        else:
            self._stop_streaming_indicator()
            active_text = ""
        active.update(active_text)
        active.display = bool(active_text)
        tool_text = "\n".join(_render_tool_summary(item.tools, show_details=self._show_tool_details))
        tools.update(tool_text)
        tools.display = bool(tool_text)
        self.set_class(item.role == "assistant" and item.status == "streaming", "timeline-streaming")
        self.set_class(item.role == "failure" or item.status == "failed", "timeline-failed")
        self.set_class(item.role == "user", "timeline-user")
        self.set_class(item.role == "assistant", "timeline-assistant")
        self.set_class(item.role == "system", "timeline-system")
        self.set_class(item.role == "failure" or item.status == "failed", "timeline-failed")

    def _update_markdown_body(self, body: Markdown, item: TimelineItem) -> None:
        content = item.content or ""
        if item.status == "streaming":
            if self._markdown_stream is None or not content.startswith(self._markdown_stream_content):
                self._stop_markdown_stream()
                self.run_worker(
                    self._update_markdown_content(body, ""),
                    group="markdown-update",
                    exit_on_error=False,
                    exclusive=True,
                )
                self._markdown_stream = Markdown.get_stream(body)
                self._markdown_stream_content = ""
            delta = content[len(self._markdown_stream_content) :]
            if delta:
                self.run_worker(
                    self._markdown_stream.write(delta),
                    group="markdown-stream",
                    exit_on_error=False,
                )
                self._markdown_stream_content = content
                self.call_after_refresh(self._scroll_timeline_to_end)
            return
        self._stop_markdown_stream()
        if body.source != content:
            self.run_worker(
                self._update_markdown_content(body, content),
                group="markdown-update",
                exit_on_error=False,
                exclusive=True,
            )

    async def _update_markdown_content(self, body: Markdown, content: str) -> None:
        await body.update(content)
        self._scroll_timeline_to_end()
        self.call_after_refresh(self._scroll_timeline_to_end)

    def _stop_markdown_stream(self) -> None:
        if self._markdown_stream is None:
            return
        stream = self._markdown_stream
        self._markdown_stream = None
        self._markdown_stream_content = ""
        self.run_worker(stream.stop(), group="markdown-stream", exit_on_error=False)

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
        parent = self.parent
        if isinstance(parent, ConversationTimeline):
            parent.scroll_end(animate=False)


def _timeline_item_classes(item: TimelineItem) -> str:
    role_class = f"timeline-{item.role}"
    classes = f"timeline-item {role_class}"
    if item.role == "assistant" and item.status == "streaming":
        classes = f"{classes} timeline-streaming"
    if item.role == "failure" or item.status == "failed":
        classes = f"{classes} timeline-failed"
    return classes


def _render_tool_summary(tools: list[ToolActivity], *, show_details: bool) -> list[str]:
    if not tools:
        return []
    counts = {
        "running": sum(1 for item in tools if item.status == "running"),
        "approval": sum(1 for item in tools if item.status == "approval"),
        "done": sum(1 for item in tools if item.status == "done"),
        "failed": sum(1 for item in tools if item.status == "failed"),
    }
    summary_parts = [f"工具 {len(tools)} 项"]
    if counts["done"]:
        summary_parts.append(f"{counts['done']} 成功")
    if counts["running"]:
        summary_parts.append(f"{counts['running']} 运行中")
    if counts["approval"]:
        summary_parts.append(f"{counts['approval']} 待确认")
    if counts["failed"]:
        summary_parts.append(f"{counts['failed']} 失败")
    lines = [f"  [工具] {' · '.join(summary_parts)}"]
    if show_details or len(tools) <= 4:
        visible_tools = tools
        for item in visible_tools:
            lines.append(f"    - 工具 {item.tool_name} {_tool_legacy_status(item.status)} · {item.summary}")
            if show_details:
                lines.extend(f"      诊断：{diagnostic}" for diagnostic in item.diagnostics)
    else:
        visible_tools = [item for item in tools if item.status in {"running", "approval", "failed"}]
        if visible_tools:
            compact_names = _unique_tool_names(visible_tools, limit=3)
            lines.append(f"  当前：{'、'.join(compact_names)}")
    return lines


def merge_tool_activity(tools: list[ToolActivity], activity: ToolActivity) -> None:
    existing_activity = _matching_open_tool_activity(tools, activity)
    if existing_activity is None:
        tools.append(activity)
        return
    existing_activity.status = activity.status
    existing_activity.summary = activity.summary
    for diagnostic in activity.diagnostics:
        if diagnostic not in existing_activity.diagnostics:
            existing_activity.diagnostics.append(diagnostic)


def _matching_open_tool_activity(tools: list[ToolActivity], activity: ToolActivity) -> ToolActivity | None:
    if activity.status == "running":
        candidate_statuses: set[ToolStatus] = {"approval"}
    else:
        candidate_statuses = {"running", "approval"}
        if activity.status == "failed":
            candidate_statuses.add("failed")
    for item in reversed(tools):
        if item.tool_name == activity.tool_name and item.status in candidate_statuses:
            return item
    return None


def _matching_latest_tool_activity(tools: list[ToolActivity], tool_name: str) -> ToolActivity | None:
    for item in reversed(tools):
        if item.tool_name == tool_name:
            return item
    return None


def _role_label(role: TimelineRole) -> str:
    return {
        "user": "你",
        "assistant": "HaAgent",
        "system": "系统",
        "failure": "失败",
    }[role]


def _role_marker(role: TimelineRole, status: TimelineStatus) -> str:
    if role == "user":
        return ">"
    if role == "failure" or status == "failed":
        return "!"
    if role == "assistant" and status == "streaming":
        return ">>"
    return "|"


def _tool_legacy_status(status: ToolStatus) -> str:
    return {
        "running": "... 运行中 (running)",
        "approval": "? 待审批",
        "done": "ok 成功",
        "failed": "! 失败 (failed)",
    }[status]


def _unique_tool_names(items: list[ToolActivity], *, limit: int) -> list[str]:
    names: list[str] = []
    for item in reversed(items):
        if item.tool_name not in names:
            names.append(item.tool_name)
        if len(names) == limit:
            break
    return list(reversed(names))


class FooterBar(Static):
    def update_footer(self, text: str) -> None:
        self.update(Text(text))


class ResizeMessage(Static):
    pass


def _end_location(text: str) -> tuple[int, int]:
    lines = text.split("\n")
    return (len(lines) - 1, len(lines[-1]))
    class Submitted(events.Message):
        def __init__(self, input: PromptInput, value: str) -> None:
            self.input = input
            self.value = value
            super().__init__()
