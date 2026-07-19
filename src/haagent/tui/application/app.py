"""
haagent/tui/application/app.py - HaAgent TUI 应用编排

组合 Textual 组件、连接 AssistantService 与各 controller/flow，并提供顶层生命周期
和 Textual binding 入口。具体业务流程（模型连接、会话、记忆、命令、附件、时间线）
均迁出到同级 controller/flow 模块，本文件只做薄分发。
"""

from __future__ import annotations

from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.geometry import Offset
from textual.screen import Screen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import TextArea

from haagent.app.assistant_service import AssistantService
from haagent.runtime.events import ContextUsageEvent, RuntimeUiEvent
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.runtime.session.attachments import ImageAttachment
from haagent.tui.application.attachments import AttachmentController, prompt_without_image_tokens
from haagent.tui.application.channel_flow import ChannelFlow
from haagent.tui.application.command_handlers import ChatCommandHandlers
from haagent.tui.application.commands import CommandDispatcher
from haagent.tui.application.completion_flow import CompletionFlow
from haagent.tui.application.conversation import ConversationController
from haagent.tui.application.memory_flow import MemoryFlow
from haagent.tui.application.model_flow import ModelFlow
from haagent.tui.application.runtime_events import handle_runtime_ui_event
from haagent.tui.application.schedule_flow import ScheduleFlow
from haagent.tui.application.session_flow import SessionFlow
from haagent.tui.commands import command_registry, is_prompt_mode_command, parse_slash_command
from haagent.tui.design.failures import FailureView, failure_from_payload
from haagent.tui.design.keys import APP_BINDINGS, footer_text
from haagent.tui.design.renderers import context_usage_line, status_line
from haagent.tui.design.theme import (
    next_theme,
    select_theme,
    textual_themes,
)
from haagent.tui.design.utils import safe_summary
from haagent.tui.files.refs import FileReferenceIndex, build_file_reference_index
from haagent.tui.flows import permissions, skills
from haagent.tui.overlays.modals import EditDiffModal, HelpModal, ToolApprovalModal
from haagent.tui.overlays.search import SearchOverlay
from haagent.tui.overlays.sessions import SessionOverlayResult
from haagent.tui.presentation.progress import ProgressStatusState
from haagent.tui.state import MIN_HEIGHT, MIN_WIDTH, PendingInteraction, layout_for_size
from haagent.tui.typography import install_textual_line_breaking
from haagent.tui.widgets import (
    ConversationTimeline,
    ContextUsageLine,
    FooterBar,
    InputDock,
    ProgressStatusLine,
    PromptInput,
    RequestHistoryPreview,
    RequestHistoryRail,
    ResizeMessage,
    StatusBar,
)


class HaAgentScreen(Screen):
    """过滤 Textual 命中缓存中已经卸载的文本选择节点。"""

    def get_widget_and_offset_at(self, x: int, y: int) -> tuple[Widget | None, Offset | None]:
        widget, offset = super().get_widget_and_offset_at(x, y)
        # Markdown 流式刷新会卸载旧段落；Textual 8.2.x 的命中缓存可能短暂返回旧节点，
        # _forward_event 随后会把空 parent 当作选择容器并访问 region，导致整个 TUI 退出。
        if widget is not None and widget is not self and not widget.is_attached:
            return None, None
        return widget, offset


class HaAgentTuiApp(App[None]):
    MIN_WIDTH = MIN_WIDTH
    MIN_HEIGHT = MIN_HEIGHT
    CSS_PATH = "../assets/haagent.tcss"
    BINDINGS = APP_BINDINGS

    def __init__(self, service: AssistantService) -> None:
        install_textual_line_breaking()
        super().__init__()
        self.service = service
        self._state = "idle"
        self._active_turn_index: int | None = None
        self._tool_details_enabled = False
        self._last_failure: FailureView | None = None
        self._pending_interaction: PendingInteraction | None = None
        self._default_prompt_placeholder = "输入消息；Ctrl+Enter 换行，/ 打开命令"
        self._commands = command_registry()
        # controller / flow：各自封装一类职责，App 只做连接与薄分发。
        self._conversation = ConversationController(self)
        self._attachments = AttachmentController(self)
        self._command_handlers = ChatCommandHandlers(self)
        self._command_dispatcher = CommandDispatcher(self)
        self.model_flow = ModelFlow(self)
        self.channel_flow = ChannelFlow(self)
        self.schedule_flow = ScheduleFlow(self)
        self.session_flow = SessionFlow(self)
        self.memory_flow = MemoryFlow(self)
        self.completion_flow = CompletionFlow(self)
        self._theme_choice = select_theme()
        self._file_ref_index: FileReferenceIndex | None = None
        # delta 热路径只调度批量 timeline 刷新，禁止每 token 全量 _refresh。
        self._streaming_refresh_scheduled = False
        self._streaming_refresh_timer: Timer | None = None
        self._timeline_widget: ConversationTimeline | None = None
        self._input_dock_widget: InputDock | None = None
        self._tool_failure_groups: dict[tuple[int, str, str], int] = {}
        # 只保存当前进程最近一次真实 provider usage；不恢复、不写 session package。
        self._context_usage: ContextUsageEvent | None = None

    # ── compose 与生命周期 ───────────────────────────────────────────────
    def get_default_screen(self) -> Screen:
        return HaAgentScreen(id="_default")

    def compose(self) -> ComposeResult:
        yield StatusBar("", id="status-bar")
        yield ResizeMessage(
            "终端尺寸过小\n请调整到至少 80x24 后继续使用 HaAgent TUI。",
            id="resize-message",
            classes="hidden",
        )
        with Horizontal(id="main"):
            yield RequestHistoryRail(id="request-history-rail")
            yield ConversationTimeline(id="conversation", wrap=True, auto_scroll=True)
            yield RequestHistoryPreview("", id="request-history-preview")
        with InputDock(id="input-panel"):
            yield ProgressStatusLine("", id="progress-status")
            yield PromptInput(placeholder=self._default_prompt_placeholder, id="prompt-input", show_line_numbers=False)
        yield ContextUsageLine("", id="context-usage")
        yield FooterBar(footer_text("chat"), id="footer-bar")

    def on_mount(self) -> None:
        self._timeline_widget = self.query_one("#conversation", ConversationTimeline)
        self._timeline_widget.bind_history_rail(self.query_one("#request-history-rail", RequestHistoryRail))
        self._input_dock_widget = self.query_one("#input-panel", InputDock)
        self._apply_theme()
        self._show_initial_configuration_state()
        self.session_flow.restore_initial_session()
        self.schedule_flow.start_background_polling()
        self._refresh()
        self._update_responsive_layout()
        self._prompt_input().focus()
        self._warm_file_reference_index()

    def on_unmount(self) -> None:
        # 停止 badge 轮询并停止 TUI 内嵌 coordinator host，释放租约。
        self.schedule_flow.stop_background_polling()
        # Textual 可能在 default screen 卸载后才触发 timer；必须显式取消，
        # 否则回调会查询已移除的 timeline 并中断退出。
        if self._streaming_refresh_timer is not None:
            self._streaming_refresh_timer.stop()
            self._streaming_refresh_timer = None
        self._streaming_refresh_scheduled = False
        self._timeline_widget = None
        self._input_dock_widget = None

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(width=event.size.width, height=event.size.height)

    def on_key(self, event: events.Key) -> None:
        if self.completion_flow.command_overlay is not None and event.key in {"escape", "up", "down", "enter"}:
            self.action_handle_command_suggestion_key(event)
            return
        if self.completion_flow.file_ref_overlay is not None and event.key in {"escape", "up", "down", "enter"}:
            self.action_handle_file_ref_key(event)
            return
        if self.memory_flow.mode and self._pending_interaction is None:
            if self.memory_flow.handle_key(event.key):
                event.stop()
                return
        if self._pending_interaction is not None:
            return
        if self._prompt_value(self._prompt_input()):
            return
        if event.key == "end":
            event.stop()
            event.prevent_default()
            self.action_conversation_end()
            return
        if event.key in {"/", "slash"} or event.character == "/":
            event.stop()
            self.action_open_command_suggestions()
            return
        if event.key == "enter" and self.memory_flow.mode:
            event.stop()
            self.action_memory_enter()
        elif event.key in {"a", "y"} and self.memory_flow.mode:
            event.stop()
            self.action_confirm_memory()
        elif event.key == "r" and self.memory_flow.mode:
            event.stop()
            self.action_reject_memory()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt-input":
            return
        text = self._prompt_value(event.text_area)
        self._attachments.sync_with_prompt(text)
        self.completion_flow.sync_file_refs(text)
        self.completion_flow.sync_command_suggestions(text)

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self._submit_prompt(event.input)

    def action_submit_prompt(self) -> None:
        self._submit_prompt(self._prompt_input())

    def _submit_prompt(self, prompt_input: PromptInput) -> None:
        prompt = self._prompt_value(prompt_input).strip()
        attachments = self._attachments.attachments_from_prompt(prompt)
        prompt_text = prompt_without_image_tokens(prompt)
        if not prompt_text and not attachments:
            return
        command = parse_slash_command(prompt, self._commands)
        if command is not None:
            self._set_prompt_value(prompt_input, "")
            self._handle_slash_command(command)
            return
        if self._pending_interaction is not None and self._pending_interaction.request.interaction_type == "user_input":
            self._set_prompt_value(prompt_input, "")
            self._complete_interaction(HumanInteractionResponse(approved=True, answer=prompt))
            return
        if self._state in {"running", "cancelling", "waiting approval"}:
            return
        if attachments and not self._current_model_accepts_images():
            return
        self._set_prompt_value(prompt_input, "")
        if is_prompt_mode_command(prompt):
            self._start_prompt(prompt)
            return
        self._attachments.reset()
        self._start_prompt(prompt_text, attachments=attachments, display_prompt=prompt)

    def _start_prompt(
        self,
        prompt: str,
        attachments: list[ImageAttachment] | None = None,
        display_prompt: str | None = None,
    ) -> None:
        self._prompt_input().append_request_history(prompt)
        self._active_turn_index = self._next_turn_index()
        self._tool_failure_groups.clear()
        self._conversation.stick_to_bottom = True
        timeline = self._timeline()
        timeline.set_stick_to_bottom(True)
        self._conversation.append_block("You", display_prompt or prompt)
        self._conversation.start_assistant(self._active_turn_index)
        self._state = "running"
        self._refresh()
        self._run_prompt(prompt, attachments=attachments or [])

    def _next_turn_index(self) -> int:
        # 不吞异常：turn index 直接决定 timeline 写入位置，失败必须可见，
        # 否则错误的 0 会把新一轮内容写到历史 turn 上。
        status = self.service.workspace.status()
        current = status.current_turn_count if status.current_turn_count is not None else 0
        return current + 1

    @work(thread=True, exclusive=True)
    def _run_prompt(self, prompt: str, attachments: list[ImageAttachment] | None = None) -> None:
        try:
            result = self.service.sessions.run_prompt_events(
                prompt,
                event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                interaction_handler=self._handle_interaction,
                attachments=list(attachments or []),
            )
        except Exception as error:
            self.call_from_thread(self._handle_prompt_error, error)
            return
        status = result.status
        self.call_from_thread(self._finish_prompt, status)

    def _handle_chat_event(self, event: RuntimeUiEvent) -> None:
        handle_runtime_ui_event(self, event)

    def _finish_prompt(self, status: str) -> None:
        self._conversation.finalize_streaming_if_needed()
        if status == "completed" and self._state not in {"waiting approval", "waiting input", "cancelled"}:
            self._state = "idle"
        elif status == "cancelled":
            self._state = "cancelled"
            self._conversation.append_block("Cancel", "任务已取消。你可以调整请求后再次提交。")
        elif status != "completed":
            self._state = "failed"
            if self._last_failure is None:
                # 运行返回非 completed 状态但没有 FailureNoticeEvent：只提供已知的
                # status/reason，其余字段交给 failure_from_payload 以「缺少字段」显式呈现，
                # 不用 "unknown" 掩盖 failed_stage/failure_category/episode_path 缺失。
                self._last_failure = failure_from_payload(
                    {"status": status, "reason": status},
                    fallback_message=status,
                )
                self._conversation.append_block("Failure", self._last_failure.block_text())
        self._refresh()

    def _handle_prompt_error(self, error: Exception) -> None:
        self._state = "failed"
        self._conversation.append_block("Failure", str(error))
        self._conversation.finalize_streaming_if_needed()
        self._refresh()

    def set_progress_status(self, status: ProgressStatusState) -> None:
        self.query_one("#progress-status", ProgressStatusLine).update_status(status.text, severity=status.severity)

    def clear_progress_status(self) -> None:
        self.query_one("#progress-status", ProgressStatusLine).clear()

    # ── slash 命令与 overlay 分发 ────────────────────────────────────────
    def _handle_slash_command(self, result) -> None:
        self._command_dispatcher.dispatch(result)

    def action_help(self) -> None:
        self.push_screen(HelpModal(self._help_context()))

    def action_open_sessions(self) -> None:
        self.session_flow.open_sessions()

    def action_open_models(self) -> None:
        self.model_flow.open_models()

    def action_open_connections(self) -> None:
        self.model_flow.open_connections()

    def action_open_search(self) -> None:
        if self._prompt_has_pending_text():
            return
        self.push_screen(SearchOverlay(list(self._conversation.lines)), self._defer_prompt_focus)

    def action_open_permissions(self) -> None:
        if self._prompt_has_pending_text():
            return
        self._show_permissions()

    def action_new_session(self) -> None:
        self.session_flow.new_session()

    def action_resume_latest(self) -> None:
        self.session_flow.resume_latest()

    def action_compact_session(self) -> None:
        self._command_handlers.compact()

    def action_paste_image_from_input(self) -> None:
        self._attachments.paste_from_clipboard()

    # ── 输入补全 overlay（薄分发到 CompletionFlow）────────────────────────
    def action_open_command_suggestions(self) -> None:
        self.completion_flow.open_command_suggestions()

    def action_handle_command_suggestion_key(self, event: events.Key) -> None:
        self.completion_flow.handle_command_key(event)

    def action_accept_command_suggestion(self) -> None:
        self.completion_flow.accept_command_suggestion()

    def command_suggestions_is_open(self) -> bool:
        return self.completion_flow.command_suggestions_is_open()

    def action_open_file_refs(self) -> None:
        self.completion_flow.open_file_refs()

    def action_handle_file_ref_key(self, event: events.Key) -> None:
        self.completion_flow.handle_file_ref_key(event)

    def action_accept_file_ref(self) -> None:
        self.completion_flow.accept_file_ref()

    def file_reference_is_open(self) -> bool:
        return self.completion_flow.file_reference_is_open()

    @work(thread=True, exclusive=True)
    def _warm_file_reference_index(self) -> None:
        status = self.service.workspace.status()
        index = build_file_reference_index(status.workspace_root)
        self.call_from_thread(self._set_file_reference_index, index)

    def _set_file_reference_index(self, index: FileReferenceIndex) -> None:
        self._file_ref_index = index
        input_dock = self._input_dock_widget
        # 后台索引可能在 default screen 已卸载后才回调；App 此时仍可能是 mounted，
        # 只能更新 on_mount 缓存且仍挂载的输入区，不能重新查询已移除的节点。
        if input_dock is not None and input_dock.is_mounted:
            input_dock.file_reference_index = index

    # ── 会话切换后台 worker（磁盘/MCP 不得阻塞 UI 线程）──────────────────
    @work(thread=True, exclusive=True, group="session-ops")
    def _load_session_list_worker(self) -> None:
        try:
            sessions = self.service.sessions.list()
        except Exception as error:
            self.call_from_thread(self.session_flow.handle_session_list_error, error)
            return
        self.call_from_thread(self.session_flow.open_sessions_with_list, sessions)

    @work(thread=True, exclusive=True, group="session-ops")
    def _run_session_create_worker(self) -> None:
        try:
            self.service.sessions.create()
        except Exception as error:
            self.call_from_thread(
                self.session_flow.apply_session_error,
                f"新建会话失败：{error}",
            )
            return
        self.call_from_thread(self.session_flow.apply_create_success)

    @work(thread=True, exclusive=True, group="session-ops")
    def _run_session_continue_worker(self) -> None:
        try:
            status = self.service.sessions.continue_latest()
        except Exception as error:
            self.call_from_thread(
                self.session_flow.apply_session_error,
                f"继续最新会话失败：{error}",
            )
            return
        self.call_from_thread(self.session_flow.apply_continue_success, status)

    @work(thread=True, exclusive=True, group="session-ops")
    def _run_session_overlay_worker(self, result: SessionOverlayResult) -> None:
        try:
            if result.action == "resume" and result.session is not None:
                status = self.service.sessions.resume(result.session.session_path)
            elif result.action == "continue_latest":
                status = self.service.sessions.continue_latest()
            else:
                status = self.service.sessions.create()
        except Exception as error:
            self.call_from_thread(
                self.session_flow.apply_session_error,
                f"会话操作失败：{error}",
            )
            return
        self.call_from_thread(self.session_flow.apply_overlay_success, result, status)

    # ── 模型目录后台 worker（薄 @work 包装，逻辑在 ModelFlow）──────────────
    @work(thread=True, exclusive=True)
    def _refresh_model_catalog_and_open_connection_setup(self) -> None:
        self.model_flow.refresh_catalog_and_open_setup()

    @work(thread=True, exclusive=True)
    def _load_model_switch_catalog(self) -> None:
        self.model_flow.load_switch_catalog()

    @work(thread=True, exclusive=True)
    def _scan_local_model_runtimes(self) -> None:
        # 本地 HTTP 探测必须留在 worker 线程，避免阻塞 Textual UI 事件循环。
        self.model_flow.scan_local_runtimes()

    @work(thread=True, exclusive=True)
    def _refresh_model_catalog_only(self) -> None:
        self.model_flow.refresh_catalog_only()

    @work(thread=True, exclusive=True)
    def _run_model_connection_test(self, connection_id: str, model: str | None = None) -> None:
        self.model_flow.run_connection_test(connection_id, model)

    # ── 计划任务后台 worker（DB 读取不得阻塞 Textual UI 线程）──────────
    @work(thread=True, exclusive=True, group="schedules")
    def _load_schedules_overlay_worker(self, tab: str = "plans") -> None:
        self.schedule_flow.load_schedules_overlay(tab)  # type: ignore[arg-type]

    # ── 渠道后台 worker（薄 @work 包装，逻辑在 ChannelFlow）──────────────
    @work(thread=True, exclusive=True, group="channels")
    def _run_channel_weixin_login(self, instance_id: str | None) -> None:
        # QR 轮询与 HTTP 必须在 worker，避免阻塞 Textual UI 线程。
        self.channel_flow.run_weixin_login(instance_id)

    @work(thread=True, exclusive=True, group="channels")
    def _run_channel_connection_test(self, instance_id: str) -> None:
        self.channel_flow.run_connection_test(instance_id)

    # ── 记忆动作（薄分发到 MemoryFlow）───────────────────────────────────
    def action_toggle_memory(self) -> None:
        self.memory_flow.toggle()

    def action_memory_enter(self) -> None:
        self.memory_flow.enter()

    def action_memory_up(self) -> None:
        self.memory_flow.move(-1)

    def action_memory_down(self) -> None:
        self.memory_flow.move(1)

    def action_memory_first(self) -> None:
        self.memory_flow.first()

    def action_memory_last(self) -> None:
        self.memory_flow.last()

    def action_confirm_memory(self) -> None:
        self.memory_flow.confirm()

    def action_reject_memory(self) -> None:
        self.memory_flow.reject()


    # ── 交互（审批 / 补充输入）───────────────────────────────────────────
    def _handle_interaction(self, request: HumanInteractionRequest) -> HumanInteractionResponse:
        pending = PendingInteraction(request)
        self.call_from_thread(self._begin_interaction, pending)
        pending.done.wait()
        return pending.response or HumanInteractionResponse(approved=False, answer="")

    def _begin_interaction(self, pending: PendingInteraction) -> None:
        self._pending_interaction = pending
        request = pending.request
        if request.interaction_type == "approval":
            self._state = "waiting approval"
            self.push_screen(ToolApprovalModal(request), self._complete_approval)
        elif request.interaction_type == "edit_diff":
            self._state = "waiting approval"
            self.push_screen(EditDiffModal(request), self._complete_edit_diff)
        else:
            self._state = "waiting input"
            self._set_answer_required(request.question)
        self._refresh()

    def _complete_approval(self, decision: str | None) -> None:
        normalized = decision or "deny"
        self._complete_interaction(
            HumanInteractionResponse(
                approved=normalized in {"once", "always"},
                answer=normalized,
            ),
        )

    def _complete_edit_diff(self, decision: str | None) -> None:
        normalized = decision or "deny"
        self._complete_interaction(HumanInteractionResponse(approved=normalized in {"once", "always"}, answer=normalized))

    def _complete_interaction(self, response: HumanInteractionResponse) -> None:
        pending = self._pending_interaction
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._pending_interaction = None
        self._restore_prompt_input()
        self._state = "running"
        self._refresh()

    def _set_answer_required(self, question: str) -> None:
        prompt_input = self._prompt_input()
        prompt_input.placeholder = f"回答 Agent 的问题：{safe_summary(question, 90)}"
        prompt_input.focus()

    def _restore_prompt_input(self) -> None:
        self._prompt_input().placeholder = self._default_prompt_placeholder

    def action_cancel_interaction(self) -> None:
        if self.memory_flow.cancel():
            return
        if self._pending_interaction is None:
            return
        self._complete_interaction(HumanInteractionResponse(approved=False, answer=""))

    def action_cancel_current_task(self) -> None:
        result = self._request_current_task_cancel()
        if result is None:
            return
        status = result.status
        if status == "unavailable":
            self._conversation.append_block("Cancel", "当前 service 未提供可取消协议；本轮不能安全取消。")
        elif status == "idle":
            self._state = "idle"
            self._conversation.append_block("Cancel", "当前没有仍在运行的任务。")
        else:
            self._state = "cancelling"
            self._conversation.append_block("Cancel", "任务正在取消，请等待当前运行态结束。")
        self._refresh()

    def _request_current_task_cancel(self):
        if self._state not in {"running", "waiting approval", "waiting input", "cancelling"}:
            return None
        result = self.service.sessions.cancel_current_run()
        if self._pending_interaction is not None:
            self._pending_interaction.response = HumanInteractionResponse(approved=False, answer="")
            self._pending_interaction.done.set()
            self._pending_interaction = None
        self._restore_prompt_input()
        return result

    def action_quit(self) -> None:
        self._request_current_task_cancel()
        self.exit(None)

    # ── 主题与导航 ───────────────────────────────────────────────────────
    def action_toggle_theme(self) -> None:
        self._theme_choice = next_theme(self._theme_choice)
        self._apply_theme()
        self._refresh()

    def action_conversation_page_up(self) -> None:
        self._conversation.page_up()

    def action_conversation_page_down(self) -> None:
        self._conversation.page_down()

    def action_conversation_end(self) -> None:
        self._conversation.stick_and_scroll_to_end()

    def action_previous_request(self) -> None:
        self._timeline().navigate_adjacent_request(-1)

    def action_next_request(self) -> None:
        self._timeline().navigate_adjacent_request(1)

    def on_request_history_rail_navigate(self, message: RequestHistoryRail.Navigate) -> None:
        self._timeline().scroll_to_request(message.turn_index)
        if not message.keep_focus:
            self.focus_prompt_input()

    def focus_prompt_input(self) -> None:
        self._prompt_input().focus()

    # ── permissions / skills 委托 ───────────────────────────────────────
    def _show_permissions(self) -> None:
        permissions.show_permissions(self)

    def _handle_permissions_result(self, result: dict[str, object] | None) -> None:
        permissions.handle_permissions_result(self, result)

    def _set_permission_mode(self, mode: str) -> None:
        permissions.set_permission_mode(self, mode)

    def _handle_permission_mode_confirmed(self, mode: str, confirmed: bool) -> None:
        permissions.handle_permission_mode_confirmed(self, mode, confirmed)

    def _handle_clear_external_roots_confirmed(self, confirmed: bool) -> None:
        permissions.handle_clear_external_roots_confirmed(self, confirmed)

    def _handle_set_full_access_confirmed(self, path: Path, confirmed: bool) -> None:
        permissions.handle_set_full_access_confirmed(self, path, confirmed)

    def _handle_skills_command(self, argument: str) -> None:
        skills.handle_skills_command(self, argument)

    def _handle_skill_marketplace_install_confirmed(self, result_id: str, confirmed: bool | None) -> None:
        skills.handle_skill_marketplace_install_confirmed(self, result_id, confirmed)

    def _handle_skill_command(self, argument: str) -> None:
        skills.handle_skill_command(self, argument)

    def _open_skill_picker(self, *, mode: str) -> None:
        skills.open_skill_picker(self, mode=mode)

    def _handle_skill_picker_result(self, skill: dict[str, object] | None) -> None:
        skills.handle_skill_picker_result(self, skill)

    # ── 初始状态、刷新与布局 ─────────────────────────────────────────────
    def _show_initial_configuration_state(self) -> None:
        status = self.service.workspace.status()
        if status.profile_error is not None:
            self._conversation.append_block("Config", "未找到默认模型配置\n输入 /connect 配置供应商连接。")
        elif status.credential_store_available is False:
            reason = status.credential_store_error or "unknown"
            self._conversation.append_block("Config", f"系统凭据库不可用：{reason}\n输入 /connect 重新配置供应商连接。")
        elif status.api_key_env and not status.api_key_available:
            self._conversation.append_block(
                "Config",
                f"API key 缺失：{status.api_key_env}\n输入 /connect 可以配置或测试供应商连接；HaAgent 不会显示真实 API key。",
            )

    def _refresh(self) -> None:
        status = self.service.workspace.status()
        status_bar = self.query_one("#status-bar", StatusBar)
        base = status_line(
            status,
            ui_state=self._state,
            # on_mount 首次刷新时 content_region 尚未完成布局；屏幕宽度减样式 gutter
            # 在首次渲染和后续 resize 中都稳定，避免宽屏裁右端、窄屏错误回退到 120 列。
            width=max(1, self.size.width - status_bar.styles.gutter.width),
        )
        status_bar.update_status(base)
        self._refresh_conversation()
        self.query_one("#footer-bar", FooterBar).update_footer(footer_text(self._help_context()))
        self._apply_focus_classes()

    def _schedule_streaming_refresh(self) -> None:
        """AssistantDelta 热路径：16～50ms 批量刷新 timeline，绝不查询 status/keyring。"""

        if self._streaming_refresh_scheduled:
            return
        self._streaming_refresh_scheduled = True
        self._streaming_refresh_timer = self.set_timer(
            0.033,
            self._flush_streaming_refresh,
            name="streaming-refresh",
        )

    def _flush_streaming_refresh(self) -> None:
        self._streaming_refresh_timer = None
        self._streaming_refresh_scheduled = False
        timeline = self._timeline_widget
        if timeline is None or not timeline.is_attached:
            return
        self._refresh_conversation()

    def _refresh_conversation(self) -> None:
        conversation = self._timeline()
        if self.memory_flow.mode:
            conversation.show_memory(self.memory_flow.panel_text())
            self._conversation.clear_placeholder_state()
            return
        self._conversation.refresh()

    def _help_context(self) -> str:
        if self.size.width < self.MIN_WIDTH or self.size.height < self.MIN_HEIGHT:
            return "too_small"
        if self._pending_interaction is not None:
            return "pending_input" if self._pending_interaction.request.interaction_type == "user_input" else "approval"
        if self.memory_flow.mode:
            return "memory_detail" if self.memory_flow.detail_mode else "memory_list"
        if self._state == "running":
            return "running"
        return "chat"

    def _apply_theme(self) -> None:
        for theme in textual_themes():
            if theme.name not in self.available_themes:
                self.register_theme(theme)
        self.theme = self._theme_choice.textual_theme
        for choice in ("theme-dark", "theme-light", "theme-monochrome"):
            self.screen.set_class(choice == self._theme_choice.css_class, choice)

    def _apply_focus_classes(self) -> None:
        prompt = self._prompt_input()
        conversation = self.query_one("#conversation", ConversationTimeline)
        prompt.set_class(prompt.has_focus, "panel-focused")
        conversation.set_class(not prompt.has_focus or self.memory_flow.mode, "panel-focused")

    def _update_responsive_layout(self, width: int | None = None, height: int | None = None) -> None:
        terminal_width = width if width is not None else self.size.width
        terminal_height = height if height is not None else self.size.height
        layout = layout_for_size(terminal_width, terminal_height)
        self.query_one("#resize-message", ResizeMessage).set_class(not layout.too_small, "hidden")
        self.query_one("#main", Horizontal).set_class(layout.too_small, "hidden")
        self.query_one("#input-panel", InputDock).set_class(layout.too_small, "hidden")
        self.query_one("#footer-bar", FooterBar).set_class(layout.too_small, "hidden")
        self._render_context_usage(terminal_width, visible=not layout.too_small)

    def update_context_usage(self, event: ContextUsageEvent) -> None:
        """模型 step 完成后只刷新用量行，避免重读 workspace/keyring。"""

        self._context_usage = event
        layout = layout_for_size(self.size.width, self.size.height)
        self._render_context_usage(self.size.width, visible=not layout.too_small)

    def clear_context_usage(self) -> None:
        self._context_usage = None
        self.query_one("#context-usage", ContextUsageLine).clear()

    def _render_context_usage(self, terminal_width: int, *, visible: bool) -> None:
        widget = self.query_one("#context-usage", ContextUsageLine)
        usage = self._context_usage
        if not visible or usage is None:
            widget.clear()
            return
        widget.update_usage(
            context_usage_line(
                usage.input_tokens,
                usage.input_window_tokens,
                terminal_width=terminal_width,
            ),
        )

    # ── 图片输入 ─────────────────────────────────────────────────────────
    def _current_model_accepts_images(self) -> bool:
        try:
            status = self.service.workspace.status()
        except Exception as error:
            self._conversation.append_block("Command", f"无法确认当前模型是否支持图片输入：{error}")
            self._refresh()
            return False
        if status.image_input_supported is not False:
            return True
        model_label = status.model or status.profile_name or "当前模型"
        self._conversation.append_block(
            "Command",
            f"当前模型不支持图片输入：{model_label}。请切换到支持视觉的模型后再发送。",
        )
        self._refresh()
        return False

    def _reset_image_input_state(self) -> None:
        self._attachments.reset()
        self._set_prompt_value(self._prompt_input(), "")


    # ── 焦点与 prompt 辅助 ───────────────────────────────────────────────
    def _prompt_input(self) -> PromptInput:
        return self.query_one("#prompt-input", PromptInput)

    def _input_dock(self) -> InputDock:
        return self.query_one("#input-panel", InputDock)

    def _timeline(self) -> ConversationTimeline:
        timeline = self._timeline_widget
        if timeline is not None and timeline.is_attached:
            return timeline
        return self.query_one("#conversation", ConversationTimeline)

    def _prompt_has_pending_text(self) -> bool:
        prompt_input = self._prompt_input()
        return bool(prompt_input.has_focus and self._prompt_value(prompt_input))

    def _prompt_value(self, prompt_input: PromptInput) -> str:
        return prompt_input.text

    def _set_prompt_value(self, prompt_input: PromptInput, value: str) -> None:
        prompt_input.value = value

    def _restore_prompt_focus(self, _result: object | None = None) -> None:
        # 延迟回调可能在 Textual 卸载 default screen 后才执行；此时不应让焦点恢复中断退出。
        try:
            self._prompt_input().focus()
        except NoMatches:
            return

    def _defer_prompt_focus(self, _result: object | None = None) -> None:
        self.set_timer(0.01, self._restore_prompt_focus)


def run_tui(service: AssistantService) -> int:
    HaAgentTuiApp(service).run()
    return 0


