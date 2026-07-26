"""
haagent/tui/application/app.py - HaAgent TUI 应用编排

组合 Textual 组件、连接 AssistantService 与各 controller/flow，并提供顶层生命周期
和 Textual binding 入口。具体业务流程（模型连接、会话、记忆、命令、附件、时间线）
均迁出到同级 controller/flow 模块，本文件只做薄分发。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

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
from haagent.tui.application.diagnostics import TuiDiagnostics
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
from haagent.tui.files.refs import FileReferenceIndex, build_file_reference_index
from haagent.tui.flows import permissions, skills
from haagent.tui.overlays.modals import EditDiffModal, HelpModal, ToolApprovalModal
from haagent.tui.overlays.search import SearchOverlay
from haagent.tui.overlays.sessions import SessionOverlayResult
from haagent.tui.presentation.progress import ProgressStatusState
from haagent.tui.state import MIN_HEIGHT, MIN_WIDTH, PendingInteraction, layout_for_size
from haagent.tui.typography import install_textual_line_breaking
from haagent.tui.widgets.plan_confirmation import format_plan_for_timeline
from haagent.tui.widgets import (
    ConversationTimeline,
    ContextUsageLine,
    FooterBar,
    InputDock,
    ProgressStatusLine,
    PromptInput,
    QuestionPrompt,
    PlanConfirmationPanel,
    RequestHistoryPreview,
    RequestHistoryRail,
    ResizeMessage,
    StatusBar,
    TodoPanel,
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
        # 运行中用户 Enter 提交的消息队列；本轮结束后自动作为下一条请求发送。
        self._steering_queue: list[str] = []
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
        self._skill_marketplace_search_generation = 0
        self._skill_marketplace_search_lock = Lock()
        # delta 热路径只调度批量 timeline 刷新，禁止每 token 全量 _refresh。
        self._streaming_refresh_scheduled = False
        self._streaming_refresh_timer: Timer | None = None
        self._timeline_widget: ConversationTimeline | None = None
        self._input_dock_widget: InputDock | None = None
        self._tool_failure_groups: dict[tuple[int, str, str], int] = {}
        # 只保存当前进程最近一次真实 provider usage；不恢复、不写 session package。
        self._context_usage: ContextUsageEvent | None = None
        self._diagnostics = TuiDiagnostics()

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
            yield TodoPanel(id="todo-panel")
            yield ProgressStatusLine("", id="progress-status")
            yield PromptInput(placeholder=self._default_prompt_placeholder, id="prompt-input", show_line_numbers=False)
        yield ContextUsageLine("", id="context-usage")
        yield FooterBar(footer_text("chat"), id="footer-bar")

    def on_mount(self) -> None:
        self._diagnostics.record_started()
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

    async def on_unmount(self) -> None:
        # 停止 badge 轮询并停止 TUI 内嵌 coordinator host，释放租约。
        self.schedule_flow.stop_background_polling()
        # Textual 可能在 default screen 卸载后才触发 timer；必须显式取消，
        # 否则回调会查询已移除的 timeline 并中断退出。
        if self._streaming_refresh_timer is not None:
            self._streaming_refresh_timer.stop()
            self._streaming_refresh_timer = None
        self._streaming_refresh_scheduled = False
        self._record_tui_shutdown()
        if self._timeline_widget is not None:
            await self._timeline_widget.close_markdown_streams()
        self._timeline_widget = None
        self._input_dock_widget = None

    def _record_tui_shutdown(self) -> None:
        timeline = self._timeline_widget
        if timeline is None:
            active_writers, pending_fragments = 0, 0
        else:
            active_writers, pending_fragments = timeline.markdown_stream_diagnostics()
        self._diagnostics.record_stopped(
            active_markdown_writers=active_writers,
            pending_markdown_fragments=pending_fragments,
        )

    def record_unhandled_exception(self, error: BaseException) -> None:
        """由 run_tui 的顶层边界调用；记录后仍让原异常按原语义传播。"""

        self._diagnostics.record_unhandled_exception(error)

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

    def on_question_prompt_submitted(self, event: QuestionPrompt.Submitted) -> None:
        self._complete_interaction(
            HumanInteractionResponse(
                approved=True,
                outcome="answered",
                answers=event.answers,
            )
        )

    def on_question_prompt_dismissed(self, _event: QuestionPrompt.Dismissed) -> None:
        self._complete_interaction(HumanInteractionResponse(approved=False, outcome="dismissed"))

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
        if self._state in {"waiting approval", "waiting plan"}:
            return
        if self._state in {"running", "cancelling"}:
            # 运行中 Enter=排队到本轮结束；Ctrl+G 走 action_steer_current_task 立即引导。
            if attachments:
                return
            self._set_prompt_value(prompt_input, "")
            self._queue_prompt_during_run(prompt_text)
            return
        if attachments and not self._current_model_accepts_images():
            return
        self._set_prompt_value(prompt_input, "")
        if is_prompt_mode_command(prompt):
            self._start_prompt(prompt)
            return
        self._attachments.reset()
        self._start_prompt(prompt_text, attachments=attachments, display_prompt=prompt)

    def _queue_prompt_during_run(self, prompt_text: str) -> None:
        self._steering_queue.append(prompt_text)
        self._conversation.append_block("You (已排队)", prompt_text)
        self.set_progress_status(
            ProgressStatusState(
                text=f"{len(self._steering_queue)} 条消息已排队，将在本轮结束后发送；Ctrl+G 可立即引导",
                severity="info",
                turn_index=self._active_turn_index or 0,
                source="task",
            ),
        )
        self._refresh()

    def action_steer_current_task(self) -> None:
        if self._state not in {"running", "cancelling"}:
            return
        prompt_input = self._prompt_input()
        text = self._prompt_value(prompt_input).strip()
        if not text:
            return
        self._set_prompt_value(prompt_input, "")
        if self.service.sessions.steer_current_run(text):
            self._conversation.append_block("You (引导)", text)
            self._refresh()
            return
        # 任务恰好已结束：回落为排队，_finish_prompt 会立即投递。
        self._queue_prompt_during_run(text)

    def _start_prompt(
        self,
        prompt: str,
        attachments: list[ImageAttachment] | None = None,
        display_prompt: str | None = None,
        show_user_block: bool = True,
    ) -> None:
        self._prompt_input().append_request_history(prompt)
        self._active_turn_index = self._next_turn_index()
        self._tool_failure_groups.clear()
        self._conversation.stick_to_bottom = True
        timeline = self._timeline()
        timeline.set_stick_to_bottom(True)
        if show_user_block:
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

    @work(thread=True, exclusive=True, group="prompt")
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
        # service 层可能返回不带 unconsumed_steering 的结果对象（如测试 Fake），缺省为空。
        unconsumed = tuple(getattr(result, "unconsumed_steering", ()) or ())
        self.call_from_thread(self._finish_prompt, result.status, unconsumed)

    def _handle_chat_event(self, event: RuntimeUiEvent) -> None:
        handle_runtime_ui_event(self, event)

    def _finish_prompt(self, status: str, unconsumed_steering: tuple[str, ...] = ()) -> None:
        if unconsumed_steering:
            # run 结束前未来得及注入的引导不丢弃，放到排队消息最前面重新投递。
            self._steering_queue[:0] = list(unconsumed_steering)
        self._conversation.finalize_streaming_if_needed()
        if status == "completed" and self._state not in {"waiting approval", "waiting input", "waiting plan", "cancelled"}:
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
        self._dispatch_queued_prompts()

    def _dispatch_queued_prompts(self) -> None:
        if not self._steering_queue:
            return
        if self._state not in {"idle", "cancelled", "failed"}:
            return
        prompt = "\n\n".join(self._steering_queue)
        self._steering_queue.clear()
        self.clear_progress_status()
        # 排队消息在入队时已渲染过 You 块，这里不再重复展示。
        self._start_prompt(prompt, show_user_block=False)

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

    def _enter_plan_mode(self, request: str) -> None:
        try:
            self.service.sessions.enter_plan_mode()
        except Exception as error:
            self._conversation.append_block("Plan", str(error))
            self._refresh()
            return
        self._state = "planning"
        if request.strip():
            self._start_prompt(request.strip())
            return
        self._conversation.append_block("Plan", "已进入 Plan Mode。请输入需要规划的任务。")
        self._refresh()

    def _restore_persisted_plan_state(self) -> None:
        try:
            state = self.service.sessions.get_planning_state()
        except Exception:
            return
        self._refresh_todo_panel()
        if state.status == "planning":
            self._state = "planning"
            return
        if state.status == "awaiting_confirmation" and state.plan_id and state.proposal is not None:
            self._state = "waiting plan"
            self._resume_plan_confirmation_worker(
                HumanInteractionRequest(
                    interaction_type="plan_confirmation",
                    tool_name="submit_plan",
                    question="确认实施方案",
                    plan_id=state.plan_id,
                    plan_revision=state.revision,
                    plan_proposal=state.proposal,
                ),
            )
            return
        if state.status == "approved_pending_execution":
            self._state = "running"
            self._resume_approved_plan_worker()

    @work(thread=True, exclusive=True, group="prompt")
    def _resume_plan_confirmation_worker(self, request: HumanInteractionRequest) -> None:
        response = self._handle_interaction(request)
        try:
            if response.plan_outcome == "approved":
                self.service.sessions.approve_plan(request.plan_id, request.plan_revision)
                result = self.service.sessions.execute_pending_plan_events(
                    event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                    interaction_handler=self._handle_interaction,
                )
            elif response.plan_outcome == "revision_requested":
                self.service.sessions.submit_plan_feedback(
                    request.plan_id,
                    request.plan_revision,
                    response.answer,
                )
                result = self.service.sessions.run_prompt_events(
                    f"请根据以下用户反馈修订当前 Plan：{response.answer}",
                    event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                    interaction_handler=self._handle_interaction,
                )
            else:
                self.service.sessions.cancel_plan()
                self.call_from_thread(self._finish_prompt, "cancelled")
                return
        except Exception as error:
            self.call_from_thread(self._handle_prompt_error, error)
            return
        self.call_from_thread(self._finish_prompt, result.status)

    @work(thread=True, exclusive=True, group="prompt")
    def _resume_approved_plan_worker(self) -> None:
        try:
            result = self.service.sessions.execute_pending_plan_events(
                event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                interaction_handler=self._handle_interaction,
            )
        except Exception as error:
            self.call_from_thread(self._handle_prompt_error, error)
            return
        self.call_from_thread(self._finish_prompt, result.status)

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

    @work(thread=True, exclusive=True, group="file-reference-index")
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

    def _start_skill_marketplace_search(self, query: str) -> None:
        self._skill_marketplace_search_generation += 1
        self._search_skill_marketplace_worker(query, self._skill_marketplace_search_generation)

    @work(thread=True, exclusive=True, group="skills-marketplace")
    def _search_skill_marketplace_worker(self, query: str, generation: int) -> None:
        try:
            # service 同时更新可安装结果缓存；串行提交可保证最新查询最后写入。
            with self._skill_marketplace_search_lock:
                result = self.service.skills.search_marketplace(query, limit=10)
        except Exception as error:
            self.call_from_thread(skills.apply_skill_marketplace_search_error, self, generation, error)
            return
        self.call_from_thread(skills.apply_skill_marketplace_search_success, self, generation, result)

    # ── 会话切换后台 worker（磁盘/MCP 不得阻塞 UI 线程）──────────────────
    @work(thread=True, exclusive=True, group="session-ops")
    def _run_initial_session_restore_worker(self, initial_resume: str | Path | None, initial_continue: bool) -> None:
        try:
            if initial_resume is not None:
                status = self.service.sessions.resume(initial_resume)
                prefix = "已恢复 session"
            elif initial_continue:
                status = self.service.sessions.continue_latest()
                prefix = "已继续最新 session"
            else:
                return
        except Exception as error:
            self.call_from_thread(self.session_flow.apply_session_error, f"恢复会话失败：{error}")
            return
        history, history_error = self._session_history_for_worker()
        self.call_from_thread(
            self.session_flow.apply_initial_restore_success,
            status,
            prefix,
            history,
            history_error,
        )

    def _session_history_for_worker(self):
        try:
            return list(self.service.sessions.history()), None
        except Exception as error:
            return None, error

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
        history, history_error = self._session_history_for_worker()
        self.call_from_thread(self.session_flow.apply_continue_success, status, history, history_error)

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
        history = None
        history_error = None
        if result.action != "new":
            history, history_error = self._session_history_for_worker()
        self.call_from_thread(self.session_flow.apply_overlay_success, result, status, history, history_error)

    # ── 模型目录后台 worker（薄 @work 包装，逻辑在 ModelFlow）──────────────
    @work(thread=True, exclusive=True, group="model-ops")
    def _refresh_model_catalog_and_open_connection_setup(self) -> None:
        self.model_flow.refresh_catalog_and_open_setup()

    @work(thread=True, exclusive=True, group="model-ops")
    def _load_model_switch_catalog(self) -> None:
        self.model_flow.load_switch_catalog()

    @work(thread=True, exclusive=True, group="model-ops")
    def _scan_local_model_runtimes(self) -> None:
        # 本地 HTTP 探测必须留在 worker 线程，避免阻塞 Textual UI 事件循环。
        self.model_flow.scan_local_runtimes()

    @work(thread=True, exclusive=True, group="model-ops")
    def _refresh_model_catalog_only(self) -> None:
        self.model_flow.refresh_catalog_only()

    @work(thread=True, exclusive=True, group="model-connection-test")
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
    @work(thread=True, exclusive=True, group="memory-ops")
    def _run_memory_action_worker(self, action: str, candidate_id: str) -> None:
        action_error = None
        try:
            if action == "confirm":
                self.service.memory.confirm_candidate(candidate_id)
            elif action == "reject":
                self.service.memory.reject_candidate(candidate_id, "rejected from TUI")
            else:
                raise ValueError(f"unsupported memory action: {action}")
        except Exception as error:
            action_error = error
        try:
            candidates = self.service.memory.list_candidates(status="pending")
            load_error = None
        except Exception as error:
            candidates = []
            load_error = error
        self.call_from_thread(
            self.memory_flow.apply_action_result,
            action,
            candidate_id,
            action_error,
            candidates,
            load_error,
        )

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
        if pending.response is not None:
            return pending.response
        if request.interaction_type == "plan_confirmation":
            return HumanInteractionResponse(approved=False, plan_outcome="cancelled")
        if request.interaction_type == "user_input":
            return HumanInteractionResponse(approved=False, outcome="dismissed")
        return HumanInteractionResponse(approved=False, answer="")

    def _begin_interaction(self, pending: PendingInteraction) -> None:
        self._pending_interaction = pending
        request = pending.request
        if request.interaction_type == "approval":
            self._state = "waiting approval"
            self.push_screen(ToolApprovalModal(request), self._complete_approval)
        elif request.interaction_type == "edit_diff":
            self._state = "waiting approval"
            self.push_screen(EditDiffModal(request), self._complete_edit_diff)
        elif request.interaction_type == "plan_confirmation":
            self._state = "waiting plan"
            if request.plan_proposal is None or request.plan_revision is None:
                pending.response = HumanInteractionResponse(approved=False, plan_outcome="cancelled")
                pending.done.set()
                self._pending_interaction = None
                raise ValueError("plan_confirmation interaction requires proposal and revision")
            self._conversation.append_block(
                "Plan",
                format_plan_for_timeline(request.plan_revision, request.plan_proposal),
            )
            self._input_dock().open_plan_confirmation(request)
        else:
            if not request.questions:
                pending.response = HumanInteractionResponse(approved=False, outcome="dismissed")
                pending.done.set()
                self._pending_interaction = None
                raise ValueError("user_input interaction requires structured questions")
            self._state = "waiting input"
            self._input_dock().open_question_prompt(request.questions)
        self._refresh_interaction_chrome()

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
        interaction_type = pending.request.interaction_type
        self._pending_interaction = None
        if interaction_type == "user_input":
            self._input_dock().close_question_prompt()
        elif interaction_type == "plan_confirmation":
            if response.plan_outcome == "revision_requested":
                panel = self._input_dock().plan_confirmation
                if panel is not None:
                    panel.set_processing("正在根据修改意见生成新版 Plan…")
            else:
                self._input_dock().close_plan_confirmation()
        else:
            self._restore_prompt_input()
        self._state = "running"
        self._refresh_interaction_chrome()

    def _restore_prompt_input(self) -> None:
        self._prompt_input().placeholder = self._default_prompt_placeholder

    def _refresh_todo_panel(self) -> None:
        try:
            state = self.service.sessions.get_todo_state()
        except Exception:
            return
        panel = self._input_dock().todo_panel()
        if panel is None:
            return
        panel.update_state(state.items)
        self._input_dock()._collapse_if_idle()

    def action_cancel_interaction(self) -> None:
        if self.memory_flow.cancel():
            return
        if self._pending_interaction is None:
            return
        if self._pending_interaction.request.interaction_type == "user_input":
            self._complete_interaction(HumanInteractionResponse(approved=False, outcome="dismissed"))
            return
        if self._pending_interaction.request.interaction_type == "plan_confirmation":
            panel = self._input_dock().plan_confirmation
            if panel is not None:
                panel.minimize()
            return
        self._complete_interaction(HumanInteractionResponse(approved=False, answer="deny"))

    def on_plan_confirmation_panel_feedback_submitted(
        self,
        event: PlanConfirmationPanel.FeedbackSubmitted,
    ) -> None:
        panel = self._input_dock().plan_confirmation
        if panel is None or panel.processing:
            return
        panel.set_processing("正在提交修改意见…")
        self._complete_interaction(
            HumanInteractionResponse(
                approved=False,
                answer=event.feedback,
                plan_outcome="revision_requested",
            ),
        )

    def on_plan_confirmation_panel_approved(self, event: PlanConfirmationPanel.Approved) -> None:
        del event
        panel = self._input_dock().plan_confirmation
        if panel is None or panel.processing:
            return
        panel.set_processing("已批准，正在准备执行…")
        self._complete_interaction(
            HumanInteractionResponse(approved=True, answer="approve", plan_outcome="approved"),
        )

    def on_plan_confirmation_panel_minimized(self, event: PlanConfirmationPanel.Minimized) -> None:
        del event
        self._refresh_interaction_chrome()

    def action_cancel_current_task(self) -> None:
        was_empty_plan_mode = self._state == "planning"
        result = self._request_current_task_cancel()
        if result is None:
            return
        status = result.status
        if status == "unavailable":
            self._conversation.append_block("Cancel", "当前 service 未提供可取消协议；本轮不能安全取消。")
        elif status == "idle":
            self._state = "idle"
            self._conversation.append_block("Cancel", "当前没有仍在运行的任务。")
        elif status == "cancelled" and was_empty_plan_mode:
            # 空 Plan Mode 没有运行中的 worker 会再调用 _finish_prompt；取消已同步完成，
            # 必须在当前 UI 事件内收束状态，避免永久停在“正在取消”。
            self._finish_prompt("cancelled")
            return
        else:
            self._state = "cancelling"
            self._conversation.append_block("Cancel", "任务正在取消，请等待当前运行态结束。")
        self._refresh()

    def _request_current_task_cancel(self):
        if self._state not in {"running", "planning", "waiting plan", "waiting approval", "waiting input", "cancelling"}:
            return None
        result = self.service.sessions.cancel_current_run()
        if self._pending_interaction is not None:
            interaction_type = self._pending_interaction.request.interaction_type
            self._pending_interaction.response = (
                HumanInteractionResponse(approved=False, outcome="dismissed")
                if interaction_type == "user_input"
                else HumanInteractionResponse(approved=False, plan_outcome="cancelled")
                if interaction_type == "plan_confirmation"
                else HumanInteractionResponse(approved=False, answer="deny")
            )
            self._pending_interaction.done.set()
            self._pending_interaction = None
            if interaction_type == "user_input":
                self._input_dock().close_question_prompt()
            elif interaction_type == "plan_confirmation":
                self._input_dock().close_plan_confirmation()
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
        self._refresh_interaction_chrome()
        self._refresh_conversation()

    def _refresh_interaction_chrome(self) -> None:
        """交互开关只更新状态、footer 和焦点，禁止重建 timeline。"""
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
        question_prompt = self._input_dock().question_prompt
        input_has_focus = prompt.has_focus or bool(question_prompt and question_prompt.has_focus)
        conversation.set_class(not input_has_focus or self.memory_flow.mode, "panel-focused")

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
        # 旧 modal 的延迟回调不得越过新交互的焦点边界，把焦点抢回已隐藏的普通输入框。
        if self._pending_interaction is not None:
            return
        # 延迟回调可能在 Textual 卸载 default screen 后才执行；此时不应让焦点恢复中断退出。
        try:
            self._prompt_input().focus()
        except NoMatches:
            return

    def _defer_prompt_focus(self, _result: object | None = None) -> None:
        self.set_timer(0.01, self._restore_prompt_focus)


def run_tui(service: AssistantService) -> int:
    app = HaAgentTuiApp(service)
    try:
        app.run()
    except BaseException as error:
        app.record_unhandled_exception(error)
        raise
    return 0


