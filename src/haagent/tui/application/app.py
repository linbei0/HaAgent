"""
haagent/tui/app.py - HaAgent TUI 应用编排

组合 Textual 组件、协调会话状态和后台 worker，具体渲染与组件细节拆分到同级模块。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import TextArea

from haagent.app.assistant_service import AssistantService, AssistantServiceError
from haagent.models.gateway_registry import catalog_provider_capability
from haagent.memory import MemoryCandidate
from haagent.runtime.events import RuntimeUiEvent
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.commands.suggestions import CommandSuggestionOverlay
from haagent.tui.commands import command_registry, is_prompt_mode_command, parse_slash_command
from haagent.tui.design.copy import BLOCK_TITLES
from haagent.tui.design.failures import FailureView
from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import FileReferenceIndex, build_file_reference_index, query_after_at, replace_at_query
from haagent.tui.design.keys import APP_BINDINGS, footer_text
from haagent.tui.memory.presenter import MemoryPanelPresenter
from haagent.tui.overlays.modals import ConfirmModal, EditDiffModal, HelpModal, PermissionsModal, ToolApprovalModal
from haagent.tui.flows import path_authorization
from haagent.tui.flows import permissions
from haagent.tui.flows import skills
from haagent.tui.overlays.models import (
    ManualModelSetupWizard,
    ModelCatalogLoadingOverlay,
    ModelCenterOverlay,
    ModelCenterResult,
    ModelSetupWizard,
)
from haagent.tui.design.renderers import status_line
from haagent.tui.application.runtime_events import handle_runtime_ui_event
from haagent.tui.state import MIN_HEIGHT, MIN_WIDTH, PendingInteraction, layout_for_size
from haagent.tui.design.theme import (
    next_theme,
    no_color_enabled,
    select_theme,
    semantic_tokens,
    status_semantic,
    textual_themes,
    theme_label,
)
from haagent.tui.overlays.search import SearchOverlay
from haagent.tui.overlays.sessions import SessionOverlay, SessionOverlayResult
from haagent.tui.design.utils import safe_summary
from haagent.tui.widgets import (
    ConversationTimeline,
    ConversationView,
    FooterBar,
    PromptInput,
    ResizeMessage,
    StatusBar,
    ToolActivity,
    ToolStatus,
    _end_location,
)


def find_untrusted_absolute_paths(
    text: str,
    *,
    project_root: Path,
    external_roots: list[dict[str, str]] | None = None,
) -> list[Path]:
    return path_authorization.find_untrusted_absolute_paths(
        text,
        project_root=project_root,
        external_roots=external_roots,
    )


def is_wide_external_root(path: Path) -> bool:
    return path_authorization.is_wide_external_root(path)


class HaAgentTuiApp(App[None]):
    MIN_WIDTH = MIN_WIDTH
    MIN_HEIGHT = MIN_HEIGHT
    CSS_PATH = "../assets/haagent.tcss"
    BINDINGS = APP_BINDINGS

    def __init__(self, service: AssistantService) -> None:
        super().__init__()
        self.service = service
        self._state = "idle"
        self._conversation_lines: list[str] = []
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = False
        self._conversation_stick_to_bottom = True
        self._streaming_assistant_turn: int | None = None
        self._streaming_assistant_text = ""
        self._active_turn_index: int | None = None
        self._tool_details_enabled = False
        self._last_failure: FailureView | None = None
        self._pending_interaction: PendingInteraction | None = None
        self._default_prompt_placeholder = "输入消息；Enter 发送，Shift+Enter 换行"
        self._memory_mode = False
        self._memory_detail_mode = False
        self._memory_candidates: list[MemoryCandidate] = []
        self._memory_selected = 0
        self._memory_error: str | None = None
        self._memory_notice: str | None = None
        self._sandbox_status: dict[str, object] | None = None
        self._pending_external_prompt: str | None = None
        self._pending_external_path: Path | None = None
        self._pending_full_trust_prompt: str | None = None
        self._pending_full_trust_path: Path | None = None
        self._model_catalog_providers: list[object] | None = None
        self._commands = command_registry()
        self._theme_choice = select_theme()
        self._file_ref_overlay: FileReferenceOverlay | None = None
        self._file_ref_index: FileReferenceIndex | None = None
        self._command_suggestion_overlay: CommandSuggestionOverlay | None = None
        self.is_wide_external_root = is_wide_external_root
        self.permission_mode_label = _permission_mode_label

    def compose(self) -> ComposeResult:
        yield StatusBar("", id="status-bar")
        yield ResizeMessage("终端尺寸过小\n请调整到至少 80x24 后继续使用 HaAgent TUI。", id="resize-message", classes="hidden")
        with Horizontal(id="main"):
            yield ConversationTimeline(id="conversation", wrap=True, auto_scroll=True)
        with Vertical(id="input-panel"):
            yield PromptInput(placeholder=self._default_prompt_placeholder, id="prompt-input", show_line_numbers=False)
        yield FooterBar(footer_text("chat"), id="footer-bar")

    def on_mount(self) -> None:
        self._apply_theme()
        self._show_initial_configuration_state()
        self._restore_initial_session()
        self._refresh()
        self._update_responsive_layout()
        self.query_one("#prompt-input", PromptInput).focus()
        self._warm_file_reference_index()

    def _restore_initial_session(self) -> None:
        initial_resume = getattr(self.service, "initial_resume", None)
        if initial_resume is not None:
            try:
                status = self.service.resume_session(initial_resume)
            except Exception as error:
                self._append_block("Session warning", f"恢复会话失败：{error}")
            else:
                self._show_session_history(status, prefix="已恢复 session")
            return
        if not bool(getattr(self.service, "initial_continue", False)):
            return
        try:
            status = self.service.continue_latest_session()
        except Exception as error:
            self._append_block("Session warning", f"继续最新 session 失败：{error}")
        else:
            self._show_session_history(status, prefix="已继续最新 session")

    def on_resize(self, event: events.Resize) -> None:
        self._update_responsive_layout(width=event.size.width, height=event.size.height)

    def on_key(self, event: events.Key) -> None:
        if self._command_suggestion_overlay is not None and event.key in {"escape", "up", "down", "enter"}:
            self.action_handle_command_suggestion_key(event)
            return
        if self._file_ref_overlay is not None and event.key in {"escape", "up", "down", "enter"}:
            self.action_handle_file_ref_key(event)
            return
        if self._memory_mode and self._pending_interaction is None:
            if self._handle_memory_key(event.key):
                event.stop()
                return
        if self._pending_interaction is not None:
            return
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if self._prompt_value(prompt_input):
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
        if event.key == "enter" and self._memory_mode:
            event.stop()
            self.action_memory_enter()
        elif event.key in {"a", "y"} and self._memory_mode:
            event.stop()
            self.action_confirm_memory()
        elif event.key == "r" and self._memory_mode:
            event.stop()
            self.action_reject_memory()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "prompt-input" or self._file_ref_overlay is None:
            if event.text_area.id == "prompt-input":
                self._sync_command_suggestions_with_prompt(self._prompt_value(event.text_area))
            return
        text = self._prompt_value(event.text_area)
        query = query_after_at(text)
        if query is None:
            self._close_file_ref_overlay()
        else:
            self._file_ref_overlay.update_query(query)
        self._sync_command_suggestions_with_prompt(text)

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        self._submit_prompt(event.input)

    def action_submit_prompt(self) -> None:
        self._submit_prompt(self.query_one("#prompt-input", PromptInput))

    def _submit_prompt(self, prompt_input: PromptInput) -> None:
        prompt = self._prompt_value(prompt_input).strip()
        if not prompt:
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
        self._set_prompt_value(prompt_input, "")
        if is_prompt_mode_command(prompt):
            self._start_prompt(prompt)
            return
        if path_authorization.handle_prompt_path_authorization(self, prompt):
            return
        self._start_prompt(prompt)

    def _start_prompt(self, prompt: str) -> None:
        self._active_turn_index = self._next_turn_index()
        self._conversation_stick_to_bottom = True
        self.query_one("#conversation", ConversationTimeline).set_stick_to_bottom(True)
        self._append_block("You", prompt)
        self.query_one("#conversation", ConversationTimeline).start_assistant_response(
            turn_index=self._active_turn_index,
        )
        self._state = "running"
        self._refresh()
        self._run_prompt(prompt)

    def _next_turn_index(self) -> int:
        try:
            status = self.service.get_workspace_status()
        except Exception:
            return 0
        return (status.current_turn_count if status.current_turn_count is not None else 0) + 1

    def action_help(self) -> None:
        self.push_screen(HelpModal(self._help_context()))

    def action_open_sessions(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if prompt_input.has_focus and self._prompt_value(prompt_input):
            prompt_input.insert("s")
            return
        self.push_screen(SessionOverlay(self.service.list_sessions()), self._handle_session_overlay_result)

    def action_open_models(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if prompt_input.has_focus and self._prompt_value(prompt_input):
            return
        self.push_screen(ModelCenterOverlay(self.service.list_model_profiles()), self._handle_model_center_result)

    def action_open_search(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if prompt_input.has_focus and self._prompt_value(prompt_input):
            return
        self.push_screen(SearchOverlay(list(self._conversation_lines)), self._defer_prompt_focus)

    def action_open_permissions(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if prompt_input.has_focus and self._prompt_value(prompt_input):
            return
        self._show_permissions()

    def action_open_command_suggestions(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        value = self._prompt_value(prompt_input)
        if prompt_input.has_focus and value:
            if value.startswith("/") and " " not in value:
                self._open_command_suggestions(value.removeprefix("/"))
            else:
                prompt_input.insert("/")
            return
        if not value:
            prompt_input.insert("/")
        self._open_command_suggestions(self._prompt_value(prompt_input).removeprefix("/"))

    def action_open_file_refs(self) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        query = query_after_at(self._prompt_value(prompt_input))
        if query is None:
            return
        status = self.service.get_workspace_status()
        if self._file_ref_overlay is None:
            overlay = FileReferenceOverlay(status.workspace_root, query, self._file_ref_index)
            self._file_ref_overlay = overlay
            self.query_one("#input-panel", Vertical).mount(overlay, before=prompt_input)
            self.query_one("#input-panel", Vertical).styles.height = 14
        else:
            self._file_ref_overlay.update_query(query)
        prompt_input.focus()

    def action_handle_file_ref_key(self, event: events.Key) -> None:
        overlay = self._file_ref_overlay
        if overlay is None:
            return
        token = overlay.handle_key(event)
        if token is None:
            return
        if token:
            self._handle_file_reference_result(token)
        self._close_file_ref_overlay()

    def action_accept_file_ref(self) -> None:
        overlay = self._file_ref_overlay
        if overlay is None:
            return
        token = overlay.selected_token()
        if token:
            self._handle_file_reference_result(token)
            self._close_file_ref_overlay()

    def action_quit(self) -> None:
        self._request_current_task_cancel()
        self.exit(None)

    def action_toggle_theme(self) -> None:
        self._theme_choice = next_theme(self._theme_choice)
        self._apply_theme()
        if no_color_enabled():
            self._append_line("NO_COLOR 已启用，主题保持单色")
        else:
            self._append_line(f"主题已切换：{theme_label(self._theme_choice)}")
        self._refresh()

    def action_conversation_page_up(self) -> None:
        self._conversation_stick_to_bottom = False
        conversation = self.query_one("#conversation", ConversationView)
        conversation.set_stick_to_bottom(False)
        conversation.scroll_page_up(animate=False, force=True)

    def action_conversation_page_down(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        conversation.scroll_page_down(animate=False, force=True)
        self.call_after_refresh(self._sync_conversation_stickiness)

    def action_conversation_end(self) -> None:
        self._conversation_stick_to_bottom = True
        self.query_one("#conversation", ConversationTimeline).set_stick_to_bottom(True)
        self._scroll_conversation_to_end()

    def action_cancel_interaction(self) -> None:
        if self._memory_mode:
            if self._memory_detail_mode:
                self._memory_detail_mode = False
            else:
                self._memory_mode = False
                self.query_one("#prompt-input", PromptInput).focus()
            self._refresh()
            return
        if self._pending_interaction is None:
            return
        self._complete_interaction(HumanInteractionResponse(approved=False, answer=""))

    def action_toggle_memory(self) -> None:
        self._memory_mode = not self._memory_mode
        self._memory_detail_mode = False
        if self._memory_mode:
            self._load_memory_candidates()
            self._set_prompt_value(self.query_one("#prompt-input", PromptInput), "")
        else:
            self.query_one("#prompt-input", PromptInput).focus()
        self._refresh()

    def action_memory_enter(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        self._memory_detail_mode = not self._memory_detail_mode
        self._refresh()

    def action_memory_up(self) -> None:
        self._move_memory_selection(-1)

    def action_memory_down(self) -> None:
        self._move_memory_selection(1)

    def action_memory_first(self) -> None:
        if self._memory_mode and self._memory_candidates and not self._memory_detail_mode:
            self._memory_selected = 0
            self._refresh()

    def action_memory_last(self) -> None:
        if self._memory_mode and self._memory_candidates and not self._memory_detail_mode:
            self._memory_selected = len(self._memory_candidates) - 1
            self._refresh()

    def action_confirm_memory(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        candidate = self._selected_memory_candidate()
        try:
            self.service.confirm_memory_candidate(candidate.candidate_id)
        except Exception as error:
            self._memory_notice = f"Memory confirm failed: {error}"
            self._append_block("Memory warning", f"Memory confirm failed: {error}")
        else:
            self._memory_notice = f"已确认记忆候选：{candidate.candidate_id}"
            self._append_line(f"记忆已确认：{candidate.candidate_id}")
        self._memory_detail_mode = False
        self._load_memory_candidates()
        self._refresh()

    def action_reject_memory(self) -> None:
        if not self._memory_mode or not self._memory_candidates:
            return
        candidate = self._selected_memory_candidate()
        try:
            self.service.reject_memory_candidate(candidate.candidate_id, "rejected from TUI")
        except Exception as error:
            self._memory_notice = f"Memory reject failed: {error}"
            self._append_block("Memory warning", f"Memory reject failed: {error}")
        else:
            self._memory_notice = f"已拒绝记忆候选：{candidate.candidate_id}"
            self._append_line(f"记忆已拒绝：{candidate.candidate_id}")
        self._memory_detail_mode = False
        self._load_memory_candidates()
        self._refresh()

    def action_new_session(self) -> None:
        try:
            status = self.service.create_session()
        except Exception as error:
            self._append_block("Session warning", f"新建会话失败：{error}")
        else:
            self._clear_conversation_for_new_session()
        self._refresh()

    def action_resume_latest(self) -> None:
        try:
            status = self.service.continue_latest_session()
        except Exception as error:
            self._append_block("Session warning", f"继续最新会话失败：{error}")
        else:
            self._append_line(f"已恢复会话：{status.session_id}")
        self._refresh()

    def _handle_slash_command(self, result) -> None:
        if result.error:
            self._append_block("Command", result.error)
            self._refresh()
            return
        command = result.command
        if command is None:
            return
        if command.action == "help":
            self.action_help()
        elif command.action == "sessions":
            self.action_open_sessions()
        elif command.action == "compact_session":
            self.action_compact_session()
        elif command.action == "open_models":
            self.action_open_models()
        elif command.action == "mcp":
            self._handle_mcp_command()
        elif command.action == "agents":
            self._handle_agents_command()
        elif command.action == "memory":
            if not self._memory_mode:
                self.action_toggle_memory()
        elif command.action == "toggle_details":
            self._tool_details_enabled = not self._tool_details_enabled
            conversation = self.query_one("#conversation", ConversationTimeline)
            conversation.set_tool_details(self._tool_details_enabled)
            state = "开启" if self._tool_details_enabled else "关闭"
            self._append_block("Command", f"工具详情已{state}")
            self._refresh()
        elif command.action == "skills":
            self._handle_skills_command(result.argument)
        elif command.action == "skill":
            self._handle_skill_command(result.argument)
        elif command.action == "sandbox":
            self._handle_sandbox_command(result.argument)
        elif command.action == "web":
            self._handle_web_command(result.argument)
        elif command.action == "turns":
            self._handle_turns_command(result.argument)
        elif command.action == "permissions":
            self._show_permissions()
        elif command.action == "cancel_task":
            self.action_cancel_current_task()
        elif command.action == "new_session":
            self.action_new_session()
        elif command.action == "resume_latest":
            self.action_resume_latest()

    def _handle_turns_command(self, argument: str) -> None:
        usage = "用法：/turns [show|unlimited|COUNT]"
        parts = argument.strip().split()
        if not parts or parts == ["show"]:
            status = self.service.get_turn_limit_status()
            current = (
                "unlimited"
                if status.current_max_turns is None
                else str(status.current_max_turns)
            )
            self._append_block(
                "Command",
                (
                    f"当前 session turn 限制：{current}\n"
                    f"已保存交互默认值：{status.configured_interactive_max_turns}\n"
                    f"{usage}"
                ),
            )
            self._refresh()
            return
        if parts == ["unlimited"]:
            try:
                self.service.set_current_turns_unlimited()
            except AssistantServiceError as error:
                self._append_block("Command", str(error))
            else:
                self._append_block(
                    "Command",
                    "当前 session turn 限制已设为 unlimited；不会写入全局配置。",
                )
            self._refresh()
            return
        count_text = parts[0] if len(parts) == 1 else parts[1] if parts[0] == "set" and len(parts) == 2 else ""
        if not count_text.isdigit() or int(count_text) <= 0:
            self._append_block("Command", usage)
            self._refresh()
            return
        self.service.set_interactive_max_turns(int(count_text))
        self._append_block(
            "Command",
            f"已保存交互默认 turn 限制：{int(count_text)}；当前 session 已同步。",
        )
        self._refresh()

    def _handle_sandbox_command(self, argument: str) -> None:
        parts = argument.strip().split()
        usage = "用法：/sandbox [status|doctor|enable docker [--allow-fallback]|disable]"
        try:
            if not parts or parts == ["status"]:
                self._append_block("Command", _sandbox_status_text(self.service.get_sandbox_status()))
            elif parts == ["doctor"]:
                self._append_block("Command", _sandbox_doctor_text(self.service.get_sandbox_doctor_report()))
            elif parts[:2] == ["enable", "docker"]:
                extra = parts[2:]
                if any(item not in {"--allow-fallback", "--fail-if-unavailable"} for item in extra):
                    self._append_block("Command", usage)
                else:
                    allow_fallback = "--allow-fallback" in extra
                    status = self.service.enable_docker_sandbox(fail_if_unavailable=not allow_fallback)
                    self._sandbox_status = {
                        "backend": status.backend,
                        "availability": {
                            "degraded": status.degraded,
                            "reason": status.reason,
                        },
                    }
                    self._append_block(
                        "Command",
                        (
                            "Docker 沙箱已启用；新 session 生效。\n"
                            f"{_sandbox_status_text(status)}"
                        ),
                    )
            elif parts == ["disable"]:
                status = self.service.disable_sandbox()
                self._sandbox_status = {
                    "backend": status.backend,
                    "availability": {
                        "degraded": status.degraded,
                        "reason": status.reason,
                    },
                }
                self._append_block(
                    "Command",
                    (
                        "已恢复 local_subprocess；后续新 session 会使用本机执行。\n"
                        f"{_sandbox_status_text(status)}"
                    ),
                )
            else:
                self._append_block("Command", usage)
        except Exception as error:
            self._append_block("Command", f"沙箱设置失败：{error}")
        self._refresh()

    def action_compact_session(self) -> None:
        try:
            result = self.service.compact_current_session()
        except Exception as error:
            self._append_block("Command", f"压缩当前会话失败：{error}")
        else:
            if result.applied:
                self._append_block(
                    "Command",
                    (
                        "已压缩当前会话："
                        f"压缩 {result.compacted_turn_count} 轮，"
                        f"保留最近 {result.preserved_recent_count} 轮，"
                        f"节省约 {result.saved_chars} 字符。"
                    ),
                )
            else:
                self._append_block("Command", f"当前会话无需压缩：{result.reason}")
        self._refresh()

    def _handle_web_command(self, argument: str) -> None:
        value = argument.strip().lower()
        if value:
            self._append_block("Command", "用法：/web")
            self._refresh()
            return
        status = self.service.get_workspace_status()
        enabled = not status.web_enabled
        self.service.set_web_enabled(enabled)
        state = "开启" if enabled else "关闭"
        self._append_block("Command", f"联网已{state}；后续任务可使用 web_search / web_fetch。")
        self._refresh()

    def _handle_mcp_command(self) -> None:
        status = self.service.get_mcp_status()
        servers = status.get("servers", [])
        if not isinstance(servers, list) or not servers:
            self._append_block("Command", "No MCP servers configured.")
            self._refresh()
            return
        lines = ["MCP servers:"]
        for item in servers:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "unknown"))
            state = str(item.get("state", "configured"))
            detail = str(item.get("detail", "")).strip()
            if state == "connected":
                lines.append(
                    f"- {name}: connected (tools: {int(item.get('tool_count', 0))}, resources: {int(item.get('resource_count', 0))})"
                )
            elif detail:
                lines.append(f"- {name}: {state} - {detail}")
            else:
                lines.append(f"- {name}: {state}")
        self._append_block("Command", "\n".join(lines))
        self._refresh()

    def _handle_agents_command(self) -> None:
        try:
            agents = self.service.list_agents()
        except Exception as error:
            self._append_block("Agents", f"读取 worker 状态失败：{error}")
            self._refresh()
            return
        if not agents:
            self._append_block("Agents", "当前 session 没有 worker。")
            self._refresh()
            return
        lines = ["Workers:"]
        for item in agents:
            agent_id = str(item.get("agent_id", "unknown"))
            status = str(item.get("status", "unknown"))
            subagent_type = str(item.get("subagent_type", "worker"))
            description = str(item.get("description", "")).strip()
            suffix = f" - {description}" if description else ""
            lines.append(f"- {agent_id} [{subagent_type}] {status}{suffix}")
        self._append_block("Agents", "\n".join(lines))
        self._refresh()

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

    def _handle_external_directory_decision(self, decision: str | None) -> None:
        path_authorization.handle_external_directory_decision(self, decision)

    def _handle_external_full_trust_confirmed(self, confirmed: bool) -> None:
        path_authorization.handle_external_full_trust_confirmed(self, confirmed)

    def _set_next_turn_target_path(self, path: Path) -> None:
        setter = getattr(self.service, "set_next_turn_target_paths", None)
        if setter is None:
            return
        setter([path])

    def _handle_session_overlay_result(self, result: SessionOverlayResult | None) -> None:
        if result is None:
            self.set_timer(0.01, self._restore_prompt_focus)
            return
        try:
            if result.action == "resume" and result.session is not None:
                status = self.service.resume_session(result.session.session_path)
            elif result.action == "continue_latest":
                status = self.service.continue_latest_session()
            else:
                status = self.service.create_session()
        except Exception as error:
            self._append_block("Session warning", f"会话操作失败：{error}")
        else:
            if result.action == "new":
                self._clear_conversation_for_new_session()
            else:
                self._show_session_history(status, prefix="当前会话")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    def _handle_model_center_result(self, result: ModelCenterResult | None) -> None:
        if result is None:
            self.set_timer(0.01, self._restore_prompt_focus)
            return
        try:
            if result.action == "switch_session":
                status = self.service.switch_current_session_model(result.profile_name)
                profile_name = status.model_profile_name or result.profile_name
                self._append_line(f"模型已切换到当前会话：{profile_name}")
            elif result.action == "set_default":
                self.service.set_default_model_profile(result.profile_name)
                self._append_line(f"默认模型 profile 已设为：{result.profile_name}")
            elif result.action == "delete_profile" and result.profile_name is not None:
                self.push_screen(
                    ConfirmModal(
                        f"删除模型 profile：{result.profile_name}",
                        "删除后不会清理 keyring 中的 API key。确认删除？",
                    ),
                    lambda confirmed, profile_name=result.profile_name: self._handle_delete_model_profile_result(
                        profile_name,
                        confirmed,
                    ),
                )
                return
            elif result.action == "new_profile":
                self.push_screen(ModelCatalogLoadingOverlay())
                self._refresh_model_catalog_and_open_setup()
                return
            elif result.action == "manual_profile":
                self.push_screen(ManualModelSetupWizard(), self._handle_model_setup_result)
                return
            elif result.action == "refresh_catalog":
                self._refresh_model_catalog_only()
                return
            elif result.action == "test_profile" and result.profile_name is not None:
                self._run_model_connection_test(result.profile_name)
                return
        except Exception as error:
            self._append_block("Model warning", f"模型操作失败：{error}")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    def _handle_delete_model_profile_result(self, profile_name: str, confirmed: bool | None) -> None:
        if not confirmed:
            self.action_open_models()
            return
        try:
            self.service.delete_model_profile(profile_name)
        except Exception as error:
            self._append_block("Model warning", f"模型删除失败：{error}")
        else:
            self._append_line(f"模型 profile 已删除：{profile_name}")
        self._refresh()
        self.action_open_models()

    def _handle_model_setup_result(self, request) -> None:
        if request is None:
            self.set_timer(0.01, self._restore_prompt_focus)
            return
        try:
            record = self.service.configure_model_profile(request)
        except Exception as error:
            self._append_block("Model warning", f"模型配置失败：{error}")
        else:
            self._append_line(f"模型 profile 已保存：{record.name}")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    @work(thread=True, exclusive=True)
    def _refresh_model_catalog_and_open_setup(self) -> None:
        if self._model_catalog_providers is not None:
            self.call_from_thread(self._open_model_setup_wizard, list(self._model_catalog_providers))
            return
        try:
            catalog = self.service.get_model_catalog()
            providers = list(catalog.providers)
            if not _configurable_model_catalog_providers(providers):
                catalog = self.service.refresh_model_catalog()
                providers = list(catalog.providers)
        except Exception as error:
            self.call_from_thread(self._handle_model_catalog_error, error)
            return
        if _configurable_model_catalog_providers(providers):
            self._model_catalog_providers = providers
        self.call_from_thread(self._open_model_setup_wizard, providers)

    @work(thread=True, exclusive=True)
    def _refresh_model_catalog_only(self) -> None:
        try:
            catalog = self.service.refresh_model_catalog()
        except Exception as error:
            self.call_from_thread(self._handle_model_catalog_error, error)
            return
        providers = list(catalog.providers)
        self._model_catalog_providers = providers
        self.call_from_thread(self._handle_model_catalog_success, providers)

    @work(thread=True, exclusive=True)
    def _run_model_connection_test(self, profile_name: str) -> None:
        try:
            result = self.service.test_model_profile(profile_name)
        except Exception as error:
            self.call_from_thread(self._handle_model_catalog_error, error)
            return
        self.call_from_thread(self._handle_model_connection_test_result, result)

    def _open_model_setup_wizard(self, providers: list[object]) -> None:
        configurable_providers = _configurable_model_catalog_providers(providers)
        if not configurable_providers:
            self._dismiss_model_catalog_loading_overlay()
            self._append_block(
                "Model warning",
                "模型目录没有可配置模型。\n请刷新目录或检查网络；如果使用缓存，请删除损坏的 models_catalog_cache.json 后重试。",
            )
            self._refresh()
            self.set_timer(0.01, self._restore_prompt_focus)
            return
        self._dismiss_model_catalog_loading_overlay()
        self.push_screen(ModelSetupWizard(configurable_providers), self._handle_model_setup_result)

    def _handle_model_catalog_success(self, providers: list[object]) -> None:
        self._append_line(f"模型目录已刷新：{len(providers)} 个 provider")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    def _handle_model_connection_test_result(self, result) -> None:
        status = "OK" if bool(getattr(result, "ok", False)) else "失败"
        message = str(getattr(result, "message", ""))
        self._append_line(f"模型连接测试 {status}: {message}")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    def _handle_model_catalog_error(self, error: Exception) -> None:
        self._dismiss_model_catalog_loading_overlay()
        self._append_block("Model warning", f"模型操作失败：{error}")
        self._refresh()
        self.set_timer(0.01, self._restore_prompt_focus)

    def _dismiss_model_catalog_loading_overlay(self) -> None:
        if isinstance(self.screen, ModelCatalogLoadingOverlay):
            self.screen.dismiss(None)

    def _open_command_suggestions(self, query: str = "") -> None:
        self._close_file_ref_overlay()
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if self._command_suggestion_overlay is None:
            overlay = CommandSuggestionOverlay(self._commands.commands())
            self._command_suggestion_overlay = overlay
            self.query_one("#input-panel", Vertical).mount(overlay, before=prompt_input)
            self.query_one("#input-panel", Vertical).styles.height = 14
        self._command_suggestion_overlay.update_query(query)
        prompt_input.focus()

    def action_handle_command_suggestion_key(self, event: events.Key) -> None:
        overlay = self._command_suggestion_overlay
        if overlay is None:
            return
        result = overlay.handle_key(event)
        if result is None:
            return
        if result == "":
            self._close_command_suggestions()
            return
        self._close_command_suggestions()
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if is_prompt_mode_command(result.token):
            self._set_prompt_value(prompt_input, f"{result.token} ")
            return
        self._set_prompt_value(prompt_input, "")
        self._handle_slash_command(parse_slash_command(result.token, self._commands))

    def action_accept_command_suggestion(self) -> None:
        overlay = self._command_suggestion_overlay
        if overlay is None:
            return
        command = overlay.state.selected_command
        if command is None:
            self._close_command_suggestions()
            return
        self._close_command_suggestions()
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if is_prompt_mode_command(command.token):
            self._set_prompt_value(prompt_input, f"{command.token} ")
            return
        self._set_prompt_value(prompt_input, "")
        self._handle_slash_command(parse_slash_command(command.token, self._commands))

    def _sync_command_suggestions_with_prompt(self, text: str) -> None:
        if self._command_suggestion_overlay is None:
            return
        if not text.startswith("/") or " " in text:
            self._close_command_suggestions()
            return
        self._command_suggestion_overlay.update_query(text.removeprefix("/"))

    def _close_command_suggestions(self) -> None:
        overlay = self._command_suggestion_overlay
        self._command_suggestion_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self.query_one("#input-panel", Vertical).styles.height = 5
        self.query_one("#prompt-input", PromptInput).focus()

    def command_suggestions_is_open(self) -> bool:
        return self._command_suggestion_overlay is not None

    def _handle_file_reference_result(self, token: str | None) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        if token is not None:
            self._set_prompt_value(prompt_input, replace_at_query(self._prompt_value(prompt_input), token))
        prompt_input.focus()

    def _close_file_ref_overlay(self) -> None:
        overlay = self._file_ref_overlay
        self._file_ref_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self.query_one("#input-panel", Vertical).styles.height = 5
        self.query_one("#prompt-input", PromptInput).focus()

    def file_reference_is_open(self) -> bool:
        return self._file_ref_overlay is not None

    @work(thread=True, exclusive=True)
    def _warm_file_reference_index(self) -> None:
        status = self.service.get_workspace_status()
        index = build_file_reference_index(status.workspace_root)
        self.call_from_thread(self._set_file_reference_index, index)

    def _set_file_reference_index(self, index: FileReferenceIndex) -> None:
        self._file_ref_index = index

    def _restore_prompt_focus(self, _result: object | None = None) -> None:
        self.query_one("#prompt-input", PromptInput).focus()

    def _defer_prompt_focus(self, _result: object | None = None) -> None:
        self.set_timer(0.01, self._restore_prompt_focus)

    @work(thread=True, exclusive=True)
    def _run_prompt(self, prompt: str) -> None:
        try:
            result = self.service.run_prompt_events(
                prompt,
                event_sink=lambda event: self.call_from_thread(self._handle_chat_event, event),
                interaction_handler=self._handle_interaction,
            )
        except Exception as error:
            self.call_from_thread(self._handle_prompt_error, error)
            return
        status = str(getattr(result, "status", "completed"))
        self.call_from_thread(self._finish_prompt, status)

    def _handle_chat_event(self, event: RuntimeUiEvent) -> None:
        handle_runtime_ui_event(self, event)

    def _record_tool_line(self, line: str) -> None:
        self._append_line(line)

    def _record_tool_activity(self, turn_index: int, tool_name: str, status: str, summary: str) -> None:
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.add_tool_activity(
            ToolActivity(
                tool_name=tool_name,
                status=_tool_activity_status(status),
                summary=safe_summary(summary, 96),
                turn_index=turn_index,
            ),
        )

    def _record_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.add_tool_diagnostic(turn_index, tool_name, safe_summary(message, 120))

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

    def _complete_approval(self, approved: bool | None) -> None:
        self._complete_interaction(HumanInteractionResponse(approved=bool(approved), answer=""))

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
        prompt_input = self.query_one("#prompt-input", PromptInput)
        prompt_input.placeholder = f"回答 Agent 的问题：{safe_summary(question, 90)}"
        prompt_input.focus()

    def _restore_prompt_input(self) -> None:
        self.query_one("#prompt-input", PromptInput).placeholder = self._default_prompt_placeholder

    def _merge_assistant_delta(self, turn_index: int, delta: str) -> None:
        if not delta:
            return
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.update_assistant_delta(turn_index, delta)
        if self._streaming_assistant_turn != turn_index:
            self._streaming_assistant_turn = turn_index
            self._streaming_assistant_text = delta
            return
        self._streaming_assistant_text += delta

    def _finalize_assistant_message(self, turn_index: int, content: str) -> None:
        conversation = self.query_one("#conversation", ConversationTimeline)
        if self._streaming_assistant_turn == turn_index:
            final_content = content or self._streaming_assistant_text
            conversation.finalize_assistant(turn_index, final_content)
        else:
            if self._streaming_assistant_turn is not None:
                conversation.finalize_assistant(self._streaming_assistant_turn, self._streaming_assistant_text)
            conversation.finalize_assistant(turn_index, content)
        self._streaming_assistant_turn = None
        self._streaming_assistant_text = ""

    def _finalize_streaming_assistant_if_needed(self) -> None:
        if self._streaming_assistant_turn is None:
            return
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.finalize_assistant(self._streaming_assistant_turn, self._streaming_assistant_text)
        self._streaming_assistant_turn = None
        self._streaming_assistant_text = ""

    def _replace_last_assistant_block(self, body: str) -> None:
        assistant_title = BLOCK_TITLES.get("Assistant", "Assistant")
        replacement = f"{assistant_title}\n  {body}"
        for index in range(len(self._conversation_lines) - 1, -1, -1):
            if self._conversation_lines[index].startswith(f"{assistant_title}\n"):
                self._conversation_lines[index] = replacement
                self._conversation_placeholder_rendered = True
                self._conversation_rendered_count = 0
                return
        self._conversation_lines.append(replacement)

    def _prompt_value(self, prompt_input: PromptInput) -> str:
        return prompt_input.text

    def _set_prompt_value(self, prompt_input: PromptInput, value: str) -> None:
        prompt_input.load_text(value)
        prompt_input.move_cursor(_end_location(value))

    def _finish_prompt(self, status: str) -> None:
        self._finalize_streaming_assistant_if_needed()
        if status == "completed" and self._state not in {"waiting approval", "waiting input", "cancelled"}:
            self._state = "idle"
        elif status == "cancelled":
            self._state = "cancelled"
            self._append_block("Cancel", "任务已取消。你可以调整请求后再次提交。")
        elif status != "completed":
            self._state = "failed"
            if self._last_failure is None:
                self._last_failure = FailureView(
                    failed_stage="executing",
                    failure_category="Runtime Failure",
                    reason=status,
                    episode_path="unknown",
                )
                self._append_block("Failure", self._last_failure.block_text())
        self._refresh()

    def _handle_prompt_error(self, error: Exception) -> None:
        self._state = "failed"
        self._append_block("Failure", str(error))
        self._finalize_streaming_assistant_if_needed()
        self._refresh()

    def _show_initial_configuration_state(self) -> None:
        status = self.service.get_workspace_status()
        if status.profile_error is not None:
            self._append_block("Config", "未找到默认模型配置\n输入 /model 打开模型中心完成配置。")
        elif status.credential_store_available is False:
            reason = status.credential_store_error or "unknown"
            self._append_block("Config", f"系统凭据库不可用：{reason}\n输入 /model 重新选择凭据来源。")
        elif status.api_key_env and not status.api_key_available:
            self._append_block(
                "Config",
                f"API key 缺失：{status.api_key_env}\n输入 /model 可以配置或测试模型；HaAgent 不会显示真实 API key。",
            )

    def _refresh(self) -> None:
        status = self.service.get_workspace_status()
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(
            status_line(
                status,
                ui_state=self._state,
                width=self.size.width,
                sandbox_status=self._sandbox_status,
            ),
        )
        self._apply_status_classes(status_bar)
        self._refresh_conversation()
        self.query_one("#footer-bar", FooterBar).update_footer(footer_text(self._help_context()))
        self._apply_focus_classes()

    def _help_context(self) -> str:
        if self.size.width < self.MIN_WIDTH or self.size.height < self.MIN_HEIGHT:
            return "too_small"
        if self._pending_interaction is not None:
            return "pending_input" if self._pending_interaction.request.interaction_type == "user_input" else "approval"
        if self._memory_mode:
            return "memory_detail" if self._memory_detail_mode else "memory_list"
        return "chat"

    def action_cancel_current_task(self) -> None:
        result = self._request_current_task_cancel()
        if result is None:
            return
        status = str(getattr(result, "status", "cancelled"))
        if status == "unavailable":
            self._append_block("Cancel", "当前 service 未提供可取消协议；本轮不能安全取消。")
        elif status == "idle":
            self._state = "idle"
            self._append_block("Cancel", "当前没有仍在运行的任务。")
        else:
            self._state = "cancelling"
            self._append_block("Cancel", "任务正在取消，请等待当前运行态结束。")
        self._refresh()

    def _request_current_task_cancel(self):
        if self._state not in {"running", "waiting approval", "waiting input", "cancelling"}:
            return None
        cancel = getattr(self.service, "cancel_current_run", None)
        if cancel is None:
            return SimpleNamespace(status="unavailable")
        result = cancel()
        if self._pending_interaction is not None:
            self._pending_interaction.response = HumanInteractionResponse(approved=False, answer="")
            self._pending_interaction.done.set()
            self._pending_interaction = None
        self._restore_prompt_input()
        return result

    def _refresh_conversation(self) -> None:
        conversation = self.query_one("#conversation", ConversationTimeline)
        if self._memory_mode:
            conversation.show_memory(self._memory_panel_text())
            self._conversation_placeholder_rendered = False
            self._conversation_rendered_count = 0
            return
        if not conversation.plain_text:
            if not self._conversation_placeholder_rendered:
                conversation.show_placeholder()
                self._conversation_placeholder_rendered = True
            return
        self._conversation_rendered_count = len(self._conversation_lines)
        if self._conversation_should_stick_to_bottom(conversation):
            self._conversation_stick_to_bottom = True
            conversation.set_stick_to_bottom(True)
            self.call_after_refresh(self._scroll_conversation_to_end_if_sticky)
        else:
            self._conversation_stick_to_bottom = False
            conversation.set_stick_to_bottom(False)

    def _scroll_conversation_to_end(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        conversation.set_stick_to_bottom(True)
        conversation.scroll_to(y=conversation.max_scroll_y, animate=False, immediate=True, force=True)

    def _scroll_conversation_to_end_if_sticky(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        if self._conversation_should_stick_to_bottom(conversation):
            self._scroll_conversation_to_end()

    def _conversation_should_stick_to_bottom(self, conversation: ConversationView) -> bool:
        if not self._conversation_stick_to_bottom:
            return False
        return conversation.scroll_y >= conversation.max_scroll_y - 1

    def _sync_conversation_stickiness(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        self._conversation_stick_to_bottom = conversation.scroll_y >= conversation.max_scroll_y - 1
        conversation.set_stick_to_bottom(self._conversation_stick_to_bottom)

    def _load_memory_candidates(self, *, silent: bool = False) -> None:
        try:
            self._memory_candidates = self.service.list_memory_candidates(status="pending")
            self._memory_error = None
            if self._memory_selected >= len(self._memory_candidates):
                self._memory_selected = max(0, len(self._memory_candidates) - 1)
        except Exception as error:
            self._memory_candidates = []
            self._memory_error = str(error)
            if not silent:
                self._append_block("Memory warning", f"Memory candidates unavailable: {error}")

    def _selected_memory_candidate(self) -> MemoryCandidate:
        return self._memory_candidates[self._memory_selected]

    def _handle_memory_key(self, key: str) -> bool:
        if key == "up":
            self.action_memory_up()
            return True
        if key == "down":
            self.action_memory_down()
            return True
        if key == "g":
            self.action_memory_first()
            return True
        if key in {"G", "shift+g", "upper_g"}:
            self.action_memory_last()
            return True
        return False

    def _move_memory_selection(self, delta: int) -> None:
        if not self._memory_mode or self._memory_detail_mode or not self._memory_candidates:
            return
        next_index = self._memory_selected + delta
        self._memory_selected = min(max(next_index, 0), len(self._memory_candidates) - 1)
        self._refresh()

    def _memory_panel_text(self) -> str:
        return MemoryPanelPresenter(
            candidates=self._memory_candidates,
            selected_index=self._memory_selected,
            detail_mode=self._memory_detail_mode,
            notice=self._memory_notice,
            error=self._memory_error,
        ).render()

    def _footer_text(self) -> str:
        return footer_text(self._help_context())

    def _show_session_history(self, status, *, prefix: str) -> None:
        lines = []
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.clear_timeline()
        try:
            history = list(self.service.current_session_history())
        except Exception as error:
            lines.append(f"{prefix}历史读取失败：{error}")
            conversation.add_system("会话", lines[-1])
            history = []
        if not history and not lines:
            lines.append("")
        for turn in history:
            assistant_text = _session_turn_assistant_text(turn)
            lines.append(f"{BLOCK_TITLES['You']}\n  {turn.request}")
            lines.append(f"{BLOCK_TITLES['Assistant']}\n  {assistant_text}")
            conversation.add_user(turn.request, turn_index=turn.turn_index)
            conversation.add_assistant_message(assistant_text, turn_index=turn.turn_index)
            if turn.status != "completed":
                lines.append(f"状态：{turn.status}")
                conversation.add_system("状态", f"状态：{turn.status}", turn_index=turn.turn_index)
        self._conversation_lines = lines
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = False

    def _clear_conversation_for_new_session(self) -> None:
        self._conversation_lines = []
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = False
        self._streaming_assistant_turn = None
        self._streaming_assistant_text = ""
        self._active_turn_index = None
        self.query_one("#conversation", ConversationTimeline).clear_timeline()

    def _append_block(self, title: str, body: str) -> None:
        display_title = BLOCK_TITLES.get(title, title)
        self._conversation_lines.append(f"{display_title}\n  {body}")
        conversation = self.query_one("#conversation", ConversationTimeline)
        turn_index = self._active_turn_index or 0
        if title == "You":
            conversation.add_user(body, turn_index=turn_index)
        elif title == "Assistant":
            conversation.add_assistant_message(body, turn_index=turn_index)
        elif title == "Failure":
            conversation.add_failure(body, turn_index=turn_index)
        else:
            conversation.add_system(display_title, body, turn_index=turn_index)

    def _append_line(self, line: str) -> None:
        self._conversation_lines.append(line)
        conversation = self.query_one("#conversation", ConversationTimeline)
        conversation.add_system("系统", line, turn_index=self._active_turn_index or 0)

    def _apply_theme(self) -> None:
        for theme in textual_themes():
            if theme.name not in self.available_themes:
                self.register_theme(theme)
        self.theme = self._theme_choice.textual_theme
        for choice in ("theme-dark", "theme-light", "theme-monochrome"):
            self.screen.set_class(choice == self._theme_choice.css_class, choice)

    def _apply_status_classes(self, widget: StatusBar) -> None:
        semantic = status_semantic(self._state)
        for token in semantic_tokens():
            widget.set_class(semantic.css_class == f"status-{token.value}", f"status-{token.value}")

    def _apply_focus_classes(self) -> None:
        prompt = self.query_one("#prompt-input", PromptInput)
        conversation = self.query_one("#conversation", ConversationView)
        prompt.set_class(prompt.has_focus, "panel-focused")
        conversation.set_class(not prompt.has_focus or self._memory_mode, "panel-focused")

    def _update_responsive_layout(self, width: int | None = None, height: int | None = None) -> None:
        terminal_width = width if width is not None else self.size.width
        terminal_height = height if height is not None else self.size.height
        layout = layout_for_size(terminal_width, terminal_height)
        self.query_one("#resize-message", ResizeMessage).set_class(not layout.too_small, "hidden")
        self.query_one("#main", Horizontal).set_class(layout.too_small, "hidden")
        self.query_one("#input-panel", Vertical).set_class(layout.too_small, "hidden")
        self.query_one("#footer-bar", FooterBar).set_class(layout.too_small, "hidden")


def run_tui(service: AssistantService) -> int:
    HaAgentTuiApp(service).run()
    return 0


def _session_turn_assistant_text(turn: object) -> str:
    for field_name in ("assistant_final_response", "final_response", "response", "content"):
        value = getattr(turn, field_name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    summary = str(getattr(turn, "summary", "")).strip()
    extracted = _summary_field(summary, "assistant_final_response") or _summary_field(summary, "final_response")
    return extracted or summary


def _summary_field(summary: str, field_name: str) -> str:
    for line in summary.splitlines():
        normalized = line.strip()
        if normalized.startswith("- "):
            normalized = normalized[2:].lstrip()
        colon_prefix = f"{field_name}:"
        equals_prefix = f"{field_name}="
        if normalized.startswith(colon_prefix):
            return normalized.removeprefix(colon_prefix).strip()
        if normalized.startswith(equals_prefix):
            return normalized.removeprefix(equals_prefix).strip()
    return ""


def _configurable_model_catalog_providers(providers: list[object]) -> list[object]:
    return [
        provider
        for provider in providers
        if getattr(catalog_provider_capability(provider), "status", None) == "runnable"
        and list(getattr(provider, "models", []) or [])
    ]


def _tool_activity_status(status: str) -> ToolStatus:
    if status in {"running", "approval", "done", "failed"}:
        return status
    return "running"


def _sandbox_status_text(status: object) -> str:
    backend = getattr(status, "backend", "unknown")
    degraded = bool(getattr(status, "degraded", True))
    reason = str(getattr(status, "reason", "") or "")
    lines = [
        f"当前沙箱：{backend}",
        f"degraded={str(degraded).lower()}",
    ]
    if reason:
        lines.append(f"reason={reason}")
    if backend != "docker":
        lines.append("开启 Docker 隔离：haagent sandbox enable docker")
    else:
        lines.append("检查 Docker 可用性：haagent sandbox doctor")
    return "\n".join(lines)


def _sandbox_doctor_text(report: object) -> str:
    lines = [
        f"当前沙箱：{getattr(report, 'backend', 'unknown')}",
        f"ready={str(bool(getattr(report, 'ready', False))).lower()}",
        f"Docker CLI: {getattr(report, 'docker_cli', 'unknown')}",
        f"Docker daemon: {getattr(report, 'docker_daemon', 'unknown')}",
        f"image={getattr(report, 'image', 'unknown')}",
        f"auto_build_image={str(bool(getattr(report, 'auto_build_image', False))).lower()}",
    ]
    reason = str(getattr(report, "reason", "") or "")
    next_action = str(getattr(report, "next_action", "") or "")
    if reason:
        lines.append(f"reason={reason}")
    if next_action:
        lines.append(f"next_action={next_action}")
    return "\n".join(lines)


def _permission_mode_label(mode: str) -> str:
    if mode == "auto_approve":
        return "自动批准"
    if mode == "full_access":
        return "完全访问权限"
    return "请求批准"


