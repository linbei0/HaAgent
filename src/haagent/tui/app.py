"""
haagent/tui/app.py - HaAgent TUI 应用编排

组合 Textual 组件、协调会话状态和后台 worker，具体渲染与组件细节拆分到同级模块。
"""

from __future__ import annotations

import re
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import TextArea

from haagent.app.assistant_service import AssistantService
from haagent.models.gateway_registry import catalog_provider_capability
from haagent.memory import MemoryCandidate
from haagent.runtime.chat_session import ChatEvent
from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.tui.changes import ChangedFileSummary, changed_files_from_tool_event, merge_changed_files
from haagent.tui.command_suggestions import CommandSuggestionOverlay
from haagent.tui.commands import command_registry, parse_slash_command
from haagent.tui.copy import BLOCK_TITLES
from haagent.tui.failures import FailureView, failure_from_payload
from haagent.tui.file_ref_modal import FileReferenceOverlay
from haagent.tui.file_refs import FileReferenceIndex, build_file_reference_index, query_after_at, replace_at_query
from haagent.tui.keys import APP_BINDINGS, footer_text
from haagent.tui.modals import ConfirmModal, ExternalDirectoryDecisionModal, HelpModal, PermissionsModal, ToolApprovalModal, ToolDetailsModal
from haagent.tui.models import (
    ManualModelSetupWizard,
    ModelCatalogLoadingOverlay,
    ModelCenterOverlay,
    ModelCenterResult,
    ModelSetupWizard,
)
from haagent.tui.renderers import (
    failure_body,
    memory_panel_text,
    payload_text,
    side_bar,
    status_line,
)
from haagent.tui.state import MIN_HEIGHT, MIN_WIDTH, PendingInteraction, layout_for_size
from haagent.tui.theme import (
    next_theme,
    no_color_enabled,
    select_theme,
    semantic_tokens,
    status_badge,
    status_semantic,
    textual_themes,
    theme_label,
)
from haagent.tui.tool_timeline import ToolTimelineState
from haagent.tui.search_modal import SearchOverlay
from haagent.tui.sessions import SessionOverlay, SessionOverlayResult
from haagent.tui.utils import safe_summary
from haagent.tui.widgets import ConversationView, FooterBar, PromptInput, ResizeMessage, SideBar, StatusBar, _end_location


WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r'(?:"([A-Za-z]:[\\/][^"\r\n]+)"|([A-Za-z]:[\\/][^\s"\']+))')
POSIX_ABSOLUTE_PATH_PATTERN = re.compile(r'(?:"(/[^"\r\n]+)"|(?<!\S)(/[^ \t\r\n"\']+))')


def find_untrusted_absolute_paths(
    text: str,
    *,
    project_root: Path,
    external_roots: list[dict[str, str]] | None = None,
) -> list[Path]:
    roots = [project_root.resolve()]
    for item in external_roots or []:
        raw_path = item.get("path")
        if raw_path:
            roots.append(Path(raw_path).resolve())
    matches: list[Path] = []
    for candidate in _absolute_path_candidates(text):
        resolved = candidate.resolve()
        if any(resolved == root or root in resolved.parents for root in roots):
            continue
        if resolved not in matches:
            matches.append(resolved)
    return matches


def _absolute_path_candidates(text: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in (WINDOWS_ABSOLUTE_PATH_PATTERN, POSIX_ABSOLUTE_PATH_PATTERN):
        for match in pattern.finditer(text):
            raw = next((group for group in match.groups() if group), "")
            if raw:
                paths.append(Path(raw.rstrip(".,;，。；")))
    return paths


def is_wide_external_root(path: Path) -> bool:
    resolved = path.resolve()
    home = Path.home().resolve()
    risky_roots = {home, home / "Desktop", home / "Downloads", home / "Documents"}
    if resolved.parent == resolved:
        return True
    if resolved.anchor and str(resolved) == resolved.anchor:
        return True
    return resolved in {root.resolve() for root in risky_roots}


class HaAgentTuiApp(App[None]):
    MIN_WIDTH = MIN_WIDTH
    MIN_HEIGHT = MIN_HEIGHT
    CSS_PATH = "haagent.tcss"
    BINDINGS = APP_BINDINGS

    def __init__(self, service: AssistantService) -> None:
        super().__init__()
        self.service = service
        self._state = "idle"
        self._conversation_lines: list[str] = []
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = False
        self._tool_lines: list[str] = []
        self._timeline = ToolTimelineState()
        self._changed_files: list[ChangedFileSummary] = []
        self._last_failure: FailureView | None = None
        self._pending_decision: str | None = None
        self._pending_interaction: PendingInteraction | None = None
        self._default_prompt_placeholder = "输入消息；Enter 发送，Shift+Enter 换行"
        self._memory_mode = False
        self._memory_detail_mode = False
        self._memory_candidates: list[MemoryCandidate] = []
        self._memory_selected = 0
        self._memory_error: str | None = None
        self._memory_notice: str | None = None
        self._pending_external_prompt: str | None = None
        self._pending_external_path: Path | None = None
        self._pending_full_trust_prompt: str | None = None
        self._pending_full_trust_path: Path | None = None
        self._model_catalog_providers: list[object] | None = None
        self._commands = command_registry()
        self._theme_choice = select_theme()
        self._file_ref_overlay: FileReferenceOverlay | None = None
        self._file_ref_index: FileReferenceIndex | None = None

    def compose(self) -> ComposeResult:
        yield StatusBar("", id="status-bar")
        yield ResizeMessage("终端尺寸过小\n请调整到至少 80x24 后继续使用 HaAgent TUI。", id="resize-message", classes="hidden")
        with Horizontal(id="main"):
            yield ConversationView(id="conversation", wrap=True, auto_scroll=True)
            yield SideBar("", id="side-bar")
        with Vertical(id="input-panel"):
            yield PromptInput(placeholder=self._default_prompt_placeholder, id="prompt-input", show_line_numbers=False)
        yield FooterBar(footer_text("chat"), id="footer-bar")

    def on_mount(self) -> None:
        self._apply_theme()
        self.query_one("#side-bar", SideBar).can_focus = True
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
            return
        query = query_after_at(self._prompt_value(event.text_area))
        if query is None:
            self._close_file_ref_overlay()
            return
        self._file_ref_overlay.update_query(query)

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
        self._set_prompt_value(prompt_input, "")
        if self._pending_interaction is not None and self._pending_interaction.request.interaction_type == "user_input":
            self._complete_interaction(HumanInteractionResponse(approved=True, answer=prompt))
            return
        status = self.service.get_workspace_status()
        untrusted_paths = find_untrusted_absolute_paths(
            prompt,
            project_root=status.workspace_root,
            external_roots=status.external_roots or [],
        )
        if untrusted_paths:
            if getattr(status, "permission_mode", "request_approval") == "full_access":
                self._set_next_turn_target_path(untrusted_paths[0])
                self._append_block("Permissions", "完全访问权限已启用；本轮不会限制工作区外路径。")
                self._start_prompt(prompt)
                return
            self._pending_external_prompt = prompt
            self._pending_external_path = untrusted_paths[0]
            self.push_screen(ExternalDirectoryDecisionModal(untrusted_paths[0]), self._handle_external_directory_decision)
            return
        self._start_prompt(prompt)

    def _start_prompt(self, prompt: str) -> None:
        self._append_block("You", prompt)
        if self._state == "cancelled":
            self._pending_decision = None
        self._state = "running"
        self._refresh()
        self._run_prompt(prompt)

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
        if prompt_input.has_focus and self._prompt_value(prompt_input):
            prompt_input.insert("/")
            return
        self.push_screen(CommandSuggestionOverlay(self._commands.commands()), self._handle_command_suggestion_result)

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
        self.query_one("#conversation", ConversationView).scroll_page_up(animate=False, force=True)

    def action_conversation_page_down(self) -> None:
        self.query_one("#conversation", ConversationView).scroll_page_down(animate=False, force=True)

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
            self.query_one("#side-bar", SideBar).focus()
            self.set_timer(0.01, self._focus_side_bar)
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

    def action_focus_tools(self) -> None:
        self.query_one("#side-bar", SideBar).focus()

    def action_new_session(self) -> None:
        try:
            status = self.service.create_session()
        except Exception as error:
            self._append_block("Session warning", f"新建会话失败：{error}")
        else:
            self._append_line(f"新建会话：{status.session_id}")
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
        elif command.action == "open_models":
            self.action_open_models()
        elif command.action == "memory":
            if not self._memory_mode:
                self.action_toggle_memory()
        elif command.action == "tools":
            self.action_focus_tools()
            if self.query_one("#side-bar", SideBar).has_class("hidden"):
                self.action_open_tool_details()
        elif command.action == "web":
            self._handle_web_command(result.argument)
        elif command.action == "permissions":
            self._show_permissions()
        elif command.action == "cancel_task":
            self.action_cancel_current_task()
        elif command.action == "new_session":
            self.action_new_session()
        elif command.action == "resume_latest":
            self.action_resume_latest()

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

    def _show_permissions(self) -> None:
        status = self.service.get_workspace_status()
        self.push_screen(
            PermissionsModal(
                status.workspace_root,
                status.external_roots or [],
                getattr(status, "permission_mode", "request_approval"),
            ),
            self._handle_permissions_result,
        )

    def _handle_permissions_result(self, result: dict[str, object] | None) -> None:
        if result is None:
            self._restore_prompt_focus()
            return
        action = result.get("action")
        try:
            if action == "clear":
                self.push_screen(
                    ConfirmModal("清空外部目录授权", "清空后本会话将只保留当前项目根权限。确认？"),
                    self._handle_clear_external_roots_confirmed,
                )
                return
            if action == "set_mode":
                mode = str(result.get("mode", "request_approval"))
                if mode == "full_access":
                    self.push_screen(
                        ConfirmModal(
                            "启用完全访问权限",
                            "启用后，本会话不再限制工作区外文件读写和执行目录。\n你将承担该模式下的本地文件和命令风险。确认？",
                        ),
                        lambda confirmed, mode=mode: self._handle_permission_mode_confirmed(mode, confirmed),
                    )
                    return
                self._set_permission_mode(mode)
                return
            path = Path(str(result.get("path", "")))
            if action == "remove":
                self.service.remove_external_root(path)
                self._append_block("Permissions", f"已移除外部目录：{path}")
            elif action == "set_access":
                access = str(result.get("access"))
                if access == "full" and is_wide_external_root(path):
                    self.push_screen(
                        ConfirmModal("完全信任高风险目录", f"将允许读取、修改并在该目录执行命令：\n{path}\n确认？"),
                        lambda confirmed, path=path: self._handle_set_full_access_confirmed(path, confirmed),
                    )
                    return
                self.service.set_external_root_access(path, access)
                label = "只读参考" if access == "read" else "完全信任"
                self._append_block("Permissions", f"已设为{label}：{path}")
        except Exception as error:
            self._append_block("Permissions warning", f"权限操作失败：{error}")
        self._refresh()
        self._restore_prompt_focus()

    def _set_permission_mode(self, mode: str) -> None:
        try:
            self.service.set_permission_mode(mode)
        except Exception as error:
            self._append_block("Permissions warning", f"权限模式切换失败：{error}")
        else:
            self._append_block("Permissions", f"权限模式已切换为：{_permission_mode_label(mode)}")
        self._refresh()
        self._restore_prompt_focus()

    def _handle_permission_mode_confirmed(self, mode: str, confirmed: bool) -> None:
        if confirmed:
            self._set_permission_mode(mode)
            return
        self._append_block("Permissions", "已取消启用完全访问权限。")
        self._refresh()
        self._restore_prompt_focus()

    def _handle_clear_external_roots_confirmed(self, confirmed: bool) -> None:
        if confirmed:
            try:
                self.service.clear_external_roots()
            except Exception as error:
                self._append_block("Permissions warning", f"清空外部目录授权失败：{error}")
            else:
                self._append_block("Permissions", "已清空本会话外部目录授权。")
        self._refresh()
        self._restore_prompt_focus()

    def _handle_set_full_access_confirmed(self, path: Path, confirmed: bool) -> None:
        if confirmed:
            try:
                self.service.set_external_root_access(path, "full")
            except Exception as error:
                self._append_block("Permissions warning", f"权限操作失败：{error}")
            else:
                self._append_block("Permissions", f"已完全信任：{path}")
        self._refresh()
        self._restore_prompt_focus()

    def _handle_external_directory_decision(self, decision: str | None) -> None:
        prompt = self._pending_external_prompt
        path = self._pending_external_path
        self._pending_external_prompt = None
        self._pending_external_path = None
        if prompt is None or path is None:
            self._restore_prompt_focus()
            return
        try:
            if decision == "read":
                self.service.add_external_root(path, "read")
                self._set_next_turn_target_path(path)
                self._append_block("Permissions", f"已作为只读参考加入：{path}")
            elif decision == "full":
                if is_wide_external_root(path):
                    self._pending_full_trust_prompt = prompt
                    self._pending_full_trust_path = path
                    self.push_screen(
                        ConfirmModal("完全信任高风险目录", f"将允许读取、修改并在该目录执行命令：\n{path}\n确认？"),
                        self._handle_external_full_trust_confirmed,
                    )
                    return
                self.service.add_external_root(path, "full")
                self._set_next_turn_target_path(path)
                self._append_block("Permissions", f"已完全信任：{path}")
            elif decision == "switch":
                self.service.switch_project_root(path)
                self._append_block("Permissions", f"已切换工作区：{path}")
            else:
                self._append_block("Permissions", f"已取消外部目录授权：{path}")
                self._refresh()
                self._restore_prompt_focus()
                return
        except Exception as error:
            self._append_block("Permissions warning", f"外部目录授权失败：{error}")
            self._refresh()
            self._restore_prompt_focus()
            return
        self._refresh()
        self._start_prompt(prompt)

    def _handle_external_full_trust_confirmed(self, confirmed: bool) -> None:
        prompt = self._pending_full_trust_prompt
        path = self._pending_full_trust_path
        self._pending_full_trust_prompt = None
        self._pending_full_trust_path = None
        if prompt is None or path is None:
            self._restore_prompt_focus()
            return
        if not confirmed:
            self._append_block("Permissions", f"已取消完全信任：{path}")
            self._refresh()
            self._restore_prompt_focus()
            return
        try:
            self.service.add_external_root(path, "full")
            self._set_next_turn_target_path(path)
        except Exception as error:
            self._append_block("Permissions warning", f"外部目录授权失败：{error}")
            self._refresh()
            self._restore_prompt_focus()
            return
        self._append_block("Permissions", f"已完全信任：{path}")
        self._refresh()
        self._start_prompt(prompt)

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
                self._conversation_lines = [f"当前会话：{status.session_id}"]
                self._conversation_rendered_count = 0
                self._conversation_placeholder_rendered = True
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

    def _handle_command_suggestion_result(self, command) -> None:
        if command is not None:
            self._handle_slash_command(parse_slash_command(command.token, self._commands))
        self.set_timer(0.01, self._restore_prompt_focus)

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

    def _focus_side_bar(self) -> None:
        self.query_one("#side-bar", SideBar).focus()

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

    def _handle_chat_event(self, event: ChatEvent) -> None:
        event_type = event.event_type
        payload = event.payload
        if event_type == "assistant_message":
            self._append_block("Assistant", payload_text(payload, "content", event.message))
        elif event_type == "tool_started":
            self._record_tool_event(event)
        elif event_type == "tool_finished":
            self._record_tool_event(event)
        elif event_type == "tool_failed":
            self._record_tool_event(event)
        elif event_type == "approval_requested":
            self._state = "waiting approval"
            tool_name = payload_text(payload, "tool_name", "unknown")
            self._pending_decision = f"{tool_name}: {payload_text(payload, 'question', event.message)}"
            self._timeline.apply_event(event)
            self._tool_lines.append(f"{tool_name} pending approval")
            self._append_line(f"工具 {tool_name} {status_badge('pending approval')} (pending approval)")
        elif event_type == "user_input_requested":
            self._state = "waiting input"
            question = payload_text(payload, "question", event.message)
            self._pending_decision = question
            self._set_answer_required(question)
            self._append_block("Answer required", question)
        elif event_type == "approval_granted":
            tool_name = payload_text(payload, "tool_name", "unknown")
            self._state = "running"
            self._pending_decision = None
            self._timeline.apply_event(event)
            self._tool_lines.append(f"{tool_name} approved")
            self._append_line(f"审批已允许：{tool_name}")
        elif event_type == "approval_denied":
            tool_name = payload_text(payload, "tool_name", "unknown")
            self._pending_decision = None
            self._timeline.apply_event(event)
            self._tool_lines.append(f"{tool_name} denied")
            self._append_line(f"审批已拒绝：{tool_name}")
        elif event_type == "user_input_received":
            self._handle_user_input_received(event)
        elif event_type == "memory_candidates_created":
            self._handle_memory_candidates_created(event)
        elif event_type == "memory_extraction_warning":
            self._append_block("Memory warning", payload_text(payload, "message", event.message))
        elif event_type == "failure":
            self._handle_failure_event(event)
        self._refresh()

    def _record_tool_line(self, line: str) -> None:
        self._tool_lines.append(line)
        self._append_line(line)

    def _record_tool_event(self, event: ChatEvent) -> None:
        self._timeline.apply_event(event)
        tool_name = payload_text(event.payload, "tool_name", "unknown")
        status = "running"
        if event.event_type == "tool_finished":
            status = "done"
            self._record_changed_files(event)
        elif event.event_type == "tool_failed":
            status = "failed"
        self._record_tool_line(f"工具 {tool_name} {status_badge(status)} ({status})")

    def _record_changed_files(self, event: ChatEvent) -> None:
        status = self.service.get_workspace_status()
        tool_name = payload_text(event.payload, "tool_name", "unknown")
        new_items = changed_files_from_tool_event(
            tool_name,
            args_summary=event.payload.get("args_summary") if isinstance(event.payload.get("args_summary"), dict) else {},
            result_summary=event.payload.get("result_summary") if isinstance(event.payload.get("result_summary"), dict) else {},
            workspace_root=status.workspace_root,
        )
        self._changed_files = merge_changed_files(self._changed_files, new_items)

    def _handle_user_input_received(self, event: ChatEvent) -> None:
        tool_name = payload_text(event.payload, "tool_name", "request_user_input")
        self._state = "running"
        self._pending_decision = None
        if event.payload.get("approved") is False:
            self._append_line(f"回答已取消：{tool_name}")
        else:
            self._append_line(f"回答已提交：{tool_name}")

    def _handle_memory_candidates_created(self, event: ChatEvent) -> None:
        message = payload_text(
            event.payload,
            "message",
            event.message or "发现可记忆候选，已放入候选队列，等待你确认。",
        )
        self._append_block("Memory", message)
        self._memory_notice = message
        self._memory_mode = True
        self._memory_detail_mode = False
        self._load_memory_candidates(silent=True)

    def _handle_failure_event(self, event: ChatEvent) -> None:
        self._state = "failed"
        self._pending_decision = None
        self._last_failure = failure_from_payload(event.payload, event.message)
        self._append_block("Failure", self._last_failure.block_text())

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
        else:
            self._state = "waiting input"
            self._set_answer_required(request.question)
        self._refresh()

    def _complete_approval(self, approved: bool | None) -> None:
        self._complete_interaction(HumanInteractionResponse(approved=bool(approved), answer=""))

    def _complete_interaction(self, response: HumanInteractionResponse) -> None:
        pending = self._pending_interaction
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._pending_interaction = None
        self._pending_decision = None
        self._restore_prompt_input()
        self._state = "running"
        self._refresh()

    def _set_answer_required(self, question: str) -> None:
        prompt_input = self.query_one("#prompt-input", PromptInput)
        prompt_input.placeholder = f"回答 Agent 的问题：{safe_summary(question, 90)}"
        prompt_input.focus()

    def _restore_prompt_input(self) -> None:
        self.query_one("#prompt-input", PromptInput).placeholder = self._default_prompt_placeholder

    def _prompt_value(self, prompt_input: PromptInput) -> str:
        return prompt_input.text

    def _set_prompt_value(self, prompt_input: PromptInput, value: str) -> None:
        prompt_input.load_text(value)
        prompt_input.move_cursor(_end_location(value))

    def _finish_prompt(self, status: str) -> None:
        if self._state == "cancelled":
            self._refresh()
            return
        if status == "completed" and self._state not in {"waiting approval", "waiting input", "cancelled"}:
            self._state = "idle"
        elif status == "cancelled":
            self._state = "cancelled"
        elif status != "completed":
            self._state = "failed"
            if self._last_failure is None:
                failed_tool = next((item for item in reversed(self._timeline.items) if item.status == "failed"), None)
                reason = failed_tool.error_summary if failed_tool is not None else status
                category = "Tool Failure" if failed_tool is not None else "Runtime Failure"
                self._last_failure = FailureView(
                    failed_stage="executing",
                    failure_category=category,
                    reason=reason or status,
                    episode_path=failed_tool.episode_path if failed_tool is not None else "unknown",
                )
                self._append_block("Failure", self._last_failure.block_text())
        self._refresh()

    def _handle_prompt_error(self, error: Exception) -> None:
        self._state = "failed"
        self._append_block("Failure", str(error))
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
        status_bar.update_status(status_line(status, ui_state=self._state, width=self.size.width))
        self._apply_status_classes(status_bar)
        self._refresh_conversation()
        side = self.query_one("#side-bar", SideBar)
        side.update_content(
            side_bar(
                status,
                ui_state=self._state,
                timeline=self._timeline,
                changed_files=self._changed_files,
                pending_decision=self._pending_decision,
                last_failure=self._last_failure,
                memory_text=self._memory_panel_text() if self._memory_mode else None,
            ),
        )
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

    def on_side_bar_move_selection(self, message: SideBar.MoveSelection) -> None:
        if self._memory_mode:
            return
        self._timeline.move(message.delta)
        self._refresh()

    def on_side_bar_open_details(self, message: SideBar.OpenDetails) -> None:
        if self._memory_mode:
            return
        self.action_open_tool_details()

    def action_open_tool_details(self) -> None:
        item = self._timeline.selected_item()
        if item is None:
            self.push_screen(HelpModal("chat"))
            return
        self.push_screen(ToolDetailsModal(item), self._defer_prompt_focus)

    def action_cancel_current_task(self) -> None:
        if self._state not in {"running", "waiting approval", "waiting input"}:
            return
        cancel = getattr(self.service, "cancel_current_run", None)
        if cancel is None:
            self._append_block("Cancel", "当前 service 未提供可取消协议；本轮不能安全取消。")
            self._refresh()
            return
        cancel()
        if self._pending_interaction is not None:
            self._pending_interaction.response = HumanInteractionResponse(approved=False, answer="")
            self._pending_interaction.done.set()
            self._pending_interaction = None
        self._pending_decision = None
        self._restore_prompt_input()
        self._state = "cancelled"
        self._append_block("Cancel", "任务已取消。你可以调整请求后再次提交。")
        self._refresh()

    def _refresh_conversation(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        if self._memory_mode:
            conversation.show_memory(self._memory_panel_text())
            self._conversation_placeholder_rendered = False
            self._conversation_rendered_count = 0
            return
        if not self._conversation_lines:
            if not self._conversation_placeholder_rendered:
                conversation.show_placeholder()
                self._conversation_placeholder_rendered = True
            return
        if self._conversation_placeholder_rendered or self._conversation_rendered_count > len(self._conversation_lines):
            conversation.clear()
            self._conversation_rendered_count = 0
            self._conversation_placeholder_rendered = False
        conversation.append_lines(self._conversation_lines, start=self._conversation_rendered_count)
        self._conversation_rendered_count = len(self._conversation_lines)
        self.call_after_refresh(self._scroll_conversation_to_end)

    def _scroll_conversation_to_end(self) -> None:
        conversation = self.query_one("#conversation", ConversationView)
        conversation.scroll_to(y=conversation.max_scroll_y, animate=False, immediate=True, force=True)

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
        return memory_panel_text(
            candidates=self._memory_candidates,
            selected_index=self._memory_selected,
            detail_mode=self._memory_detail_mode,
            notice=self._memory_notice,
            error=self._memory_error,
        )

    def _footer_text(self) -> str:
        return footer_text(self._help_context())

    def _show_session_history(self, status, *, prefix: str) -> None:
        lines = []
        try:
            history = list(self.service.current_session_history())
        except Exception as error:
            lines.append(f"{prefix}历史读取失败：{error}")
            history = []
        if not history and not lines:
            lines.append("")
        for turn in history:
            lines.append(f"{BLOCK_TITLES['You']}\n  {turn.request}")
            lines.append(f"{BLOCK_TITLES['Assistant']}\n  {turn.summary}")
            if turn.status != "completed":
                lines.append(f"状态：{turn.status}")
        self._conversation_lines = lines
        self._conversation_rendered_count = 0
        self._conversation_placeholder_rendered = True

    def _append_block(self, title: str, body: str) -> None:
        self._conversation_lines.append(f"{BLOCK_TITLES.get(title, title)}\n  {body}")

    def _append_line(self, line: str) -> None:
        self._conversation_lines.append(line)

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
        side = self.query_one("#side-bar", SideBar)
        prompt = self.query_one("#prompt-input", PromptInput)
        conversation = self.query_one("#conversation", ConversationView)
        side.set_class(side.has_focus or self._memory_mode, "panel-focused")
        prompt.set_class(prompt.has_focus, "panel-focused")
        conversation.set_class(not side.has_focus and not prompt.has_focus and not self._memory_mode, "panel-focused")

    def _update_responsive_layout(self, width: int | None = None, height: int | None = None) -> None:
        terminal_width = width if width is not None else self.size.width
        terminal_height = height if height is not None else self.size.height
        layout = layout_for_size(terminal_width, terminal_height)
        self.query_one("#resize-message", ResizeMessage).set_class(not layout.too_small, "hidden")
        self.query_one("#main", Horizontal).set_class(layout.too_small, "hidden")
        self.query_one("#input-panel", Vertical).set_class(layout.too_small, "hidden")
        self.query_one("#footer-bar", FooterBar).set_class(layout.too_small, "hidden")
        self.query_one("#side-bar", SideBar).set_class(not layout.show_side_bar, "hidden")


def run_tui(service: AssistantService) -> int:
    HaAgentTuiApp(service).run()
    return 0


def _configurable_model_catalog_providers(providers: list[object]) -> list[object]:
    return [
        provider
        for provider in providers
        if getattr(catalog_provider_capability(provider), "status", None) == "runnable"
        and list(getattr(provider, "models", []) or [])
    ]


def _permission_mode_label(mode: str) -> str:
    if mode == "auto_approve":
        return "自动批准"
    if mode == "full_access":
        return "完全访问权限"
    return "请求批准"
