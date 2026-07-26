"""
src/haagent/tui/widgets/input_dock.py - TUI 输入停靠区组件

统一管理输入区 overlay、prompt 读写和焦点恢复，避免 App 直接操控布局细节。
"""

from __future__ import annotations

from pathlib import Path

from textual.widget import Widget
from textual.containers import Vertical
from textual.css.query import NoMatches

from haagent.runtime.execution.human_interaction import HumanInteractionRequest, UserQuestion
from haagent.tui.commands import command_registry
from haagent.tui.commands.suggestions import CommandSuggestionOverlay
from haagent.tui.files.overlay import FileReferenceOverlay
from haagent.tui.files.refs import FileReferenceIndex
from haagent.tui.widgets.prompt_input import PromptInput
from haagent.tui.widgets.question_prompt import QuestionPrompt
from haagent.tui.widgets.plan_confirmation import PlanConfirmationPanel
from haagent.tui.widgets.todo_panel import TodoPanel


class InputDock(Vertical):
    """输入区容器：负责 prompt 与补全 overlay 的 DOM 生命周期。"""

    COLLAPSED_HEIGHT = 5
    EXPANDED_HEIGHT = 14
    PLAN_HEIGHT = 10

    def __init__(
        self,
        *children: Widget,
        workspace_root: Path | None = None,
        file_reference_index: FileReferenceIndex | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*children, **kwargs)
        self.workspace_root = workspace_root or Path.cwd()
        self.file_reference_index = file_reference_index
        self.command_overlay: CommandSuggestionOverlay | None = None
        self.file_ref_overlay: FileReferenceOverlay | None = None
        self.question_prompt: QuestionPrompt | None = None
        self.plan_confirmation: PlanConfirmationPanel | None = None

    def open_question_prompt(self, questions: tuple[UserQuestion, ...]) -> QuestionPrompt:
        self.close_command_suggestions()
        self.close_file_refs()
        prompt = self._prompt()
        # 交互期间彻底隔离普通 prompt 的 slash、附件、文件引用和历史输入事件。
        prompt.display = False
        if self.question_prompt is None:
            self.question_prompt = QuestionPrompt(questions, id="question-prompt")
            self.mount(self.question_prompt, before=prompt)
        self.styles.height = self.EXPANDED_HEIGHT
        # mount 完成后再恢复焦点，避免 Textual 将按键投递给已隐藏的 PromptInput。
        self.call_after_refresh(self.question_prompt.focus)
        return self.question_prompt

    def close_question_prompt(self) -> None:
        question_prompt = self.question_prompt
        self.question_prompt = None
        if question_prompt is not None and question_prompt.is_mounted:
            question_prompt.remove()
        prompt = self._prompt()
        prompt.display = True
        self._collapse_if_idle()

    def open_plan_confirmation(self, request: HumanInteractionRequest) -> PlanConfirmationPanel:
        if request.plan_id is None or request.plan_revision is None or request.plan_proposal is None:
            raise ValueError("plan_confirmation requires plan id, revision, and proposal")
        self.close_command_suggestions()
        self.close_file_refs()
        self.close_question_prompt()
        prompt = self._prompt()
        prompt.display = False
        if self.plan_confirmation is None:
            self.plan_confirmation = PlanConfirmationPanel(
                request.plan_id,
                request.plan_revision,
                request.plan_proposal,
                id="plan-confirmation",
            )
            self.mount(self.plan_confirmation, before=prompt)
        else:
            self.plan_confirmation.update_plan(
                request.plan_id,
                request.plan_revision,
                request.plan_proposal,
            )
        self.styles.height = self.PLAN_HEIGHT
        # mount 是异步消息；排到当前消息队列末尾再校正焦点，避免刷新回调早于子组件挂载。
        self.app.call_later(self._focus_plan_feedback)
        return self.plan_confirmation

    def close_plan_confirmation(self, *, restore_prompt: bool = True) -> None:
        panel = self.plan_confirmation
        self.plan_confirmation = None
        if panel is not None and panel.is_mounted:
            panel.remove()
        self._prompt().display = restore_prompt
        self._collapse_if_idle()

    def todo_panel(self) -> TodoPanel | None:
        try:
            return self.query_one("#todo-panel", TodoPanel)
        except NoMatches:
            return None

    def on_todo_panel_expanded_changed(self, event: TodoPanel.ExpandedChanged) -> None:
        del event
        self._collapse_if_idle()

    def _focus_plan_feedback(self) -> None:
        if self.plan_confirmation is not None and self.plan_confirmation.is_mounted:
            self.plan_confirmation.feedback_input.focus()

    def open_command_suggestions(self, query: str) -> CommandSuggestionOverlay:
        self.close_file_refs()
        prompt = self._prompt()
        if self.command_overlay is None:
            self.command_overlay = CommandSuggestionOverlay(command_registry().commands())
            self.mount(self.command_overlay, before=prompt)
        self.styles.height = self.EXPANDED_HEIGHT
        self.command_overlay.update_query(query)
        self.call_after_refresh(prompt.focus)
        return self.command_overlay

    def open_file_refs(self, query: str) -> FileReferenceOverlay:
        self.close_command_suggestions()
        prompt = self._prompt()
        if self.file_ref_overlay is None:
            self.file_ref_overlay = FileReferenceOverlay(
                self.workspace_root,
                query,
                index=self.file_reference_index,
            )
            self.mount(self.file_ref_overlay, before=prompt)
        self.styles.height = self.EXPANDED_HEIGHT
        self.file_ref_overlay.update_query(query)
        self.call_after_refresh(prompt.focus)
        return self.file_ref_overlay

    def close_command_suggestions(self) -> None:
        overlay = self.command_overlay
        self.command_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self._collapse_if_idle()

    def close_file_refs(self) -> None:
        overlay = self.file_ref_overlay
        self.file_ref_overlay = None
        if overlay is not None and overlay.is_mounted:
            overlay.remove()
        self._collapse_if_idle()

    def _prompt(self) -> PromptInput:
        return self.query_one("#prompt-input", PromptInput)

    def _collapse_if_idle(self) -> None:
        if (
            self.command_overlay is None
            and self.file_ref_overlay is None
            and self.question_prompt is None
            and self.plan_confirmation is None
        ):
            todo_panel = self.todo_panel()
            todo_height = todo_panel.layout_height() if todo_panel is not None else 0
            self.styles.height = self.COLLAPSED_HEIGHT + todo_height
            if self.is_mounted:
                self._prompt().focus()
