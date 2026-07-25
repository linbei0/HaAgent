"""
src/haagent/tui/widgets/question_prompt.py - InputDock 内联结构化提问组件

在单个可聚焦 Widget 内处理选择、文本编辑、导航和 Review，仅刷新自身。
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widgets import Static

from haagent.runtime.execution.human_interaction import UserQuestion
from haagent.tui.state.question_prompt import PromptAction, QuestionPromptState


class QuestionPrompt(Static):
    """有界结构化提问面板；选项变化不会重建 DOM。"""

    can_focus = True

    class Submitted(Message):
        def __init__(self, answers: dict[str, tuple[str, ...]]) -> None:
            super().__init__()
            self.answers = answers

    class Dismissed(Message):
        pass

    def __init__(self, questions: tuple[UserQuestion, ...], **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = QuestionPromptState(questions)

    def on_key(self, event: events.Key) -> None:
        key = event.key
        question = self.state.current_question
        if key == "ctrl+x":
            return
        if (key in {"?", "question_mark"} or event.character == "?") and not self.state.current_draft:
            self._consume(event)
            self.app.action_help()
            return
        if key == "escape":
            self._consume(event)
            self.post_message(self.Dismissed())
            return
        if key == "tab":
            self._consume(event)
            self.state.next_page()
            self.refresh()
            return
        if key == "shift+tab":
            self._consume(event)
            self.state.previous_page()
            self.refresh()
            return
        if self.state.review:
            if key == "enter":
                self._consume(event)
                self._apply_action(self.state.confirm_review())
            return
        if key in {"up", "down"} and question.options:
            self._consume(event)
            self.state.move_option(-1 if key == "up" else 1)
            self.refresh()
            return
        if question.options and event.character and event.character in "1234":
            if self.state.choose_number(int(event.character)):
                self._consume(event)
                self.refresh()
            return
        if key == "space" and question.multiple and not self.state.current_is_custom:
            self._consume(event)
            self.state.toggle_current()
            self.refresh()
            return
        if key == "ctrl+enter":
            self._consume(event)
            if not question.options or self.state.current_is_custom:
                self.state.append_newline()
                self.refresh()
            return
        if key == "enter":
            self._consume(event)
            self._apply_action(self.state.confirm_current())
            return
        if key == "backspace" and (not question.options or self.state.current_is_custom):
            self._consume(event)
            self.state.backspace()
            self.refresh()
            return
        character = event.character
        if character and character.isprintable() and (not question.options or question.custom):
            self._consume(event)
            self.state.append_text(character)
            self.refresh()

    def render(self) -> Text:
        return self._render_review() if self.state.review else self._render_question()

    def _render_question(self) -> Text:
        state = self.state
        question = state.current_question
        wide = self.app.size.width >= 120
        text = Text()
        text.append(
            f"问题 {state.current_index + 1}/{len(state.questions)} · {question.header}\n",
            style="bold",
        )
        if wide and len(state.questions) > 1:
            labels = [
                f"[{item.header}]" if index == state.current_index else item.header
                for index, item in enumerate(state.questions)
            ]
            text.append(" · ".join(labels) + "\n", style="dim")
        text.append(f"{question.question}\n")
        if question.options:
            selected = state.selections.get(question.id, set())
            for index, option in enumerate(question.options):
                cursor = "›" if state.current_cursor == index else " "
                marker = "[x]" if index in selected else "[ ]"
                line = f"{cursor} {index + 1}. {marker} {option.label}"
                if wide or state.current_cursor == index:
                    line += f" — {option.description}"
                text.append(line + "\n", style="reverse" if state.current_cursor == index else "")
            if question.custom:
                cursor = "›" if state.current_is_custom else " "
                value = state.current_draft or question.placeholder or "输入其他方案"
                text.append(
                    f"{cursor} 其他：{value}{'▌' if state.current_is_custom else ''}\n",
                    style="reverse" if state.current_is_custom else "",
                )
        else:
            value = state.current_draft or question.placeholder or "请输入回答"
            text.append(f"{value}▌\n", style="reverse")
        hints = ["↑↓/数字 选择" if question.options else "Enter 提交", "Esc 关闭", "Ctrl+X 取消任务"]
        if question.multiple:
            hints.insert(1, "Space 多选")
        if len(state.questions) > 1:
            hints.insert(-2, "Tab 切题")
        text.append(" · ".join(hints), style="dim")
        return text

    def _render_review(self) -> Text:
        text = Text("确认回答\n", style="bold")
        for header, answer in self.state.review_rows():
            normalized = " ".join(answer.split())
            if len(normalized) > 100:
                normalized = normalized[:99] + "…"
            text.append(f"{header}：{normalized}\n")
        text.append("Enter 提交 · Shift+Tab 返回 · Esc 关闭 · Ctrl+X 取消任务", style="dim")
        return text

    def _apply_action(self, action: PromptAction) -> None:
        if action == "submit":
            self.post_message(self.Submitted(self.state.answers()))
            return
        self.refresh()

    @staticmethod
    def _consume(event: events.Key) -> None:
        event.stop()
        event.prevent_default()
