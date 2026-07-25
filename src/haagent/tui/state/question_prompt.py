"""
src/haagent/tui/state/question_prompt.py - 结构化用户提问状态模型

保存问题导航、选项选择、文本草稿和 Review 状态，不依赖 Textual。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from haagent.runtime.execution.human_interaction import UserQuestion


PromptAction = Literal["stay", "navigate", "review", "submit"]


@dataclass
class QuestionPromptState:
    questions: tuple[UserQuestion, ...]
    current_index: int = 0
    review: bool = False
    option_cursors: list[int] = field(init=False)
    selections: dict[str, set[int]] = field(default_factory=dict)
    drafts: dict[str, str] = field(default_factory=dict)
    custom_selected: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("questions must not be empty")
        self.option_cursors = [0 for _ in self.questions]

    @property
    def current_question(self) -> UserQuestion:
        return self.questions[self.current_index]

    @property
    def current_cursor(self) -> int:
        return self.option_cursors[self.current_index]

    @property
    def current_draft(self) -> str:
        return self.drafts.get(self.current_question.id, "")

    @property
    def current_is_custom(self) -> bool:
        question = self.current_question
        return bool(question.options and question.custom and self.current_cursor == len(question.options))

    def move_option(self, delta: int) -> None:
        question = self.current_question
        if not question.options:
            return
        count = len(question.options) + int(question.custom)
        self.option_cursors[self.current_index] = (self.current_cursor + delta) % count

    def choose_number(self, number: int) -> bool:
        question = self.current_question
        if not question.options or number < 1 or number > len(question.options):
            return False
        self.option_cursors[self.current_index] = number - 1
        return True

    def toggle_current(self) -> None:
        question = self.current_question
        cursor = self.current_cursor
        if not question.multiple or cursor >= len(question.options):
            return
        selected = self.selections.setdefault(question.id, set())
        if cursor in selected:
            selected.remove(cursor)
        else:
            selected.add(cursor)
        self.custom_selected.discard(question.id)

    def append_text(self, text: str) -> None:
        question = self.current_question
        if question.options:
            if not question.custom:
                return
            self.option_cursors[self.current_index] = len(question.options)
            self.selections.pop(question.id, None)
            self.custom_selected.add(question.id)
        self.drafts[question.id] = self.current_draft + text

    def append_newline(self) -> None:
        self.append_text("\n")

    def backspace(self) -> None:
        question_id = self.current_question.id
        draft = self.current_draft
        if draft:
            self.drafts[question_id] = draft[:-1]

    def confirm_current(self) -> PromptAction:
        question = self.current_question
        if question.options:
            cursor = self.current_cursor
            if cursor < len(question.options):
                if question.multiple:
                    selected = self.selections.setdefault(question.id, set())
                    if not selected:
                        selected.add(cursor)
                else:
                    self.selections[question.id] = {cursor}
                self.custom_selected.discard(question.id)
            elif self.current_draft.strip():
                self.selections.pop(question.id, None)
                self.custom_selected.add(question.id)
        if not self.is_answered(self.current_index):
            return "stay"
        if len(self.questions) == 1:
            return "submit"
        if self.current_index < len(self.questions) - 1:
            self.current_index += 1
            return "navigate"
        self.review = True
        return "review"

    def next_page(self) -> PromptAction:
        if self.review:
            return "stay"
        if self.current_index < len(self.questions) - 1:
            self.current_index += 1
            return "navigate"
        self.review = True
        return "review"

    def previous_page(self) -> PromptAction:
        if self.review:
            self.review = False
            self.current_index = len(self.questions) - 1
            return "navigate"
        if self.current_index > 0:
            self.current_index -= 1
            return "navigate"
        return "stay"

    def confirm_review(self) -> PromptAction:
        for index in range(len(self.questions)):
            if not self.is_answered(index):
                self.current_index = index
                self.review = False
                return "navigate"
        return "submit"

    def is_answered(self, index: int) -> bool:
        question = self.questions[index]
        if not question.options:
            return bool(self.drafts.get(question.id, "").strip())
        if question.id in self.custom_selected:
            return bool(self.drafts.get(question.id, "").strip())
        return bool(self.selections.get(question.id))

    def answers(self) -> dict[str, tuple[str, ...]]:
        answers: dict[str, tuple[str, ...]] = {}
        for question in self.questions:
            if question.id in self.custom_selected or not question.options:
                draft = self.drafts.get(question.id, "")
                if draft.strip():
                    answers[question.id] = (draft,)
                continue
            selected = sorted(self.selections.get(question.id, set()))
            if selected:
                answers[question.id] = tuple(question.options[index].label for index in selected)
        return answers

    def review_rows(self) -> tuple[tuple[str, str], ...]:
        answers = self.answers()
        return tuple(
            (question.header, "、".join(answers.get(question.id, ())) or "未回答")
            for question in self.questions
        )
