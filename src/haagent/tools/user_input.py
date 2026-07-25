"""
src/haagent/tools/user_input.py - 结构化用户提问参数转换

把已通过 JSON Schema 的工具参数转换为 runtime 强类型问题。
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.human_interaction import UserQuestion, UserQuestionOption


class UserQuestionValidationError(ValueError):
    pass


def parse_user_questions(args: dict[str, Any]) -> tuple[UserQuestion, ...]:
    raw_questions = args.get("questions")
    if not isinstance(raw_questions, list):
        raise UserQuestionValidationError("questions must be an array")
    questions: list[UserQuestion] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            raise UserQuestionValidationError(f"questions[{index}] must be an object")
        question_id = _required_text(raw, "id", index)
        if question_id in seen_ids:
            raise UserQuestionValidationError(f"duplicate question id: {question_id}")
        seen_ids.add(question_id)
        options = tuple(
            UserQuestionOption(
                label=_required_option_text(item, "label", index, option_index),
                description=_option_description(item, index, option_index),
            )
            for option_index, item in enumerate(raw.get("options", []))
        )
        multiple = bool(raw.get("multiple", False))
        if multiple and not options:
            raise UserQuestionValidationError(f"questions[{index}].multiple requires options")
        questions.append(
            UserQuestion(
                id=question_id,
                header=_required_text(raw, "header", index),
                question=_required_text(raw, "question", index),
                options=options,
                multiple=multiple,
                custom=bool(raw.get("custom", True)),
                placeholder=str(raw.get("placeholder", "")).strip(),
            ),
        )
    return tuple(questions)


def _required_text(raw: dict[str, Any], name: str, index: int) -> str:
    value = str(raw.get(name, "")).strip()
    if not value:
        raise UserQuestionValidationError(f"questions[{index}].{name} must not be empty")
    return value


def _required_option_text(raw: object, name: str, question_index: int, option_index: int) -> str:
    if not isinstance(raw, dict):
        raise UserQuestionValidationError(f"questions[{question_index}].options[{option_index}] must be an object")
    value = str(raw.get(name, "")).strip()
    if not value:
        raise UserQuestionValidationError(f"questions[{question_index}].options[{option_index}].{name} must not be empty")
    return value


def _option_description(raw: object, question_index: int, option_index: int) -> str:
    if not isinstance(raw, dict):
        raise UserQuestionValidationError(f"questions[{question_index}].options[{option_index}] must be an object")
    return str(raw.get("description", "")).strip()
