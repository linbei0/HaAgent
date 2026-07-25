"""
tests/unit/tui/test_question_prompt_state.py - 结构化提问纯状态模型测试

验证选项、文本、多题导航和 Review 行为不依赖 Textual。
"""

from __future__ import annotations

from haagent.runtime.execution.human_interaction import UserQuestion, UserQuestionOption
from haagent.tui.state.question_prompt import QuestionPromptState


def _choice(*, multiple: bool = False, custom: bool = True) -> UserQuestion:
    return UserQuestion(
        id="storage",
        header="存储方式",
        question="请选择存储方式。",
        options=(
            UserQuestionOption(label="SQLite（推荐）", description="零配置。"),
            UserQuestionOption(label="JSON 文件", description="容易查看。"),
        ),
        multiple=multiple,
        custom=custom,
        placeholder="输入其他方案",
    )


def test_single_choice_confirms_and_submits_one_question() -> None:
    state = QuestionPromptState((_choice(),))

    state.move_option(1)

    assert state.confirm_current() == "submit"
    assert state.answers() == {"storage": ("JSON 文件",)}


def test_multiple_choice_toggles_and_reaches_review_for_multiple_questions() -> None:
    state = QuestionPromptState(
        (
            _choice(multiple=True),
            UserQuestion(id="note", header="说明", question="补充说明。"),
        )
    )

    state.toggle_current()
    state.move_option(1)
    state.toggle_current()
    assert state.confirm_current() == "navigate"
    state.append_text("先使用本地存储")
    assert state.confirm_current() == "review"

    assert state.review is True
    assert state.confirm_review() == "submit"
    assert state.answers() == {
        "storage": ("SQLite（推荐）", "JSON 文件"),
        "note": ("先使用本地存储",),
    }


def test_custom_draft_and_multiline_text_survive_navigation() -> None:
    state = QuestionPromptState(
        (
            _choice(),
            UserQuestion(id="details", header="细节", question="补充细节。"),
        )
    )

    state.move_option(2)
    state.append_text("自定义数据库")
    assert state.next_page() == "navigate"
    state.append_text("第一行")
    state.append_newline()
    state.append_text("/第二行")
    assert state.previous_page() == "navigate"
    assert state.current_draft == "自定义数据库"
    assert state.next_page() == "navigate"
    assert state.current_draft == "第一行\n/第二行"


def test_review_enter_jumps_to_first_unanswered_question() -> None:
    state = QuestionPromptState(
        (
            UserQuestion(id="first", header="一", question="第一题。"),
            UserQuestion(id="second", header="二", question="第二题。"),
        )
    )

    state.next_page()
    state.next_page()

    assert state.review is True
    assert state.confirm_review() == "navigate"
    assert state.review is False
    assert state.current_index == 0
