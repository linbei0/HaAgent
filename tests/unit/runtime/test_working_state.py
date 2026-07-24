"""
tests/unit/runtime/test_working_state.py - 短期 working_state 测试

验证工作记忆只保存有界摘要，不复制完整工具输出或 episode trace。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.session.turn_completion import ChatTurnResult
from haagent.runtime.session.working_state import (
    WORKING_STATE_MODEL_CHAR_LIMIT,
    WorkingStateError,
    empty_working_state,
    format_working_state_for_model,
    raw_working_state_text,
    update_working_state,
    working_state_from_dict,
)


def test_working_state_update_summarizes_turn_without_full_tool_output(tmp_path) -> None:
    state = empty_working_state()
    secret_output = "SECRET_TOOL_OUTPUT_SHOULD_NOT_ENTER_WORKING_STATE"
    result = ChatTurnResult(
        session_id="session-test",
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episode",
        provider="fake",
        final_response="Inspected README and found the setup command.",
        verification_status="not_run",
    )

    updated = update_working_state(
        state,
        prompt="Inspect the project",
        result=result,
        runtime_events=[
            {
                "event_type": "tool_finished",
                "tool_name": "shell",
                "result": {
                    "status": "success",
                    "exit_code": 0,
                    "stdout": secret_output,
                    "stderr": secret_output,
                },
            },
            {"event_type": "assistant_message", "content": "Read README and summarized setup."},
        ],
    )

    model_text = format_working_state_for_model(updated)
    raw_text = json.dumps(updated.to_dict(), ensure_ascii=False, sort_keys=True)
    assert updated.current_goal == "Inspect the project"
    assert updated.last_updated_turn == 1
    assert updated.completed_actions
    assert updated.key_findings
    assert secret_output not in model_text
    assert secret_output not in raw_text
    assert "assistant_actions:" in model_text
    assert "actor=assistant tool=shell status=success exit_code=0" in model_text


def test_working_state_distinguishes_user_request_from_assistant_tool_actions(tmp_path) -> None:
    state = empty_working_state()
    result = ChatTurnResult(
        session_id="session-test",
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episode",
        provider="fake",
        final_response="Summarized the project.",
        verification_status="not_run",
    )

    updated = update_working_state(
        state,
        prompt="介绍这个项目",
        result=result,
        runtime_events=[
            {
                "event_type": "tool_finished",
                "tool_name": "file_read",
                "result": {
                    "status": "success",
                    "path": "docs/harness-requirements.md",
                },
            },
        ],
    )

    model_text = format_working_state_for_model(updated)

    assert "last_user_request: 介绍这个项目" in model_text
    assert "assistant_actions:" in model_text
    assert "actor=assistant tool=file_read status=success path=docs/harness-requirements.md" in model_text
    assert "user viewed" not in model_text


def test_working_state_does_not_duplicate_final_response(tmp_path) -> None:
    """回归：assistant_message 事件与 final_response 同源时，key_findings 只保留一份。"""
    state = empty_working_state()
    answer = "第一轮完整回答：亚马尔神预言，西班牙夺冠。"
    result = ChatTurnResult(
        session_id="session-test",
        turn_index=1,
        status="completed",
        episode_path=tmp_path / ".runs" / "episode",
        provider="fake",
        final_response=answer,
        verification_status="not_run",
    )

    updated = update_working_state(
        state,
        prompt="搜新闻",
        result=result,
        runtime_events=[
            {"event_type": "assistant_message", "content": answer},
        ],
    )

    occurrences = sum(1 for item in updated.key_findings if "亚马尔" in item)
    assert occurrences == 1, f"expected 1 copy, got {occurrences}: {updated.key_findings}"


def test_working_state_model_text_is_bounded() -> None:
    raw = {
        "current_goal": "G" * 5000,
        "key_findings": ["F" * 5000 for _ in range(20)],
        "completed_actions": ["A" * 5000 for _ in range(20)],
        "next_steps": ["N" * 5000 for _ in range(20)],
        "last_updated_turn": 7,
    }

    state = working_state_from_dict(raw)
    model_text = format_working_state_for_model(state)

    assert len(model_text) <= WORKING_STATE_MODEL_CHAR_LIMIT
    assert "G" * 1000 not in model_text
    assert state.last_updated_turn == 7
    assert len(state.key_findings) <= 5
    assert len(state.completed_actions) <= 5
    assert len(state.next_steps) <= 5
    assert len(raw_working_state_text(raw)) > len(model_text)


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"current_goal": 1, "key_findings": [], "completed_actions": [], "next_steps": [], "last_updated_turn": 0},
        {"current_goal": "", "key_findings": "bad", "completed_actions": [], "next_steps": [], "last_updated_turn": 0},
        {"current_goal": "", "key_findings": [], "completed_actions": [], "next_steps": [], "last_updated_turn": "1"},
    ],
)
def test_working_state_rejects_corrupt_shape(raw: dict[str, object]) -> None:
    with pytest.raises(WorkingStateError):
        working_state_from_dict(raw)
