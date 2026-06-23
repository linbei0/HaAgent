"""
tests/test_working_state.py - 短期 working_state 测试

验证工作记忆只保存有界摘要，不复制完整工具输出或 episode trace。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.chat_session import ChatTurnResult
from haagent.runtime.working_state import (
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
    assert "tool shell status=success exit_code=0" in model_text


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
