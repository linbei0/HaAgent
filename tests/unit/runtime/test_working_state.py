"""
tests/unit/runtime/test_working_state.py - 精简 WorkingState 测试

验证 WorkingState 只保存有界关键发现，不复制 Todo、目标或工具 trace。
"""

from __future__ import annotations

import pytest

from haagent.runtime.session.turn_completion import ChatTurnResult
from haagent.runtime.session.working_state import (
    WorkingStateError,
    empty_working_state,
    format_working_state_for_model,
    update_working_state,
    working_state_from_dict,
)


def test_working_state_contains_only_key_findings_and_turn(tmp_path) -> None:
    result = ChatTurnResult(
        session_id="session-test",
        turn_index=2,
        status="completed",
        episode_path=tmp_path / "episode",
        provider="fake",
        final_response="确认新的状态边界。",
        verification_status="not_run",
    )

    state = update_working_state(
        empty_working_state(),
        prompt="实现 Plan Mode",
        result=result,
        runtime_events=[
            {"event_type": "tool_finished", "tool_name": "shell", "result": {"stdout": "SECRET"}},
            {"event_type": "assistant_message", "content": "已确认 TaskLedger 是唯一事实源。"},
        ],
    )

    assert set(state.to_dict()) == {"key_findings", "last_updated_turn"}
    assert state.last_updated_turn == 2
    assert "SECRET" not in format_working_state_for_model(state)
    assert all("实现 Plan Mode" not in item for item in state.key_findings)


def test_old_working_state_shape_is_rejected() -> None:
    with pytest.raises(WorkingStateError, match="current schema"):
        working_state_from_dict(
            {
                "current_goal": "旧目标",
                "key_findings": [],
                "completed_actions": [],
                "next_steps": [],
                "last_updated_turn": 1,
            },
        )


def test_working_state_is_bounded() -> None:
    state = working_state_from_dict(
        {
            "key_findings": ["F" * 500 for _ in range(20)],
            "last_updated_turn": 7,
        },
    )

    assert len(state.key_findings) == 5
    assert all(len(item) <= 240 for item in state.key_findings)
