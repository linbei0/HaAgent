"""
tests/unit/runtime/test_planning_state.py - Plan Mode 状态机测试

验证 revision、迟到操作、批准执行和严格持久化边界。
"""

from __future__ import annotations

import pytest

from haagent.runtime.session.planning_state import (
    PlanningStateError,
    approve_plan,
    cancel_plan,
    empty_planning_state,
    enter_plan_mode,
    mark_plan_execution_started,
    planning_state_from_dict,
    request_plan_revision,
    submit_plan_revision,
)


def _proposal() -> dict[str, object]:
    return {
        "goal": "实现 Plan Mode",
        "summary": "收敛状态并接入 TUI。",
        "steps": [
            {"content": "实现状态机", "completion_condition": "状态测试通过"},
            {"content": "实现确认面板", "completion_condition": "键盘路径通过"},
        ],
        "verification": {"required": True, "description": "运行完整 pytest"},
        "assumptions": ["不兼容开发期旧 schema"],
    }


def test_plan_revision_uses_runtime_ids_and_increments() -> None:
    state = enter_plan_mode(updated_turn=0)
    first = submit_plan_revision(state, _proposal(), updated_turn=1)
    revising = request_plan_revision(first, plan_id=first.plan_id, revision=1, updated_turn=1)
    second = submit_plan_revision(revising, _proposal(), updated_turn=1)

    assert first.revision == 1
    assert second.revision == 2
    assert first.proposal.steps[0].id != second.proposal.steps[0].id
    assert first.status == "awaiting_confirmation"


def test_stale_revision_cannot_be_approved_or_revised() -> None:
    state = submit_plan_revision(enter_plan_mode(updated_turn=0), _proposal(), updated_turn=1)

    with pytest.raises(PlanningStateError, match="stale"):
        approve_plan(state, plan_id=state.plan_id, revision=0, updated_turn=1)
    with pytest.raises(PlanningStateError, match="stale"):
        request_plan_revision(state, plan_id="plan-stale", revision=1, updated_turn=1)


def test_approved_plan_transitions_to_execution_started() -> None:
    submitted = submit_plan_revision(enter_plan_mode(updated_turn=0), _proposal(), updated_turn=1)
    approved = approve_plan(submitted, plan_id=submitted.plan_id, revision=1, updated_turn=1)
    started = mark_plan_execution_started(approved, execution_id=approved.execution_id, updated_turn=2)

    assert approved.status == "approved_pending_execution"
    assert started.status == "execution_started"
    assert started.execution_started_revision == 1
    assert started.is_plan_mode is False


def test_plan_item_limit_includes_verification() -> None:
    proposal = _proposal()
    proposal["steps"] = [
        {"content": f"步骤 {index}", "completion_condition": "完成"}
        for index in range(20)
    ]
    with pytest.raises(PlanningStateError, match="20 Todo"):
        submit_plan_revision(enter_plan_mode(updated_turn=0), proposal, updated_turn=1)


def test_planning_state_round_trip_and_cancel() -> None:
    submitted = submit_plan_revision(enter_plan_mode(updated_turn=0), _proposal(), updated_turn=1)
    restored = planning_state_from_dict(submitted.to_dict())
    cancelled = cancel_plan(restored, updated_turn=2)

    assert restored.proposal.goal == "实现 Plan Mode"
    assert cancelled.status == "cancelled"
    assert empty_planning_state().status == "inactive"
