"""
tests/unit/runtime/test_plan_session_flow.py - AgentSession Plan/Todo 协调测试

验证进入条件、持久化、批准后确定性初始化和整任务取消。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from haagent.runtime.session.agent import AgentSession
from haagent.runtime.session.package import ChatSessionError
from haagent.runtime.session.task_ledger import TodoItem, replace_todos


def _proposal() -> dict[str, object]:
    return {
        "goal": "实现 Plan Mode",
        "summary": "状态与 TUI 闭环",
        "steps": [
            {"content": "实现状态机", "completion_condition": "单测通过"},
            {"content": "接入 TUI", "completion_condition": "交互测试通过"},
        ],
        "verification": {"required": True, "description": "运行完整 pytest"},
        "assumptions": [],
    }


def test_session_enters_and_resumes_plan_mode(tmp_path) -> None:
    session = AgentSession(workspace_root=tmp_path, runs_root=tmp_path / ".runs")
    state = session.enter_plan_mode()

    resumed = AgentSession.resume(session.session_path)

    assert state.status == "planning"
    assert resumed.snapshot.planning_state.plan_id == state.plan_id
    assert resumed.snapshot.planning_state.is_plan_mode is True


def test_active_todo_prevents_entering_plan_mode(tmp_path) -> None:
    session = AgentSession(workspace_root=tmp_path, runs_root=tmp_path / ".runs")
    session._task_ledger = replace_todos(
        session._task_ledger,
        items=[TodoItem(id="a", content="当前任务", status="in_progress")],
        turn_index=1,
        goal="当前任务",
    )

    with pytest.raises(ChatSessionError, match="Todo 尚未结束"):
        session.enter_plan_mode()


def test_approved_plan_initializes_todo_once_and_starts_execution(tmp_path, monkeypatch) -> None:
    session = AgentSession(workspace_root=tmp_path, runs_root=tmp_path / ".runs")
    session.enter_plan_mode()
    submitted = session._handle_planning_state_action("submit", _proposal(), 1)
    session._handle_planning_state_action(
        "approve",
        {"plan_id": submitted["plan_id"], "revision": submitted["revision"]},
        1,
    )
    calls: list[str] = []

    def fake_run(prompt, **kwargs):
        del kwargs
        calls.append(prompt)
        return SimpleNamespace(status="completed")

    monkeypatch.setattr(session, "run_prompt_events", fake_run)
    result = session._run_approved_plan_execution(event_sink=None, interaction_handler=None)

    assert result.status == "completed"
    assert calls == ["实现 Plan Mode"]
    assert [item.status for item in session.snapshot.task_ledger.todos] == [
        "in_progress",
        "pending",
        "pending",
    ]
    assert session.snapshot.planning_state.status == "execution_started"
    assert len({item.id for item in session.snapshot.task_ledger.todos}) == 3


def test_cancel_marks_plan_and_active_todos_terminal(tmp_path) -> None:
    session = AgentSession(workspace_root=tmp_path, runs_root=tmp_path / ".runs")
    session.enter_plan_mode()
    session._task_ledger = replace_todos(
        session._task_ledger,
        items=[TodoItem(id="a", content="当前任务", status="in_progress")],
        turn_index=1,
        goal="当前任务",
    )

    assert session.cancel_current_run() is True
    assert session.snapshot.planning_state.status == "cancelled"
    assert session.snapshot.task_ledger.todos[0].status == "cancelled"
