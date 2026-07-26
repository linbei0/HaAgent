"""
tests/unit/multi_agent/test_task_ledger_integration.py - Worker 与主 Todo 边界测试

验证 Worker 结果只形成运行时元数据，不替主 Agent 改变 Todo 状态。
"""

from __future__ import annotations

from haagent.runtime.session.task_ledger import TaskLedger, TodoItem, update_task_ledger_runtime


def test_worker_failure_blocks_active_todo_without_changing_status(tmp_path) -> None:
    ledger = TaskLedger(
        goal="并行调查",
        todos=[TodoItem(id="research", content="汇总 worker 调查", status="in_progress")],
    )

    updated = update_task_ledger_runtime(
        ledger,
        turn_index=2,
        episode_path=tmp_path / "episode",
        runtime_events=[
            {
                "event_type": "worker_failed",
                "agent_id": "worker-1",
                "category": "worker_failure",
                "reason": "读取失败",
            },
        ],
    )

    assert updated.active_todo().status == "in_progress"
    assert updated.active_todo().blocker["category"] == "worker_failure"


def test_worker_completion_does_not_complete_main_todo(tmp_path) -> None:
    ledger = TaskLedger(
        goal="并行调查",
        todos=[TodoItem(id="research", content="汇总 worker 调查", status="in_progress")],
    )

    updated = update_task_ledger_runtime(
        ledger,
        turn_index=2,
        episode_path=tmp_path / "episode",
        runtime_events=[{"event_type": "worker_completed", "agent_id": "worker-1"}],
    )

    assert updated.active_todo().status == "in_progress"
