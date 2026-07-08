"""
tests/unit/multi_agent/test_task_ledger_integration.py - 多智能体长任务账本集成测试

验证 worker 通知与 runtime 事件能进入同一个 TaskLedger，而不是形成独立进度状态。
"""

from __future__ import annotations

from types import SimpleNamespace

from haagent.multi_agent.messages import WorkerNotification
from haagent.multi_agent.runtime import MultiAgentRuntime, _WorkerTask
from haagent.runtime.session.task_ledger import TaskStep, empty_task_ledger, update_task_ledger


def test_worker_notification_carries_parent_step_and_bounded_evidence_refs() -> None:
    notification = WorkerNotification(
        event_type="worker_status",
        team_id="team-1",
        agent_id="verification-1",
        task_id="task-123",
        status="completed",
        summary="pytest 验证完成",
        result_excerpt="FULL WORKER OUTPUT MUST STAY OUT",
        episode_path=".runs/episodes/worker-episode",
        error="",
        needs_attention=False,
        parent_step_id="step-002",
        evidence_refs=("episode=.runs/episodes/worker-episode", "worker=verification-1"),
    ).to_dict()

    assert notification["parent_step_id"] == "step-002"
    assert notification["evidence_refs"] == ("episode=.runs/episodes/worker-episode", "worker=verification-1")
    assert notification["result_excerpt"] == "FULL WORKER OUTPUT MUST STAY OUT"


def test_worker_events_roll_up_to_parent_task_step(tmp_path) -> None:
    ledger = empty_task_ledger("增强长任务能力")
    ledger.steps.append(
        TaskStep(
            id="step-002",
            title="委派验证 worker",
            kind="delegate",
            owner="main",
            status="running",
            updated_turn=1,
        )
    )
    ledger.current_step_id = "step-002"
    updated = update_task_ledger(
        ledger,
        prompt="继续",
        turn_index=2,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-2",
        runtime_events=[
            {
                "event_type": "worker_completed",
                "agent_id": "verification-1",
                "task_id": "task-123",
                "description": "运行回归测试",
                "parent_step_id": "step-002",
                "evidence_refs": ["episode=.runs/episodes/worker-episode", "worker=verification-1"],
            }
        ],
    )

    parent = next(step for step in updated.steps if step.id == "step-002")
    worker = next(step for step in updated.steps if step.id == "verification-1")
    assert parent.status == "completed"
    assert parent.kind == "delegate"
    assert "episode=.runs/episodes/worker-episode" in parent.evidence_refs
    assert worker.owner == "worker"
    assert worker.status == "completed"
    assert worker.worker_id == "verification-1"
    assert updated.current_step_id == "step-002"


def test_worker_failure_blocks_parent_task_step(tmp_path) -> None:
    ledger = empty_task_ledger("增强长任务能力")
    ledger.steps.append(
        TaskStep(
            id="step-002",
            title="委派验证 worker",
            kind="delegate",
            owner="main",
            status="running",
            updated_turn=1,
        )
    )
    ledger.current_step_id = "step-002"
    updated = update_task_ledger(
        ledger,
        prompt="继续",
        turn_index=2,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-2",
        runtime_events=[
            {
                "event_type": "worker_failed",
                "agent_id": "verification-1",
                "task_id": "task-123",
                "description": "运行回归测试",
                "parent_step_id": "step-002",
                "reason": "pytest failed",
            }
        ],
    )

    parent = next(step for step in updated.steps if step.id == "step-002")
    worker = next(step for step in updated.steps if step.id == "verification-1")
    assert updated.status == "blocked"
    assert parent.status == "blocked"
    assert parent.blocker["category"] == "worker_failure"
    assert parent.blocker["suggested_action"] == "retry_worker_or_take_over"
    assert worker.status == "blocked"
    assert worker.blocker["category"] == "worker_failure"
    assert worker.blocker["suggested_action"] == "retry_worker_or_take_over"


def test_worker_failed_event_emits_task_recovery_suggestion() -> None:
    events: list[dict[str, object]] = []
    runtime = SimpleNamespace(event_sink=events.append)
    worker = _WorkerTask(
        agent_id="verification-1",
        task_id="task-123",
        team_id="team-1",
        session=None,
        parent_step_id="step-002",
    )

    MultiAgentRuntime._emit_worker_event(
        runtime,
        "worker_failed",
        worker,
        status="failed",
        subagent_type="verifier",
        description="运行回归测试",
        episode_path=".runs/episodes/worker-episode",
    )

    assert [event["event_type"] for event in events] == [
        "worker_failed",
        "task_recovery_suggested",
    ]
    recovery = events[-1]
    assert recovery["step_id"] == "step-002"
    assert recovery["category"] == "worker_failure"
    assert recovery["suggested_action"] == "retry_worker_or_take_over"
