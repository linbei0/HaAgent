"""
tests/unit/runtime/test_task_ledger.py - 长任务账本测试

验证 task ledger 保存结构化长任务进度，并向模型只暴露有界摘要。
"""

from __future__ import annotations

import json

import pytest

from haagent.runtime.session.task_ledger import (
    TASK_LEDGER_MODEL_CHAR_LIMIT,
    TaskCheckpoint,
    TaskLedger,
    TaskLedgerError,
    TaskStep,
    empty_task_ledger,
    format_task_ledger_for_model,
    load_task_ledger,
    task_ledger_from_dict,
    update_task_ledger,
    write_task_ledger,
)


def test_empty_task_ledger_records_goal_and_defaults() -> None:
    ledger = empty_task_ledger("增强长任务能力")

    assert ledger.goal == "增强长任务能力"
    assert ledger.status == "planning"
    assert ledger.current_step_id is None
    assert ledger.steps == []
    assert ledger.checkpoints == []
    assert ledger.updated_turn == 0
    assert ledger.is_empty() is False


def test_task_ledger_round_trips_with_worker_step_and_checkpoint(tmp_path) -> None:
    path = tmp_path / "task-ledger.json"
    ledger = TaskLedger(
        goal="完成长任务恢复",
        status="running",
        current_step_id="step-002",
        steps=[
            TaskStep(
                id="step-001",
                title="建立任务账本",
                kind="plan",
                owner="main",
                status="completed",
                evidence_refs=["episode=episode-1", "tool=tool-1"],
                checkpoint_ids=["ckpt-001"],
                updated_turn=1,
            ),
            TaskStep(
                id="step-002",
                title="分析 worker 结果",
                kind="delegate",
                owner="worker",
                worker_id="worker-a",
                parent_step_id="step-001",
                status="running",
                evidence_refs=["worker_notification=worker-a:1"],
                updated_turn=2,
            ),
        ],
        checkpoints=[
            TaskCheckpoint(
                id="ckpt-001",
                step_id="step-001",
                turn_index=1,
                episode_path=".runs/episodes/episode-1",
                tool_call_ids=["tool-1"],
                changed_paths=["src/haagent/runtime/session/task_ledger.py"],
                verification_refs=["pytest tests/unit/runtime/test_task_ledger.py -q"],
                state_digest="sha256:abc",
                created_at="2026-07-07T00:00:00Z",
            ),
        ],
        budgets={"max_turns": 12, "tool_calls": 4},
        updated_turn=2,
    )

    write_task_ledger(path, ledger)
    loaded = load_task_ledger(path)

    assert loaded.to_dict() == ledger.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["current_step_id"] == "step-002"


def test_task_ledger_model_text_is_bounded_and_omits_full_evidence() -> None:
    secret = "SECRET_FULL_TOOL_OUTPUT_SHOULD_NOT_ENTER_MODEL"
    ledger = TaskLedger(
        goal="G" * 5000,
        status="blocked",
        current_step_id="step-009",
        steps=[
            TaskStep(
                id=f"step-{i:03d}",
                title=("分析长任务状态 " + str(i)) * 100,
                kind="research",
                owner="main",
                status="completed" if i < 9 else "blocked",
                evidence_refs=[secret, "episode=.runs/episodes/very-long"],
                blocker={
                    "category": "tool_timeout",
                    "reason": secret,
                    "suggested_action": "retry_with_narrower_command",
                },
                updated_turn=i,
            )
            for i in range(10)
        ],
        checkpoints=[],
        budgets={"warning": secret, "tool_calls": 99},
        updated_turn=10,
    )

    model_text = format_task_ledger_for_model(ledger)
    raw_text = json.dumps(ledger.to_dict(), ensure_ascii=False)

    assert len(model_text) <= TASK_LEDGER_MODEL_CHAR_LIMIT
    assert secret in raw_text
    assert secret not in model_text
    assert "task_goal:" in model_text
    assert "active_step:" in model_text
    assert "completed_steps:" in model_text
    assert "blocker:" in model_text
    assert "suggested_action=retry_with_narrower_command" in model_text


def test_recovery_and_checkpoint_events_update_active_step(tmp_path) -> None:
    ledger = update_task_ledger(
        empty_task_ledger(),
        prompt="修复长任务失败恢复",
        turn_index=1,
        result_status="failed",
        episode_path=tmp_path / ".runs" / "episode-failed",
        runtime_events=[
            {
                "event_type": "task_checkpoint_saved",
                "step_id": "step-001",
                "title": "修复长任务失败恢复",
                "status": "failed",
                "evidence_count": 1,
                "checkpoint_count": 0,
            },
            {
                "event_type": "task_recovery_suggested",
                "step_id": "step-001",
                "title": "修复长任务失败恢复",
                "category": "verification_failed",
                "summary": "verification_failed: reason_chars=4000",
                "suggested_action": "repair_and_rerun_verification",
                "reason_chars": 4000,
            },
        ],
    )

    step = ledger.steps[0]
    assert ledger.status == "blocked"
    assert ledger.current_step_id == "step-001"
    assert step.status == "blocked"
    assert step.blocker == {
        "category": "verification_failed",
        "reason": "reason_chars=4000",
        "suggested_action": "repair_and_rerun_verification",
    }
    assert step.checkpoint_ids
    assert step.checkpoint_ids[-1] == ledger.checkpoints[-1].id
    assert any(ref.startswith("checkpoint=") for ref in step.evidence_refs)


def test_model_retry_events_accumulate_model_attempt_budget(tmp_path) -> None:
    ledger = update_task_ledger(
        empty_task_ledger(),
        prompt="测试模型重试预算",
        turn_index=1,
        result_status="failed",
        episode_path=tmp_path / ".runs" / "episode-failed",
        runtime_events=[
            {"event_type": "model_retry_scheduled", "turn": 1, "attempt": 1, "next_attempt": 2},
            {"event_type": "model_retry_scheduled", "turn": 1, "attempt": 2, "next_attempt": 3},
            {"event_type": "model_retry_exhausted", "turn": 1, "attempt": 3},
        ],
    )

    assert ledger.budgets["model_attempts"] == 3


def test_model_retry_budget_counts_initial_attempt_and_scheduled_replay(tmp_path) -> None:
    ledger = update_task_ledger(
        empty_task_ledger(),
        prompt="测试模型重试预算",
        turn_index=1,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episode-completed",
        runtime_events=[
            {"event_type": "task_step_progress", "category": "model_turn_started"},
            {"event_type": "model_retry_scheduled", "turn": 1, "attempt": 1, "next_attempt": 2},
        ],
    )

    assert ledger.budgets["model_attempts"] == 2


def test_blocked_ledger_resumes_same_active_step(tmp_path) -> None:
    blocked = update_task_ledger(
        empty_task_ledger(),
        prompt="继续实现长任务恢复",
        turn_index=1,
        result_status="failed",
        episode_path=tmp_path / ".runs" / "episode-failed",
        runtime_events=[
            {
                "event_type": "task_recovery_suggested",
                "step_id": "step-001",
                "title": "继续实现长任务恢复",
                "category": "tool_failure",
                "suggested_action": "inspect_failure_and_replan",
                "reason_chars": 80,
            },
        ],
    )

    resumed = update_task_ledger(
        blocked,
        prompt="按恢复建议继续",
        turn_index=2,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episode-completed",
        runtime_events=[],
    )

    assert len(resumed.steps) == 1
    assert resumed.current_step_id == "step-001"
    assert resumed.steps[0].title == "继续实现长任务恢复"
    assert resumed.steps[0].status == "completed"


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"schema_version": 2, "goal": "", "status": "planning", "current_step_id": None, "steps": [], "checkpoints": [], "budgets": {}, "updated_turn": 0},
        {"schema_version": 1, "goal": 1, "status": "planning", "current_step_id": None, "steps": [], "checkpoints": [], "budgets": {}, "updated_turn": 0},
        {"schema_version": 1, "goal": "", "status": "unknown", "current_step_id": None, "steps": [], "checkpoints": [], "budgets": {}, "updated_turn": 0},
        {"schema_version": 1, "goal": "", "status": "planning", "current_step_id": None, "steps": "bad", "checkpoints": [], "budgets": {}, "updated_turn": 0},
        {"schema_version": 1, "goal": "", "status": "planning", "current_step_id": None, "steps": [{"id": "step-001", "title": "bad", "kind": "plan", "owner": "alien", "status": "pending", "evidence_refs": [], "checkpoint_ids": [], "blocker": None, "retry_count": 0, "updated_turn": 0}], "checkpoints": [], "budgets": {}, "updated_turn": 0},
    ],
)
def test_task_ledger_rejects_corrupt_shape(raw: dict[str, object]) -> None:
    with pytest.raises(TaskLedgerError):
        task_ledger_from_dict(raw)


def test_update_task_ledger_tracks_progress_and_worker_failure(tmp_path) -> None:
    ledger = empty_task_ledger()
    updated = update_task_ledger(
        ledger,
        prompt="增强项目长任务能力",
        turn_index=1,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episodes" / "episode-1",
        runtime_events=[
            {
                "event_type": "tool_finished",
                "tool_name": "file_read",
                "result": {"status": "success", "path": "src/haagent/runtime/session/agent.py"},
            },
            {
                "event_type": "task_step_finished",
                "step_id": "step-001",
                "title": "建立任务账本",
                "owner": "main",
                "evidence_count": 1,
                "checkpoint_count": 0,
            },
            {
                "event_type": "worker_failed",
                "agent_id": "worker-a",
                "task_id": "task-1",
                "reason": "worker failed",
            },
        ],
    )

    assert updated.goal == "增强项目长任务能力"
    assert updated.status == "blocked"
    assert updated.current_step_id == "worker-a"
    assert updated.steps[0].status == "completed"
    worker_step = updated.steps[-1]
    assert worker_step.id == "worker-a"
    assert worker_step.owner == "worker"
    assert worker_step.status == "blocked"
    assert worker_step.blocker["category"] == "worker_failure"
    assert updated.checkpoints
    assert updated.checkpoints[-1].episode_path.endswith("episode-1")


def test_completed_turn_closes_running_task_ledger_step(tmp_path) -> None:
    ledger = update_task_ledger(
        empty_task_ledger(),
        prompt="修复长任务状态同步",
        turn_index=1,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episode-completed",
        runtime_events=[
            {
                "event_type": "tool_finished",
                "tool_name": "shell",
                "args": {"command": "uv run pytest tests/unit/runtime/test_task_ledger.py -q"},
                "result": {"status": "success", "exit_code": 0},
            },
        ],
    )

    assert ledger.goal == "修复长任务状态同步"
    assert ledger.status == "completed"
    assert ledger.current_step_id == "step-001"
    assert ledger.steps[0].status == "completed"
    assert ledger.steps[0].title == "修复长任务状态同步"
    assert ledger.steps[0].evidence_refs[-1].endswith("episode-completed")


def test_terminal_ledger_starts_new_goal_on_next_turn(tmp_path) -> None:
    previous = update_task_ledger(
        empty_task_ledger(),
        prompt="yy",
        turn_index=1,
        result_status="cancelled",
        episode_path=tmp_path / ".runs" / "episode-cancelled",
        runtime_events=[],
    )

    next_turn = update_task_ledger(
        previous,
        prompt="正式修复 inspect warning",
        turn_index=2,
        result_status="completed",
        episode_path=tmp_path / ".runs" / "episode-2",
        runtime_events=[],
    )

    assert previous.status == "cancelled"
    assert next_turn.goal == "正式修复 inspect warning"
    assert next_turn.status == "completed"
    assert next_turn.current_step_id == "step-001"
    assert len(next_turn.steps) == 1
    assert next_turn.steps[0].title == "正式修复 inspect warning"
    assert next_turn.steps[0].status == "completed"
