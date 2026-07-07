"""
tests/unit/context/test_task_ledger_context.py - 长任务账本上下文测试

验证 ContextBuilder 只注入 task ledger 的有界当前状态。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.session.task_ledger import TaskLedger, TaskStep


def test_context_builder_injects_bounded_task_ledger(tmp_path: Path) -> None:
    secret = "SECRET_FULL_EVIDENCE_SHOULD_STAY_ON_DISK"
    writer = _make_writer(tmp_path)
    ledger = TaskLedger(
        goal="增强长任务能力",
        status="blocked",
        current_step_id="step-002",
        steps=[
            TaskStep(
                id="step-001",
                title="建立任务账本",
                kind="plan",
                owner="main",
                status="completed",
                evidence_refs=[secret],
                checkpoint_ids=["ckpt-001"],
                updated_turn=1,
            ),
            TaskStep(
                id="step-002",
                title="恢复 worker 子任务",
                kind="delegate",
                owner="worker",
                worker_id="worker-a",
                status="blocked",
                evidence_refs=[secret],
                blocker={"category": "worker_failure", "reason": secret},
                retry_count=2,
                updated_turn=2,
            ),
        ],
        checkpoints=[],
        budgets={"tool_calls": 8, "warning": secret},
        updated_turn=2,
    )

    context = ContextBuilder(
        task=_task("继续长任务"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        task_ledger=ledger.to_dict(),
    ).build()

    model_input = context.model_input
    manifest = json.loads(
        (writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8")
    )

    assert "task_goal: 增强长任务能力" in model_input
    assert "active_step: id=step-002 status=blocked owner=worker" in model_input
    assert "active_worker: worker-a" in model_input
    assert "blocker: category=worker_failure" in model_input
    assert "SECRET_FULL_EVIDENCE" not in model_input
    assert any(item["source_id"] == "task_ledger" for item in manifest["selection"]["selected"])


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\nworkspace_root: .\n", encoding="utf-8")
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)
    writer.write_plan(
        {
            "goal": "test",
            "constraints": [],
            "acceptance_criteria": [],
            "verification_commands": [],
            "planned_steps": ["Use allowed tools."],
        },
    )
    return writer


def _task(goal: str) -> TaskSpec:
    return TaskSpec(
        goal=goal,
        workspace_root=".",
        allowed_tools=["file_read"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )

