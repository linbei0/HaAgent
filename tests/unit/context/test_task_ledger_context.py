"""
tests/unit/context/test_task_ledger_context.py - Todo 上下文选择测试

验证活动 Todo 完整注入，终态清单确定性跳过。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.session.task_ledger import TaskLedger, TodoItem


def test_context_builder_injects_active_todos_without_evidence(tmp_path: Path) -> None:
    secret = "SECRET_FULL_EVIDENCE_SHOULD_STAY_ON_DISK"
    writer = _make_writer(tmp_path)
    ledger = TaskLedger(
        goal="实现 Plan Mode",
        todos=[
            TodoItem(id="state", content="实现状态机", status="completed", evidence_refs=[secret]),
            TodoItem(id="tui", content="接入确认面板", status="in_progress", evidence_refs=[secret]),
        ],
        updated_turn=2,
    )

    context = ContextBuilder(
        task=_task("继续开发"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        task_ledger=ledger.to_dict(),
    ).build()
    manifest = json.loads(
        (writer.path / "contexts" / f"{context.context_id}-manifest.json").read_text(encoding="utf-8")
    )

    assert "todo_goal: 实现 Plan Mode" in context.model_input
    assert "id=state status=completed content=实现状态机" in context.model_input
    assert "id=tui status=in_progress content=接入确认面板" in context.model_input
    assert secret not in context.model_input
    assert any(item["source_id"] == "task_ledger" for item in manifest["selection"]["selected"])


def test_context_builder_skips_all_terminal_todos(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    ledger = TaskLedger(
        goal="目标",
        todos=[TodoItem(id="done", content="完成", status="completed")],
    )

    context = ContextBuilder(
        task=_task("新问题"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        task_ledger=ledger.to_dict(),
    ).build()

    assert "Todo" not in context.model_input
    assert "todo_goal" not in context.model_input


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
