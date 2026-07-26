"""
tests/unit/context/test_plan_mode_context.py - Plan Mode 工具与上下文测试

验证模型只看到固定白名单，并获得结构化规划约束。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from haagent.context.builder import ContextBuilder
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.session.planning_state import enter_plan_mode, submit_plan_revision
from haagent.runtime.session.turn import write_chat_task_yaml


def test_plan_mode_task_contract_uses_fixed_read_only_tools(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"

    write_chat_task_yaml(
        path,
        "规划功能",
        tmp_path,
        enable_web=False,
        include_session_history=True,
        planning_status="planning",
        allowed_tools_override=["shell", "file_write", "mcp__demo__read"],
    )

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "file_read" in raw["allowed_tools"]
    assert "request_user_input" in raw["allowed_tools"]
    assert "submit_plan" in raw["allowed_tools"]
    assert "todo_update" not in raw["allowed_tools"]
    assert "shell" not in raw["allowed_tools"]
    assert not any(name.startswith("mcp__") for name in raw["allowed_tools"])
    assert raw["policy"]["approval_allowed_tools"] == []


def test_normal_task_contract_exposes_todo_but_not_submit_plan(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    write_chat_task_yaml(path, "执行功能", tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "todo_update" in raw["allowed_tools"]
    assert "submit_plan" not in raw["allowed_tools"]


def test_context_builder_injects_latest_plan_revision(tmp_path: Path) -> None:
    state = submit_plan_revision(
        enter_plan_mode(updated_turn=0),
        {
            "goal": "实现 Plan Mode",
            "summary": "只读调查后提交方案",
            "steps": [{"content": "实现状态机", "completion_condition": "测试通过"}],
            "verification": {"required": True, "description": "运行 pytest"},
            "assumptions": [],
        },
        updated_turn=1,
    )
    writer = _writer(tmp_path)
    context = ContextBuilder(
        task=_task(),
        workspace_root=tmp_path,
        provider_name="test",
        episode_writer=writer,
        planning_state=state.to_dict(),
    ).build()

    assert "Plan Mode：先只读调查" in context.model_input
    assert "latest_plan_revision: 1" in context.model_input
    assert "完成条件：测试通过" in context.model_input


def _writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "source-task.yaml"
    task_path.write_text("goal: test\nworkspace_root: .\n", encoding="utf-8")
    writer = EpisodeWriter.create(tmp_path / ".runs", task_path)
    writer.write_plan(
        {
            "goal": "test",
            "constraints": [],
            "acceptance_criteria": [],
            "verification_commands": [],
            "planned_steps": [],
        },
    )
    return writer


def _task() -> TaskSpec:
    return TaskSpec(
        goal="规划功能",
        workspace_root=".",
        allowed_tools=["file_read", "submit_plan"],
        acceptance_criteria=[],
        verification_commands=[],
        constraints=[],
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )
