"""
tests/test_task_loading.py - task.yaml 加载测试

验证任务规格字段能被读取，并拒绝缺少必填字段的输入。
"""

from pathlib import Path

import pytest

from agent_foundry.task import TaskLoadError, TaskSpec, load_task


def write_task(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_task_reads_required_fields(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Build the smallest runtime
constraints:
  - No real LLM
allowed_tools:
  - fake_tool
acceptance_criteria:
  - State flow is recorded
verification_commands:
  - uv run pytest
""".strip(),
    )

    task = load_task(task_path)

    assert task == TaskSpec(
        goal="Build the smallest runtime",
        constraints=["No real LLM"],
        allowed_tools=["fake_tool"],
        acceptance_criteria=["State flow is recorded"],
        verification_commands=["uv run pytest"],
    )


def test_load_task_rejects_missing_required_field(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Missing allowed tools
constraints: []
acceptance_criteria: []
verification_commands: []
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="allowed_tools"):
        load_task(task_path)
