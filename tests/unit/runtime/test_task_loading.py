"""
tests/unit/runtime/test_task_loading.py - task.yaml 加载测试

验证任务规格字段能被读取，并拒绝缺少必填字段的输入。
"""

from pathlib import Path

import pytest

from haagent.runtime.task_contract import (
    TaskLoadError,
    TaskSpec,
    load_task,
    resolve_workspace_root,
)


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
        workspace_root=None,
        policy={"approval_allowed_tools": [], "approved_tools": []},
    )


def test_load_task_reads_optional_workspace_root(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Build with workspace root
workspace_root: ..
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
    )

    task = load_task(task_path)

    assert task.workspace_root == ".."


def test_load_task_defaults_policy_approval_allowed_tools_to_empty(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Default policy
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
    )

    task = load_task(task_path)

    assert task.policy == {"approval_allowed_tools": [], "approved_tools": []}


def test_load_task_reads_policy_approval_allowed_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Policy config
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
  approved_tools: []
""".strip(),
    )

    task = load_task(task_path)

    assert task.policy == {"approval_allowed_tools": ["shell"], "approved_tools": []}


def test_load_task_reads_policy_approved_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Policy approvals
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
  approved_tools:
    - shell
""".strip(),
    )

    task = load_task(task_path)

    assert task.policy == {"approval_allowed_tools": ["shell"], "approved_tools": ["shell"]}


def test_load_task_allows_dynamic_policy_tool_when_listed_in_allowed_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Dynamic MCP tool
constraints: []
allowed_tools:
  - mcp__fixture__echo
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - mcp__fixture__echo
""".strip(),
    )

    task = load_task(task_path)

    assert task.policy == {
        "approval_allowed_tools": ["mcp__fixture__echo"],
        "approved_tools": [],
    }


def test_load_task_rejects_non_list_policy_approval_allowed_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Bad policy
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools: shell
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="policy.approval_allowed_tools"):
        load_task(task_path)


def test_load_task_rejects_non_list_policy_approved_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Bad approved tools
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
policy:
  approved_tools: shell
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="policy.approved_tools"):
        load_task(task_path)


def test_load_task_rejects_policy_approval_tool_not_in_allowed_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Unknown approval tool
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - mystery_tool
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="policy.approval_allowed_tools must be included in allowed_tools"):
        load_task(task_path)


def test_load_task_rejects_policy_approved_tool_not_in_allowed_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Unknown approved tool
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - fake_tool
  approved_tools:
    - mystery_tool
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="policy.approved_tools must be included in allowed_tools"):
        load_task(task_path)


def test_load_task_rejects_approved_tool_not_approval_allowed(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(
        task_path,
        """
goal: Approved without allowed
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools: []
  approved_tools:
    - shell
""".strip(),
    )

    with pytest.raises(TaskLoadError, match="approved_tools must also appear in approval_allowed_tools"):
        load_task(task_path)


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


def test_openai_tool_call_smoke_task_loads_without_network_verification() -> None:
    task = load_task(Path("examples/tasks/openai_tool_call_smoke.yaml"))

    assert task.allowed_tools == ["fake_tool"]
    assert task.verification_commands == []


def test_openai_chat_file_read_smoke_task_loads_with_existing_workspace() -> None:
    task_path = Path("examples/tasks/openai_chat_file_read_smoke.yaml")
    task = load_task(task_path)
    workspace_root = resolve_workspace_root(task, task_path)

    assert task.workspace_root == "../workspaces/file_read_smoke"
    assert task.allowed_tools == ["file_read"]
    assert task.verification_commands == []
    assert workspace_root.is_dir()
    assert (workspace_root / "notes.txt").read_text(encoding="utf-8").strip() == (
        "HaAgent file_read smoke phrase: harness reads workspace notes."
    )


def test_openai_chat_edit_smoke_task_loads_with_existing_workspace_and_verification() -> None:
    task_path = Path("examples/tasks/openai_chat_edit_smoke.yaml")
    task = load_task(task_path)
    workspace_root = resolve_workspace_root(task, task_path)

    assert task.workspace_root == "../workspaces/edit_smoke"
    assert task.allowed_tools == ["file_read", "apply_patch", "shell"]
    assert task.verification_commands
    assert task.policy == {
        "approval_allowed_tools": ["apply_patch", "shell"],
        "approved_tools": ["apply_patch", "shell"],
    }
    assert workspace_root.is_dir()
    assert "return a - b" in (workspace_root / "app.py").read_text(encoding="utf-8")
