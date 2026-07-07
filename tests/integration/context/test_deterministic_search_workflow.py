"""
tests/integration/context/test_deterministic_search_workflow.py - 确定性搜索工作流测试

验证默认上下文发现路径统一为 file_list、grep、file_read，不再暴露自然语言启发式 context_find。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from haagent.context.builder import ContextBuilder
from haagent.runtime.session.turn import write_chat_task_yaml
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.contracts.task import TaskSpec
from haagent.tools.registry import TOOL_REGISTRY, export_tool_schemas
from haagent.tools.router import ToolRouter


def test_context_find_is_not_registered_or_exported() -> None:
    assert "context_find" not in TOOL_REGISTRY
    with pytest.raises(KeyError):
        export_tool_schemas(["context_find"])


def test_tool_router_rejects_context_find_without_dispatch_entry(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, allowed_tools=["file_list", "grep", "file_read"])
    router = ToolRouter(["context_find"], writer, workspace_root=tmp_path)

    result = router.dispatch("context_find", {"query": "greeting"})

    assert result["status"] == "error"
    assert result["error"]["type"] in {"unknown_tool", "tool_not_allowed"}


def test_context_builder_recommends_file_list_search_read_workflow(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path, allowed_tools=["file_list", "grep", "file_read"])
    context = ContextBuilder(
        task=TaskSpec(
            goal="Find greeting implementation",
            workspace_root=".",
            allowed_tools=["file_list", "grep", "file_read"],
            acceptance_criteria=[],
            verification_commands=[],
            constraints=[],
            policy={"approval_allowed_tools": [], "approved_tools": []},
        ),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    assert "Use file_list to inspect directory structure or narrow the search scope." in context.model_input
    assert "Use grep for exact deterministic text search" in context.model_input
    assert "Then use file_read on candidate files before editing or summarizing." in context.model_input
    assert "context_find" not in context.model_input


def test_chat_task_yaml_allowed_tools_excludes_context_find(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"

    write_chat_task_yaml(task_path, "summarize docs", tmp_path)

    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    assert "context_find" not in task["allowed_tools"]
    assert task["allowed_tools"][:3] == ["file_list", "grep", "file_read"]


def test_grep_then_file_read_workflow_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def greet():\n    return 'Hello'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Usage: greet from src/app.py\n", encoding="utf-8")
    writer = _make_writer(tmp_path, allowed_tools=["file_list", "grep", "file_read"])
    router = ToolRouter(["file_list", "grep", "file_read"], writer, workspace_root=tmp_path)

    listed = router.dispatch("file_list", {"path": ".", "max_depth": 2})
    searched = router.dispatch("grep", {"pattern": "greet", "root": "."})
    read = router.dispatch("file_read", {"path": "src/app.py", "keyword": "greet", "limit": 20})

    assert listed["status"] == "success"
    assert searched["status"] == "success"
    assert {match["path"] for match in searched["matches"]} == {"README.md", "src/app.py"}
    assert read["status"] == "success"
    assert "def greet" in read["content"]
    tool_names = [
        json.loads(line)["tool_name"]
        for line in (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert tool_names == ["file_list", "grep", "file_read"]


def _make_writer(tmp_path: Path, *, allowed_tools: list[str]) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "goal": "Find context",
                "allowed_tools": allowed_tools,
                "acceptance_criteria": ["Context found"],
                "verification_commands": [],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    writer = EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
    writer.write_plan(
        {
            "goal": "Find context",
            "constraints": [],
            "acceptance_criteria": ["Context found"],
            "verification_commands": [],
            "planned_steps": ["Use deterministic search primitives."],
        },
    )
    return writer
