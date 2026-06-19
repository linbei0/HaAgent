"""
tests/test_context_builder.py - ContextBuilder v1 测试

验证上下文文件、上下文索引和工具用途会被写入 episode package。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.context.builder import ContextBuildError, ContextBuilder
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.plan import build_plan
from agentfoundry.runtime.task_contract import TaskSpec


def make_task(allowed_tools: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        goal="Build context",
        constraints=["No retrieval"],
        allowed_tools=allowed_tools or ["fake_tool", "file_read"],
        acceptance_criteria=["Context is auditable"],
        verification_commands=["uv run pytest"],
    )


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task = make_task()
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Build context
constraints:
  - No retrieval
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Context is auditable
verification_commands:
  - uv run pytest
""".strip(),
        encoding="utf-8",
    )
    writer = EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
    writer.write_plan(build_plan(task))
    return writer


def test_context_builder_writes_context_files_and_manifest(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    result = builder.build()

    assert result.context_id == "0001"
    assert (writer.path / "contexts" / "0001.txt").exists()
    assert (writer.path / "contexts" / "0001.json").exists()
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    assert context_manifest["generated_at"]
    assert context_manifest["budget"] == {
        "character_count": len(result.model_input),
        "character_limit": 12000,
        "status": "within_limit",
    }
    goal_source = next(
        source for source in context_manifest["sources"] if source["source_type"] == "task" and source["name"] == "goal"
    )
    assert goal_source["budget"] == {
        "char_count": len("goal: Build context"),
        "included_in_model_input": True,
        "inclusion_reason": "The model needs the task goal.",
    }
    assert all("budget" in source for source in context_manifest["sources"])
    assert all(source["budget"]["inclusion_reason"] for source in context_manifest["sources"])
    manifest = json.loads((writer.path / "context-manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.2"
    assert manifest["generated_at"]
    assert manifest["context_count"] == 1
    assert manifest["summary"]["provider"] == "fake"
    assert manifest["summary"]["workspace_root"] == str(tmp_path)
    assert manifest["contexts"][0] == {
        "context_id": "0001",
        "model_input_path": "contexts/0001.txt",
        "manifest_path": "contexts/0001.json",
        "budget": {
            "context_id": "0001",
            "total_chars": len(result.model_input),
            "max_chars": 12000,
            "status": "within_limit",
            "source_count": len(context_manifest["sources"]),
            "included_source_count": len(context_manifest["sources"]),
        },
    }


def test_context_builder_model_input_contains_tool_usage(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    assert "goal: Build context" in model_input
    assert "fake_tool: deterministic test tool" in model_input
    assert "file_read: read a workspace text file with offset and limit" in model_input
    assert "verification_commands:" in model_input
    assert "Plan:" in model_input
    assert "Observations:" in model_input
    assert "Pending next step:" in model_input
    assert "- none" in model_input


def test_context_builder_includes_plan_source_and_budget(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    plan_source = next(source for source in context_manifest["sources"] if source["source_type"] == "plan")
    injected_plan = "\n".join(
        [
            "Plan:",
            "- Clarify the task goal and constraints from task.yaml.",
            "- Use allowed tools: fake_tool, file_read.",
            "- Check acceptance criteria: Context is auditable.",
            "- Run verification commands if provided.",
        ],
    )
    assert injected_plan in model_input
    assert plan_source["name"] == "plan.json"
    assert plan_source["description"]
    assert plan_source["inclusion_reason"]
    assert plan_source["budget"] == {
        "char_count": len(injected_plan),
        "included_in_model_input": True,
        "inclusion_reason": plan_source["inclusion_reason"],
    }


def test_context_builder_includes_pending_next_step_none_source_and_budget(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    pending_source = next(
        source for source in context_manifest["sources"] if source["source_type"] == "pending_next_step"
    )
    injected_pending_next_step = "\n".join(["Pending next step:", "- none"])
    assert injected_pending_next_step in model_input
    assert pending_source["name"] == "pending_next_step"
    assert pending_source["description"]
    assert pending_source["inclusion_reason"]
    assert pending_source["budget"] == {
        "char_count": len(injected_pending_next_step),
        "included_in_model_input": True,
        "inclusion_reason": pending_source["inclusion_reason"],
    }


def test_context_builder_includes_project_instructions_when_agents_md_exists(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use concise Chinese comments.", encoding="utf-8")
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    project_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "project_instructions"
    ]
    assert "Project Instructions:" in model_input
    assert "Use concise Chinese comments." in model_input
    assert len(project_sources) == 1
    assert project_sources[0]["source_type"] == "project_instructions"
    assert project_sources[0]["name"] == "AGENTS.md"
    assert project_sources[0]["description"] == "Project instructions from workspace AGENTS.md"
    assert project_sources[0]["status"] == "present"
    assert project_sources[0]["inclusion_reason"] == "Workspace AGENTS.md is the project instruction source for this run."
    assert project_sources[0]["budget"]["included_in_model_input"] is True


def test_context_builder_records_absent_project_instructions(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    project_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "project_instructions"
    ]
    assert len(project_sources) == 1
    assert project_sources[0]["source_type"] == "project_instructions"
    assert project_sources[0]["name"] == "AGENTS.md"
    assert project_sources[0]["description"] == "workspace AGENTS.md not found"
    assert project_sources[0]["status"] == "absent"
    assert project_sources[0]["inclusion_reason"] == "Absence is recorded so audits can see no project instructions were loaded."
    assert project_sources[0]["budget"]["included_in_model_input"] is True


def test_context_builder_model_input_contains_observation_summary(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "fake_tool",
                "args": {"round": 1},
                "result": {"status": "success", "echo": {"round": 1}},
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    observation_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "observation"
    ]
    assert "Observations:" in model_input
    assert "fake_tool" in model_input
    assert '"args": {"round": 1}' in model_input
    assert '"result": {"echo": {"round": 1}, "status": "success"}' in model_input
    assert "Pending next step:" in model_input
    assert "Continue from the latest successful tool observation and judge whether the acceptance criteria are satisfied." in model_input
    assert len(observation_sources) == 1
    assert observation_sources[0]["source_type"] == "observation"
    assert observation_sources[0]["name"] == "fake_tool"
    assert observation_sources[0]["description"] == "Tool observation from previous turn"
    assert observation_sources[0]["inclusion_reason"] == "Previous tool result is needed for the next model turn."
    assert observation_sources[0]["budget"]["included_in_model_input"] is True


def test_context_builder_pending_next_step_handles_tool_error(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "fake_tool",
                "args": {"round": 1},
                "result": {"status": "error", "error": "bad args"},
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    pending_source = next(
        source for source in context_manifest["sources"] if source["source_type"] == "pending_next_step"
    )
    assert "Pending next step:" in model_input
    assert "Use the latest tool error to adjust parameters, or stop and explain the failure explicitly." in model_input
    assert pending_source["budget"]["included_in_model_input"] is True
    assert pending_source["budget"]["inclusion_reason"]


def test_context_builder_records_each_allowed_tool_as_source(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(["fake_tool", "file_read"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    tool_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "tool_catalog"
    ]
    assert [source["name"] for source in tool_sources] == ["fake_tool", "file_read"]
    assert tool_sources[0]["description"] == "deterministic test tool"


def test_context_builder_sources_include_inclusion_reason(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project instruction.", encoding="utf-8")
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[{"tool_name": "fake_tool", "args": {}, "result": {"status": "success"}}],
    )

    builder.build()

    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    source_types = {source["source_type"] for source in context_manifest["sources"]}
    assert {"task", "tool_catalog", "observation", "project_instructions", "plan", "pending_next_step"} <= source_types
    assert all(source["inclusion_reason"] for source in context_manifest["sources"])


def test_context_builder_rejects_context_over_character_limit(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 13000, encoding="utf-8")
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    with pytest.raises(ContextBuildError, match="context character budget exceeded"):
        builder.build()


def test_context_builder_rejects_unknown_allowed_tool(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(["fake_tool", "mystery_tool"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    with pytest.raises(ContextBuildError, match="mystery_tool"):
        builder.build()
