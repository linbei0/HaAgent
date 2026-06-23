"""
tests/test_context_builder.py - ContextBuilder v1 测试

验证上下文文件、上下文索引和工具用途会被写入 episode package。
"""

import json
from pathlib import Path

import pytest

from haagent.context.builder import ContextBuildError, ContextBuilder
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.plan import build_plan
from haagent.runtime.task_contract import TaskSpec
from haagent.tools.router import ToolRouter


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
    _assert_budget(
        goal_source["budget"],
        raw_char_count=len("goal: Build context"),
        model_input_char_count=len("goal: Build context"),
        inclusion_reason="The model needs the task goal.",
    )
    assert all("budget" in source for source in context_manifest["sources"])
    assert all(_has_complete_budget(source) for source in context_manifest["sources"])
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
            "included_source_count": sum(
                1
                for source in context_manifest["sources"]
                if source["budget"]["included_in_model_input"]
            ),
        },
    }


def test_context_builder_model_input_contains_tool_usage(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(["context_find", "file_read", "apply_patch", "apply_patch_set", "shell"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    assert "goal: Build context" in model_input
    assert "context_find: primary choice for locating relevant workspace files" in model_input
    assert "file_read: read a workspace text file with offset, limit, or keyword context" in model_input
    assert "Prefer context_find before file_search when the user describes functionality without paths." in model_input
    assert "Use apply_patch_set for related edits across multiple files or multiple replacements." in model_input
    assert "Use workspace-relative paths in tool arguments; use cwd='.' or omit cwd for the workspace root." in model_input
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
    _assert_budget(
        plan_source["budget"],
        raw_char_count=len(injected_plan),
        model_input_char_count=len(injected_plan),
        inclusion_reason=plan_source["inclusion_reason"],
    )


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
    assert context_manifest["next_action"] == {
        "status": "none",
        "reason": "none",
        "based_on_observation_index": None,
        "based_on_tool_name": None,
    }
    assert pending_source["name"] == "pending_next_step"
    assert pending_source["description"]
    assert pending_source["inclusion_reason"]
    _assert_budget(
        pending_source["budget"],
        raw_char_count=len(injected_pending_next_step),
        model_input_char_count=len(injected_pending_next_step),
        inclusion_reason=pending_source["inclusion_reason"],
    )


def test_context_builder_includes_bounded_working_state_source_and_budget(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    working_state = {
        "current_goal": "Understand the project",
        "key_findings": ["README explains setup"],
        "completed_actions": ["Read README"],
        "next_steps": ["Inspect tests"],
        "last_updated_turn": 2,
    }
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        working_state=working_state,
    )

    result = builder.build()

    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    working_state_source = next(
        source for source in context_manifest["sources"] if source["source_type"] == "working_state"
    )
    assert "Working State:" in result.model_input
    assert "current_goal: Understand the project" in result.model_input
    assert "- README explains setup" in result.model_input
    assert working_state_source["name"] == "working_state"
    assert working_state_source["budget"]["included_in_model_input"] is True
    assert working_state_source["budget"]["model_input_char_count"] <= 1200
    assert _has_complete_budget(working_state_source)


def test_context_builder_truncates_long_working_state_and_excludes_trace_text(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    trace_text = '"event": "model_call" tool-calls.jsonl TRANSCRIPT_SENTINEL'
    working_state = {
        "current_goal": "G" * 5000,
        "key_findings": ["safe finding", trace_text],
        "completed_actions": ["A" * 5000],
        "next_steps": ["N" * 5000],
        "last_updated_turn": 9,
    }
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        working_state=working_state,
    )

    result = builder.build()

    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    working_state_source = next(
        source for source in context_manifest["sources"] if source["source_type"] == "working_state"
    )
    assert "Working State:" in result.model_input
    assert len(result.model_input) <= 12000
    assert "G" * 1000 not in result.model_input
    assert trace_text not in result.model_input
    assert '"event": "model_call"' not in result.model_input
    assert "tool-calls.jsonl" not in result.model_input
    assert working_state_source["budget"]["raw_char_count"] > working_state_source["budget"]["model_input_char_count"]
    assert working_state_source["budget"]["truncated"] is True


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
    _assert_budget(
        project_sources[0]["budget"],
        raw_char_count=len("Use concise Chinese comments."),
        model_input_char_count=len("Use concise Chinese comments."),
        inclusion_reason=project_sources[0]["inclusion_reason"],
    )


def test_context_builder_truncates_long_project_instructions(tmp_path: Path) -> None:
    content = "BEGIN-" + ("x" * 5000) + "-TAIL"
    (tmp_path / "AGENTS.md").write_text(content, encoding="utf-8")
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
    source = next(
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "project_instructions"
    )
    assert "BEGIN-" in model_input
    assert "-TAIL" not in model_input
    assert content not in model_input
    assert source["budget"]["raw_char_count"] == len(content)
    assert source["budget"]["model_input_char_count"] < source["budget"]["raw_char_count"]
    assert source["budget"]["truncated"] is True


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
    _assert_budget(
        project_sources[0]["budget"],
        raw_char_count=0,
        model_input_char_count=len("- none"),
        inclusion_reason=project_sources[0]["inclusion_reason"],
    )


def test_context_builder_excludes_episode_audit_sources_from_model_input(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    writer.append_transcript({"event": "trace_sentinel", "content": "TRACE_SENTINEL"})
    writer.append_tool_call(
        {
            "tool_name": "fake_tool",
            "status": "success",
            "result": {"content": "TOOL_CALL_SENTINEL"},
            "policy": {"reason": "POLICY_SENTINEL"},
        },
    )
    verification_dir = writer.path / "verification"
    verification_dir.mkdir()
    (verification_dir / "commands.jsonl").write_text(
        json.dumps({"stdout": "VERIFICATION_SENTINEL"}) + "\n",
        encoding="utf-8",
    )
    (writer.path / "failure.json").write_text(
        json.dumps({"evidence": "FAILURE_SENTINEL"}),
        encoding="utf-8",
    )
    (writer.path / "eval-case.json").write_text(
        json.dumps({"input": "EVAL_SENTINEL"}),
        encoding="utf-8",
    )
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    for sentinel in [
        "TRACE_SENTINEL",
        "TOOL_CALL_SENTINEL",
        "POLICY_SENTINEL",
        "VERIFICATION_SENTINEL",
        "FAILURE_SENTINEL",
        "EVAL_SENTINEL",
    ]:
        assert sentinel not in model_input
    audit_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"].startswith("audit_")
    ]
    assert {source["name"] for source in audit_sources} == {
        "transcript.jsonl",
        "tool-calls.jsonl",
        "verification/commands.jsonl",
        "failure.json",
        "eval export",
    }
    assert all(source["budget"]["included_in_model_input"] is False for source in audit_sources)
    assert all(source["budget"]["model_input_char_count"] == 0 for source in audit_sources)
    assert all(source["budget"]["exclusion_reason"] for source in audit_sources)


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
    assert '"args_keys": ["round"]' in model_input
    assert '"result_keys": ["echo", "status"]' in model_input
    assert '"status": "success"' in model_input
    assert '"result": {"echo": {"round": 1}, "status": "success"}' not in model_input
    assert "Pending next step:" in model_input
    expected_reason = (
        "Continue from the latest successful tool observation. "
        "A successful tool result has already been received; do not repeat the same "
        "successful tool call unless new information is truly needed. If the acceptance "
        "criteria are satisfied, produce the final answer instead of continuing with "
        "another tool call."
    )
    assert expected_reason in model_input
    assert "successful tool result has already been received" in model_input
    assert "do not repeat the same successful tool call" in model_input
    assert "produce the final answer instead of continuing with another tool call" in model_input
    assert context_manifest["next_action"] == {
        "status": "continue",
        "reason": expected_reason,
        "based_on_observation_index": 0,
        "based_on_tool_name": "fake_tool",
    }
    assert len(observation_sources) == 1
    assert observation_sources[0]["source_type"] == "observation"
    assert observation_sources[0]["name"] == "fake_tool"
    assert observation_sources[0]["description"] == "Tool observation from previous turn"
    assert observation_sources[0]["inclusion_reason"] == "Previous tool result is needed for the next model turn."
    assert observation_sources[0]["budget"]["included_in_model_input"] is True
    assert _has_complete_budget(observation_sources[0])


def test_context_builder_compacts_request_user_input_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    long_answer = "A" * 1200
    builder = ContextBuilder(
        task=make_task(["request_user_input"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "request_user_input",
                "args": {"question": "Which file?", "reason": "Need target"},
                "result": {
                    "status": "success",
                    "question": "Which file?",
                    "answer": long_answer,
                    "answer_chars": len(long_answer),
                },
            },
        ],
    )

    result = builder.build()

    assert "Which file?" in result.model_input
    assert "A" * 240 in result.model_input
    assert "A" * 400 not in result.model_input
    assert '"answer_chars": 1200' in result.model_input
    assert '"truncated": true' in result.model_input


def test_context_builder_compacts_long_file_read_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    long_content = "\n".join([f"line-{index:03d}-" + ("x" * 20) for index in range(80)])
    ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    ).build()
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "file_read",
                "args": {"path": "big.txt", "offset": 10, "limit": 80},
                "result": {
                    "status": "success",
                    "path": str(tmp_path / "big.txt"),
                    "offset": 10,
                    "limit": 80,
                    "content": long_content,
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0002.json").read_text(encoding="utf-8"))
    observation_line = _single_observation_line(model_input)
    observation_source = _single_observation_source(context_manifest)
    assert '"path": "big.txt"' in observation_line
    assert '"offset": 10' in observation_line
    assert '"limit": 80' in observation_line
    assert '"start_line": 11' in observation_line
    assert '"end_line": 90' in observation_line
    assert '"line_count": 80' in observation_line
    assert '"excerpt":' in observation_line
    assert '"truncated": true' in observation_line
    assert "line-000-" in observation_line
    assert "line-079-" not in model_input
    assert long_content not in model_input
    assert observation_source["budget"]["model_input_char_count"] == len(observation_line)
    assert observation_source["budget"]["raw_char_count"] > len(observation_line)


def test_context_builder_compacts_long_file_search_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    matches = [
        {
            "path": str(tmp_path / "notes.txt"),
            "line": index + 1,
            "column": 1,
            "text": f"needle match {index:03d} " + ("m" * 40),
        }
        for index in range(30)
    ]
    ContextBuilder(
        task=make_task(["file_search"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    ).build()
    builder = ContextBuilder(
        task=make_task(["file_search"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "file_search",
                "args": {"query": "needle", "root": "."},
                "result": {"status": "success", "matches": matches},
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"query": "needle"' in observation_line
    assert '"match_count": 30' in observation_line
    assert '"excerpt":' in observation_line
    assert '"truncated": true' in observation_line
    assert "needle match 000" in observation_line
    assert "needle match 029" not in model_input
    assert json.dumps(matches, ensure_ascii=False) not in model_input


def test_context_builder_compacts_context_find_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    candidates = [
        {
            "path": "src/app.py",
            "line": 2,
            "excerpt": "def greet(name): " + ("x" * 500),
            "score": 6,
            "reasons": ["filename", "text"],
            "recommended_file_read": {"path": "src/app.py", "keyword": "greet", "limit": 80},
        },
        {
            "path": "README.md",
            "line": 4,
            "excerpt": "Usage mentions greeting",
            "score": 2,
            "reasons": ["text"],
            "recommended_file_read": {"path": "README.md", "keyword": "greeting", "limit": 80},
        },
    ]
    ContextBuilder(
        task=make_task(["context_find"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    ).build()
    builder = ContextBuilder(
        task=make_task(["context_find"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "context_find",
                "args": {"query": "find greeting logic", "max_results": 5},
                "result": {
                    "status": "success",
                    "query": "find greeting logic",
                    "keywords": ["greeting", "logic"],
                    "candidate_count": 2,
                    "candidates": candidates,
                    "truncated": True,
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"query": "find greeting logic"' in observation_line
    assert '"keywords": ["greeting", "logic"]' in observation_line
    assert '"candidate_count": 2' in observation_line
    assert '"path": "src/app.py"' in observation_line
    assert '"recommended_file_read": {"keyword": "greet", "limit": 80, "path": "src/app.py"}' in observation_line
    assert "x" * 300 not in model_input
    assert json.dumps(candidates, ensure_ascii=False) not in model_input


def test_context_builder_compacts_long_file_list_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    tree = "\n".join(f"src/file_{index:03d}.py" for index in range(80))
    ContextBuilder(
        task=make_task(["file_list"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    ).build()
    builder = ContextBuilder(
        task=make_task(["file_list"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "file_list",
                "args": {"path": ".", "max_depth": 2, "max_entries": 100},
                "result": {
                    "status": "success",
                    "path": ".",
                    "max_depth": 2,
                    "max_entries": 100,
                    "entry_count": 80,
                    "truncated": False,
                    "tree": tree,
                    "skipped_dirs": [".git", ".runs"],
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"path": "."' in observation_line
    assert '"entry_count": 80' in observation_line
    assert '"tree_excerpt":' in observation_line
    assert '"truncated": true' in observation_line
    assert "src/file_000.py" in observation_line
    assert "src/file_079.py" not in model_input
    assert tree not in model_input


def test_context_builder_compacts_long_shell_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    stdout = "stdout-start-" + ("o" * 600) + "-stdout-end"
    stderr = "stderr-start-" + ("e" * 600) + "-stderr-end"
    ContextBuilder(
        task=make_task(["shell"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    ).build()
    builder = ContextBuilder(
        task=make_task(["shell"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "shell",
                "args": {"command": "pytest -q", "cwd": "src"},
                "result": {
                    "status": "error",
                    "exit_code": 1,
                    "stdout": stdout,
                    "stderr": stderr,
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"command": "pytest -q"' in observation_line
    assert '"cwd": "src"' in observation_line
    assert '"exit_code": 1' in observation_line
    assert '"stdout_excerpt": "stdout-start-' in observation_line
    assert '"stderr_excerpt": "stderr-start-' in observation_line
    assert '"truncated": true' in observation_line
    assert "-stdout-end" not in model_input
    assert "-stderr-end" not in model_input
    assert stdout not in model_input
    assert stderr not in model_input


def test_context_builder_compacts_apply_patch_observation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(["apply_patch"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "apply_patch",
                "args": {
                    "path": "app.py",
                    "old_text": "old value",
                    "new_text": "new value with more detail",
                },
                "result": {
                    "status": "success",
                    "path": str(tmp_path / "app.py"),
                    "replacements": 1,
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"path": "app.py"' in observation_line
    assert '"status": "success"' in observation_line
    assert '"old_text_length": 9' in observation_line
    assert '"new_text_length": 26' in observation_line
    assert '"old_text_excerpt": "old value"' in observation_line
    assert '"new_text_excerpt": "new value with more detail"' in observation_line


def test_context_builder_compacts_file_write_observation_without_content(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    secret_content = "FULL_FILE_CONTENT_SHOULD_NOT_ENTER_MODEL"
    builder = ContextBuilder(
        task=make_task(["file_write"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "file_write",
                "args": {
                    "path": "notes.txt",
                    "content": secret_content,
                    "mode": "overwrite",
                },
                "result": {
                    "status": "success",
                    "path": str(tmp_path / "notes.txt"),
                    "mode": "overwrite",
                    "bytes_written": len(secret_content.encode("utf-8")),
                    "created": False,
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"path": "notes.txt"' in observation_line
    assert '"mode": "overwrite"' in observation_line
    assert '"bytes_written": 40' in observation_line
    assert '"created": false' in observation_line
    assert secret_content not in model_input


def test_context_builder_compacts_code_run_observation_without_full_output_or_code(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    code = "print('CODE_SHOULD_NOT_ENTER_MODEL')"
    stdout = "stdout-start-" + ("o" * 600) + "-stdout-end"
    stderr = "stderr-start-" + ("e" * 600) + "-stderr-end"
    builder = ContextBuilder(
        task=make_task(["code_run"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "code_run",
                "args": {"code": code, "timeout_seconds": 5},
                "result": {
                    "status": "error",
                    "exit_code": 2,
                    "stdout_excerpt": stdout[:240],
                    "stderr_excerpt": stderr[:240],
                    "truncated": True,
                    "script_path": ".haagent-tmp/code-run-test.py",
                    "error": {"type": "code_run_failed", "message": "python exited with code 2"},
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"exit_code": 2' in observation_line
    assert '"script_path": ".haagent-tmp/code-run-test.py"' in observation_line
    assert '"stdout_excerpt": "stdout-start-' in observation_line
    assert '"stderr_excerpt": "stderr-start-' in observation_line
    assert '"truncated": true' in observation_line
    assert "CODE_SHOULD_NOT_ENTER_MODEL" not in model_input
    assert "-stdout-end" not in model_input
    assert "-stderr-end" not in model_input


def test_context_builder_compacts_apply_patch_set_without_full_patch_text(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(["apply_patch_set"]),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "apply_patch_set",
                "args": {
                    "replacements": [
                        {
                            "path": "src/app.py",
                            "old_text": "SECRET_OLD_TEXT_SHOULD_NOT_ENTER_MODEL",
                            "new_text": "SECRET_NEW_TEXT_SHOULD_NOT_ENTER_MODEL",
                        },
                    ],
                },
                "result": {
                    "status": "error",
                    "replacement_count": 1,
                    "error": {"type": "patch_text_not_found", "message": "old_text was not found"},
                    "replacements": [
                        {
                            "index": 0,
                            "path": "src/app.py",
                            "status": "error",
                            "reason": "old_text was not found",
                            "match_count": 0,
                        },
                    ],
                },
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    observation_line = _single_observation_line(model_input)
    assert '"path": "src/app.py"' in observation_line
    assert '"replacement_count": 1' in observation_line
    assert "patch_text_not_found" in observation_line
    assert "SECRET_OLD_TEXT_SHOULD_NOT_ENTER_MODEL" not in model_input
    assert "SECRET_NEW_TEXT_SHOULD_NOT_ENTER_MODEL" not in model_input


def test_tool_call_trace_keeps_full_result_when_context_compacts_observation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "big.txt"
    full_content = "alpha\n" + ("z" * 900) + "\nomega\n"
    target.write_text(full_content, encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "big.txt", "offset": 0, "limit": 20})
    ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "file_read",
                "args": {"path": "big.txt"},
                "result": result,
            },
        ],
    ).build()

    trace_record = json.loads((writer.path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    assert trace_record["result"]["content"] == full_content
    assert full_content not in model_input
    assert '"truncated": true' in model_input


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
    expected_reason = "Use the latest tool error to adjust parameters, or stop and explain the failure explicitly."
    assert "Pending next step:" in model_input
    assert expected_reason in model_input
    assert "do not repeat the same successful tool call" not in model_input
    assert "produce the final answer instead of continuing with another tool call" not in model_input
    assert context_manifest["next_action"] == {
        "status": "handle_error",
        "reason": expected_reason,
        "based_on_observation_index": 0,
        "based_on_tool_name": "fake_tool",
    }
    assert pending_source["budget"]["included_in_model_input"] is True
    assert pending_source["budget"]["inclusion_reason"]


@pytest.mark.parametrize("latest_status", ["unknown", "failed"])
def test_context_builder_next_action_handles_unknown_tool_status(
    tmp_path: Path,
    latest_status: str,
) -> None:
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
                "result": {"status": "weird"},
            },
            {
                "tool_name": "file_read",
                "args": {"path": "notes.txt"},
                "result": {"status": latest_status},
            },
        ],
    )

    builder.build()

    model_input = (writer.path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((writer.path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    expected_reason = "Use the latest tool observation to decide the next action explicitly."
    assert expected_reason in model_input
    assert context_manifest["next_action"] == {
        "status": "decide",
        "reason": expected_reason,
        "based_on_observation_index": 1,
        "based_on_tool_name": "file_read",
    }


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
    assert all(_has_complete_budget(source) for source in context_manifest["sources"])


def test_context_builder_allows_long_agents_md_after_truncation(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 13000, encoding="utf-8")
    writer = make_writer(tmp_path)
    builder = ContextBuilder(
        task=make_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
    )

    result = builder.build()

    assert len(result.model_input) <= 12000


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


def _single_observation_line(model_input: str) -> str:
    observation_lines = [
        line
        for line in model_input.splitlines()
        if line.startswith("- ") and line != "- none" and "truncated" in line
    ]
    assert len(observation_lines) == 1
    return observation_lines[0]


def _single_observation_source(context_manifest: dict[str, object]) -> dict[str, object]:
    sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "observation"
    ]
    assert len(sources) == 1
    return sources[0]


def _assert_budget(
    budget: dict[str, object],
    raw_char_count: int,
    model_input_char_count: int,
    inclusion_reason: str,
    truncated: bool = False,
) -> None:
    assert budget == {
        "raw_char_count": raw_char_count,
        "model_input_char_count": model_input_char_count,
        "included_in_model_input": True,
        "truncated": truncated,
        "inclusion_reason": inclusion_reason,
        "exclusion_reason": None,
    }


def _has_complete_budget(source: dict[str, object]) -> bool:
    budget = source["budget"]
    return {
        "raw_char_count",
        "model_input_char_count",
        "included_in_model_input",
        "truncated",
        "inclusion_reason",
        "exclusion_reason",
    } <= set(budget)
