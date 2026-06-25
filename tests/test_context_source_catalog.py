"""
tests/test_context_source_catalog.py - Context source catalog 测试

验证 context manifest source 的预算、排除规则和 raw/model 内容映射。
"""

from pathlib import Path

from haagent.context.source_catalog import (
    AUDIT_SOURCE_EXCLUSION_REASON,
    ContextSourceCatalog,
)
from haagent.runtime.task_contract import TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        goal="Build context",
        constraints=["No retrieval"],
        allowed_tools=["fake_tool", "file_read"],
        acceptance_criteria=["Context is auditable"],
        verification_commands=["uv run pytest"],
    )


def make_catalog(tmp_path: Path, **overrides) -> ContextSourceCatalog:
    values = {
        "task": make_task(),
        "tool_workflow_hints": ["Use the allowed tools only as needed for the task."],
        "project_instructions": None,
        "project_instruction_lines": ["- none"],
        "plan_lines": ["- Clarify the task goal."],
        "pending_next_step_lines": ["- none"],
        "observations": [],
        "session_summary": None,
        "session_summary_lines": ["- none"],
        "working_state": None,
        "working_state_model_content": "",
        "working_state_raw_content": "",
        "episode_path": tmp_path,
    }
    values.update(overrides)
    return ContextSourceCatalog(**values)


def source_by(catalog: ContextSourceCatalog, source_type: str, name: str):
    return next(
        source
        for source in catalog.sources_with_budget()
        if source.source_type == source_type and source.name == name
    )


def test_task_source_budget_uses_rendered_task_content(tmp_path: Path) -> None:
    source = source_by(make_catalog(tmp_path), "task", "goal")

    assert source.budget is not None
    assert source.budget.raw_char_count == len("goal: Build context")
    assert source.budget.model_input_char_count == len("goal: Build context")
    assert source.budget.included_in_model_input is True
    assert source.budget.truncated is False
    assert source.budget.exclusion_reason is None


def test_observation_budget_tracks_raw_and_compacted_content(tmp_path: Path) -> None:
    observation = {
        "tool_name": "file_read",
        "args": {"path": "README.md"},
        "result": {
            "status": "success",
            "path": "README.md",
            "content": "x" * 1000,
            "truncated": True,
        },
    }
    source = source_by(
        make_catalog(tmp_path, observations=[observation]),
        "observation",
        "file_read",
    )

    assert source.budget is not None
    assert source.budget.raw_char_count > 0
    assert source.budget.model_input_char_count > 0
    assert source.budget.included_in_model_input is True
    assert source.budget.truncated is True
    assert source.budget.exclusion_reason is None


def test_interaction_state_source_is_included_in_model_input_without_prompt_instructions(tmp_path: Path) -> None:
    source = source_by(
        make_catalog(
            tmp_path,
            interaction_state=[
                {
                    "type": "user_input",
                    "tool": "request_user_input",
                    "status": "answered",
                    "question": "Which file?",
                    "answer_excerpt": "docs/harness-requirements.md",
                    "answer_chars": len("docs/harness-requirements.md"),
                },
            ],
            interaction_state_lines=[
                'type=user_input tool=request_user_input status=answered question="Which file?" answer_excerpt="docs/harness-requirements.md"',
            ],
        ),
        "interaction_state",
        "human_interaction_state",
    )

    assert source.budget is not None
    assert source.description == "Human interaction state recorded during this run"
    assert source.inclusion_reason == "Structured human interaction facts are included in the current context."
    assert "model needs" not in source.inclusion_reason.lower()
    assert "repeat" not in source.inclusion_reason.lower()
    assert source.budget.raw_char_count > 0
    assert source.budget.model_input_char_count > 0
    assert source.budget.included_in_model_input is True
    assert source.budget.exclusion_reason is None


def test_audit_source_is_excluded_from_model_input(tmp_path: Path) -> None:
    (tmp_path / "tool-calls.jsonl").write_text('{"tool_name":"fake_tool"}\n', encoding="utf-8")

    source = source_by(make_catalog(tmp_path), "audit_tool_calls", "tool-calls.jsonl")

    assert source.status == "excluded"
    assert source.budget is not None
    assert source.budget.raw_char_count == len('{"tool_name":"fake_tool"}\n')
    assert source.budget.model_input_char_count == 0
    assert source.budget.included_in_model_input is False
    assert source.budget.truncated is False
    assert source.budget.exclusion_reason == AUDIT_SOURCE_EXCLUSION_REASON


def test_project_instructions_absent_and_present_sources(tmp_path: Path) -> None:
    absent = source_by(make_catalog(tmp_path), "project_instructions", "AGENTS.md")
    present = source_by(
        make_catalog(
            tmp_path,
            project_instructions="Use concise Chinese comments.",
            project_instruction_lines=["Use concise Chinese comments."],
        ),
        "project_instructions",
        "AGENTS.md",
    )

    assert absent.status == "absent"
    assert absent.description == "workspace AGENTS.md not found"
    assert absent.budget is not None
    assert absent.budget.raw_char_count == 0
    assert absent.budget.model_input_char_count == len("- none")
    assert present.status == "present"
    assert present.description == "Project instructions from workspace AGENTS.md"
    assert present.budget is not None
    assert present.budget.raw_char_count == len("Use concise Chinese comments.")
    assert present.budget.model_input_char_count == len("Use concise Chinese comments.")
