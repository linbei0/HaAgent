from __future__ import annotations

from haagent.context.selection import ContextSelector, ContextSelectionBudget
from haagent.context.sources import ContextCandidate


def test_hard_required_candidate_survives_budget() -> None:
    result = ContextSelector(
        budget=ContextSelectionBudget(max_system_chars=10, max_task_chars=10, max_source_chars=100),
    ).select(
        [
            ContextCandidate(
                source_type="task_contract",
                source_id="task",
                placement="task",
                title="Task",
                content="hard-required-content",
                reason="current_user_request",
                priority=0,
                hard_required=True,
            ),
            ContextCandidate(
                source_type="memory",
                source_id="memory-1",
                placement="task",
                title="Memory",
                content="optional",
                reason="memory_retrieval_match",
                priority=50,
            ),
        ],
    )

    assert [section.source_id for section in result.task_sections] == ["task"]
    skipped = {decision.source_id: decision for decision in result.skipped}
    assert skipped["memory-1"].skip_reason == "over_budget"


def test_low_priority_candidate_is_skipped_when_over_budget() -> None:
    result = ContextSelector(
        budget=ContextSelectionBudget(max_system_chars=40, max_task_chars=100, max_source_chars=100),
    ).select(
        [
            ContextCandidate(
                source_type="project_instructions",
                source_id="AGENTS.md",
                placement="system",
                title="Project Instructions",
                content="A" * 30,
                reason="workspace_agents_md_found",
                priority=10,
            ),
            ContextCandidate(
                source_type="tool_workflow",
                source_id="tool-workflow",
                placement="system",
                title="Tool Workflow",
                content="B" * 30,
                reason="allowed_tools_present",
                priority=70,
            ),
        ],
    )

    assert [section.source_id for section in result.system_sections] == ["AGENTS.md"]
    assert result.skipped[0].source_id == "tool-workflow"
    assert result.skipped[0].skip_reason == "over_budget"


def test_selection_manifest_records_selected_skipped_and_budget() -> None:
    result = ContextSelector(
        budget=ContextSelectionBudget(max_system_chars=100, max_task_chars=10, max_source_chars=8),
    ).select(
        [
            ContextCandidate(
                source_type="session_summary",
                source_id="session",
                placement="task",
                title="Session Summary",
                content="123456789012",
                reason="resumed_session",
                priority=30,
            ),
            ContextCandidate(
                source_type="working_state",
                source_id="working-state",
                placement="task",
                title="Working State",
                content="abcd",
                reason="working_state_present",
                priority=20,
            ),
        ],
    )

    manifest = result.to_manifest_dict()

    assert manifest["budget"]["max_task_chars"] == 10
    assert manifest["selected"][0]["source_id"] == "working-state"
    assert manifest["skipped"][0]["source_id"] == "session"
    assert manifest["skipped"][0]["skip_reason"] == "over_budget"
    assert result.skipped[0].metadata["truncated"] is True
