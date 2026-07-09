from __future__ import annotations

import json
from pathlib import Path

from haagent.context.builder import ContextBuilder
from haagent.context.compression.sections import (
    ContextBudget,
    ContextCompactionResult,
    ContextSection,
    ContextSelectionRecord,
    assess_compact_readiness,
    assess_auto_compact_trigger,
    compact_context_sections,
)
from haagent.context.compression.session_memory import compact_session_memory
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.orchestration.preparation import prepare_initial_messages


def test_collapse_oversized_section_keeps_head_and_tail() -> None:
    section = ContextSection(
        key="large",
        title="Large",
        content="H" * 12 + "M" * 20 + "T" * 8,
        source="test",
        priority=10,
        kind="project_instructions",
    )

    result = compact_context_sections(
        [section],
        ContextBudget(max_total_chars=100, max_section_chars=20, collapse_head_chars=10, collapse_tail_chars=5),
    )

    assert result.sections[0].content.startswith("H" * 10)
    assert result.sections[0].content.endswith("T" * 5)
    assert "...[collapsed 25 chars]..." in result.sections[0].content
    assert result.diagnostics[0].decision == "collapsed"
    assert result.diagnostics[0].original_chars == 40
    assert result.diagnostics[0].final_chars == len(result.sections[0].content)


def test_keeps_recent_observations_before_old_ones() -> None:
    sections = [
        ContextSection(
            key="old",
            title="Old Observation",
            content="old observation " * 20,
            source="tool-calls",
            priority=5,
            kind="observation",
            recent_rank=2,
        ),
        ContextSection(
            key="middle",
            title="Middle Observation",
            content="middle observation " * 20,
            source="tool-calls",
            priority=5,
            kind="observation",
            recent_rank=1,
        ),
        ContextSection(
            key="recent",
            title="Recent Observation",
            content="recent observation",
            source="tool-calls",
            priority=5,
            kind="observation",
            recent_rank=0,
        ),
    ]

    result = compact_context_sections(
        sections,
        ContextBudget(
            max_total_chars=80,
            max_section_chars=500,
            max_tool_observation_chars=40,
            keep_recent_observations=1,
            collapse_head_chars=20,
            collapse_tail_chars=10,
        ),
    )

    selected_keys = {section.key for section in result.sections}
    assert "recent" in selected_keys
    assert "old" not in selected_keys
    assert {record.key for record in result.diagnostics if record.decision == "skipped"}
    assert result.diagnostics[-1].decision == "selected"


def test_skipped_sections_are_recorded() -> None:
    sections = [
        ContextSection("high", "High", "important", "test", 10, "task"),
        ContextSection("low", "Low", "low priority content", "test", 1, "memory"),
    ]

    result = compact_context_sections(sections, ContextBudget(max_total_chars=12, max_section_chars=100))

    assert [section.key for section in result.sections] == ["high"]
    skipped = [record for record in result.diagnostics if record.decision == "skipped"]
    assert [record.key for record in skipped] == ["low"]
    assert skipped[0].reason == "over_total_budget"
    assert skipped[0].final_chars == 0


def test_no_compaction_when_under_budget() -> None:
    sections = [
        ContextSection("a", "A", "alpha", "test", 2, "task"),
        ContextSection("b", "B", "beta", "test", 1, "memory"),
    ]

    result = compact_context_sections(sections, ContextBudget(max_total_chars=100, max_section_chars=50))

    assert [section.content for section in result.sections] == ["alpha", "beta"]
    assert all(record.decision == "selected" for record in result.diagnostics)
    assert result.original_chars == len("alphabeta")
    assert result.final_chars == len("alphabeta")


def test_compact_readiness_is_sufficient_under_budget() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("a", "test", "task", "selected", "within_budget", 40, 40, 10),
        ],
        original_chars=50,
        final_chars=40,
    )

    readiness = assess_compact_readiness(result, ContextBudget(max_total_chars=100))

    assert readiness["status"] == "deterministic_sufficient"
    assert readiness["budget_pressure"] == 0.4
    assert readiness["saved_ratio"] == 0.2
    assert readiness["recommendation"] == "keep_deterministic"
    assert readiness["reasons"] == ["within_budget_after_compaction"]


def test_compact_readiness_watches_near_budget_limit() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("a", "test", "task", "selected", "within_budget", 84, 84, 10),
        ],
        original_chars=100,
        final_chars=84,
    )

    readiness = assess_compact_readiness(result, ContextBudget(max_total_chars=100))

    assert readiness["status"] == "watch"
    assert readiness["recommendation"] == "keep_deterministic"
    assert "near_budget_limit" in readiness["reasons"]


def test_compact_readiness_marks_full_compact_candidate_for_severe_pressure() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("kept", "test", "task", "selected", "within_budget", 60, 60, 10),
            ContextSelectionRecord("collapsed", "test", "memory", "collapsed", "section_over_budget", 100, 35, 9),
            ContextSelectionRecord("skipped", "test", "memory", "skipped", "over_total_budget", 80, 0, 1),
        ],
        original_chars=240,
        final_chars=95,
    )

    readiness = assess_compact_readiness(result, ContextBudget(max_total_chars=100))

    assert readiness["status"] == "full_compact_candidate"
    assert readiness["recommendation"] == "evaluate_full_compact"
    assert readiness["skipped_count"] == 1
    assert readiness["collapsed_count"] == 1
    assert "near_budget_limit" in readiness["reasons"]
    assert "skipped_context_present" in readiness["reasons"]
    assert "collapsed_context_present" in readiness["reasons"]


def test_auto_compact_trigger_not_needed_under_low_pressure() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("a", "test", "task", "selected", "within_budget", 40, 40, 10),
        ],
        original_chars=50,
        final_chars=40,
    )
    readiness = assess_compact_readiness(result, ContextBudget(max_total_chars=100))

    trigger = assess_auto_compact_trigger(
        compact_readiness=readiness,
        compaction=result,
        budget=ContextBudget(max_total_chars=100),
        historical_tool_compression_count=0,
        session_summary_count=2,
        session_summary_chars=160,
    )

    assert trigger["status"] == "not_needed"
    assert trigger["triggered"] is False
    assert trigger["trigger_kind"] is None
    assert trigger["recommendation"] == "keep_deterministic"
    assert trigger["reasons"] == ["within_budget_after_compaction"]


def test_auto_compact_trigger_watches_near_budget_limit() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("a", "test", "task", "selected", "within_budget", 84, 84, 10),
        ],
        original_chars=100,
        final_chars=84,
    )
    budget = ContextBudget(max_total_chars=100)
    readiness = assess_compact_readiness(result, budget)

    trigger = assess_auto_compact_trigger(
        compact_readiness=readiness,
        compaction=result,
        budget=budget,
        historical_tool_compression_count=0,
        session_summary_count=4,
        session_summary_chars=300,
    )

    assert trigger["status"] == "watch"
    assert trigger["triggered"] is False
    assert trigger["recommendation"] == "keep_deterministic"
    assert "near_budget_limit" in trigger["reasons"]


def test_auto_compact_trigger_applies_session_memory_for_history_over_budget() -> None:
    result = ContextCompactionResult(
        sections=[],
        diagnostics=[
            ContextSelectionRecord("kept", "test", "task", "selected", "within_budget", 60, 60, 10),
            ContextSelectionRecord("collapsed", "test", "memory", "collapsed", "section_over_budget", 100, 35, 9),
        ],
        original_chars=160,
        final_chars=95,
    )
    budget = ContextBudget(max_total_chars=100)
    readiness = assess_compact_readiness(result, budget)

    trigger = assess_auto_compact_trigger(
        compact_readiness=readiness,
        compaction=result,
        budget=budget,
        historical_tool_compression_count=1,
        session_summary_count=12,
        session_summary_chars=1800,
    )

    assert trigger["status"] == "triggered"
    assert trigger["triggered"] is True
    assert trigger["trigger_kind"] == "session_memory"
    assert trigger["recommendation"] == "apply_session_memory_compaction"
    assert "session_history_over_budget" in trigger["reasons"]
    assert "collapsed_context_present" in trigger["reasons"]


def test_session_memory_compaction_preserves_recent_summaries_and_compacts_older_turns(tmp_path: Path) -> None:
    summaries = [
        "\n".join(
            [
                f"- user_request: request {index}",
                "  status: completed" if index % 3 else "  status: failed",
                f"  episode_path: {tmp_path / f'episode-{index}'}",
                f"  assistant_final_response: response {index}",
                "  verification: passed" if index % 2 else "  verification: failed",
            ],
        )
        for index in range(1, 13)
    ]

    result = compact_session_memory(summaries, keep_recent=6, memory_char_limit=2000)

    assert result.diagnostics["decision"] == "compacted"
    assert result.diagnostics["original_turn_count"] == 12
    assert result.diagnostics["compacted_turn_count"] == 6
    assert result.diagnostics["preserved_recent_count"] == 6
    assert result.diagnostics["saved_chars"] > 0
    assert "[session_memory_compacted 6 earlier turns]" in result.summary_text
    lines = result.summary_text.splitlines()
    for index in range(7, 13):
        assert f"- user_request: request {index}" in lines
    for index in range(1, 7):
        assert f"- user_request: request {index}" not in lines
    assert "total_turns: 6" in result.summary_text
    assert "completed: 4" in result.summary_text
    assert "failed: 2" in result.summary_text


def test_context_builder_returns_compaction_diagnostics_and_manifest(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    context = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        session_summary="summary from previous turns",
        session_compaction={
            "decision": "compacted",
            "original_turn_count": 20,
            "compacted_turn_count": 14,
            "preserved_recent_count": 6,
            "original_chars": 8000,
            "final_chars": 1800,
            "saved_chars": 6200,
            "reason": "session_history_over_budget",
        },
    ).build()

    manifest_path = writer.path / "contexts" / f"{context.context_id}-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert context.diagnostics
    assert "compaction" in manifest
    assert manifest["selection"]["selected"]
    assert manifest["selection"]["selected"][0]["source_type"] == "session_summary"
    assert manifest["selection"]["selected"][0]["source_id"] == "session_summary"
    assert [
        record["skip_reason"]
        for record in manifest["selection"]["skipped"]
        if record["source_type"] == "memory_index"
    ] == ["missing_index"]
    assert "selection" not in context.model_input
    compaction = manifest["compaction"]
    assert compaction["selected_count"] == 1
    assert compaction["collapsed_count"] == 0
    assert compaction["skipped_count"] == 0
    assert compaction["selected_chars"] == len("summary from previous turns")
    assert compaction["collapsed_saved_chars"] == 0
    assert compaction["skipped_chars"] == 0
    assert compaction["skipped_reasons"] == {}
    assert compaction["diagnostics"][0]["decision"] == "selected"
    assert {record["key"] for record in compaction["diagnostics"]} == {"session_summary"}
    assert manifest["compact_readiness"]["status"] == "deterministic_sufficient"
    assert manifest["compact_readiness"]["recommendation"] == "keep_deterministic"
    assert manifest["session_compaction"] == {
        "decision": "compacted",
        "original_turn_count": 20,
        "compacted_turn_count": 14,
        "preserved_recent_count": 6,
        "original_chars": 8000,
        "final_chars": 1800,
        "saved_chars": 6200,
        "reason": "session_history_over_budget",
    }
    assert manifest["auto_compact_trigger"]["status"] == "triggered"
    assert manifest["auto_compact_trigger"]["triggered"] is True
    assert manifest["auto_compact_trigger"]["trigger_kind"] == "session_memory"
    assert manifest["auto_compact_trigger"]["recommendation"] == "apply_session_memory_compaction"
    assert manifest["full_compact_contract"] == {
        "eligible": False,
        "reason": "deterministic_context_sufficient",
        "trigger_kind": None,
        "required_preserve_recent": 6,
    }
    assert manifest["source_diagnostics"]["session_summary"] == {
        "present": True,
        "included": True,
        "original_chars": len("summary from previous turns"),
        "model_input_chars": len("summary from previous turns"),
        "limit": 2000,
    }
    assert manifest["source_diagnostics"]["observations"] == {
        "included_in_model_input": False,
        "observation_section_count": 0,
        "compacted_count": 0,
        "truncated_count": 0,
        "skipped_count": 0,
        "original_chars": 0,
        "final_chars": 0,
        "saved_chars": 0,
        "reason": "context_builder_does_not_include_observation_sections",
    }
    assert "diagnostics" not in context.model_input
    assert "compaction" not in context.model_input
    assert "source_diagnostics" not in context.model_input
    assert "compact_readiness" not in context.model_input
    assert "full_compact_contract" not in context.model_input


def test_context_builder_records_historical_tool_compression_count_in_auto_trigger(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    context = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        session_summary="\n".join(f"- user_request: turn {index}" for index in range(1, 8)),
        historical_tool_compression_count=2,
    ).build()

    manifest_path = writer.path / "contexts" / f"{context.context_id}-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["auto_compact_trigger"]["historical_tool_compression_count"] == 2
    assert "historical_tool_compression_present" in manifest["auto_compact_trigger"]["reasons"]


def test_context_builder_records_observation_compression_source_diagnostics(tmp_path: Path) -> None:
    writer = _make_writer(tmp_path)
    raw_output = "OBS-HEAD-" + ("body-" * 500) + "OBS-TAIL"

    context = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        observations=[
            {
                "tool_name": "shell",
                "args": {"command": "pytest"},
                "result": {"status": "success", "stdout": raw_output, "stderr": ""},
            },
        ],
    ).build()

    manifest_path = writer.path / "contexts" / f"{context.context_id}-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observations = manifest["source_diagnostics"]["observations"]

    assert observations["included_in_model_input"] is False
    assert observations["observation_section_count"] == 0
    assert observations["compacted_count"] == 1
    assert observations["truncated_count"] == 0
    assert observations["skipped_count"] == 0
    assert observations["original_chars"] > observations["final_chars"]
    assert observations["saved_chars"] == observations["original_chars"] - observations["final_chars"]
    assert "source_diagnostics" not in context.model_input
    assert "OBS-HEAD-" not in context.model_input


def test_context_builder_diagnostics_match_real_model_input(tmp_path: Path) -> None:
    project_instructions = "HEAD-" + ("middle-" * 700) + "TAIL"
    skipped_memory = "SKIPPED-MEMORY-" * 400
    writer = _make_writer(tmp_path)
    (tmp_path / "AGENTS.md").write_text(project_instructions, encoding="utf-8")

    compact_budget = ContextBudget(
        max_total_chars=90,
        max_section_chars=200,
        max_tool_observation_chars=1200,
        keep_recent_observations=4,
        collapse_head_chars=20,
        collapse_tail_chars=20,
    )
    context = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        session_summary="SESSION-KEPT",
        interaction_state=[
            {
                "type": "question",
                "tool": "request_user_input",
                "status": "answered",
                "question": "Continue?",
                "answer_excerpt": skipped_memory,
            },
        ],
        compaction_budget=compact_budget,
    ).build()

    assert "SKIPPED-MEMORY-" not in context.model_input
    assert project_instructions not in context.model_input
    bounded_project_instructions = project_instructions[:4000]
    assert bounded_project_instructions[:20] in context.model_input
    assert bounded_project_instructions[-20:] in context.model_input
    assert "...[collapsed " in context.model_input
    assert "SESSION-KEPT" in context.model_input
    assert "task_envelope" not in {record.key for record in context.diagnostics}


def test_prepare_initial_messages_derives_context_budget_from_model_metadata(tmp_path: Path) -> None:
    (tmp_path / "small").mkdir()
    (tmp_path / "large").mkdir()
    small_writer = _make_writer(tmp_path / "small")
    large_writer = _make_writer(tmp_path / "large")

    prepare_initial_messages(
        context_builder_cls=ContextBuilder,
        task=_task("small window"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        writer=small_writer,
        model_gateway=_GatewayWithContextWindow(32_000),
        session_summary=None,
        session_compaction=None,
        historical_tool_compression_count=0,
        working_state=None,
        interaction_resolver=_InteractionResolver(),
    )
    prepare_initial_messages(
        context_builder_cls=ContextBuilder,
        task=_task("large window"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        writer=large_writer,
        model_gateway=_GatewayWithContextWindow(256_000),
        session_summary=None,
        session_compaction=None,
        historical_tool_compression_count=0,
        working_state=None,
        interaction_resolver=_InteractionResolver(),
    )

    small_manifest = _latest_context_manifest(small_writer)
    large_manifest = _latest_context_manifest(large_writer)

    assert small_manifest["contexts"][0]["budget"]["max_tokens"] == 8_000
    assert large_manifest["contexts"][0]["budget"]["max_tokens"] > small_manifest["contexts"][0]["budget"]["max_tokens"]
    assert large_manifest["contexts"][0]["budget"]["max_chars"] > small_manifest["contexts"][0]["budget"]["max_chars"]


def test_context_builder_records_memory_source_diagnostics_when_memory_is_skipped(tmp_path: Path, monkeypatch) -> None:
    writer = _make_writer(tmp_path)
    memory_block = "\n".join(["Relevant Memory:", "- memory-id: skipped by context budget"])
    fake_result = _FakeMemoryResult(memory_block)
    monkeypatch.setattr(ContextBuilder, "_memory_result", lambda self: fake_result)

    compact_budget = ContextBudget(
        max_total_chars=10,
        max_section_chars=200,
        max_tool_observation_chars=1200,
        keep_recent_observations=4,
        collapse_head_chars=20,
        collapse_tail_chars=20,
    )
    context = ContextBuilder(
        task=_task("summarize project"),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
        compaction_budget=compact_budget,
    ).build()

    manifest_path = writer.path / "contexts" / f"{context.context_id}-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "Relevant Memory:" not in context.model_input
    assert any(record.key == "memory" and record.decision == "skipped" for record in context.diagnostics)
    assert "memory" in manifest
    assert manifest["source_diagnostics"]["memory"]["included_in_model_input"] is False


def test_context_builder_exposes_target_paths_as_task_facts(tmp_path: Path) -> None:
    target = tmp_path / "external-project"
    target.mkdir()
    writer = _make_writer(tmp_path)
    context = ContextBuilder(
        task=TaskSpec(
            goal=f'介绍 "{target}"',
            workspace_root=".",
            allowed_tools=["file_list", "file_read"],
            acceptance_criteria=[],
            verification_commands=[],
            constraints=[],
            target_paths=[str(target)],
            policy={"approval_allowed_tools": [], "approved_tools": []},
        ),
        workspace_root=tmp_path,
        provider_name="test-provider",
        episode_writer=writer,
    ).build()

    model_input = context.model_input
    assert "Target Paths:" in model_input
    assert str(target) in model_input
    assert "Start by listing the target path" in model_input


class _FakeMemoryResult:
    def __init__(self, model_block: str) -> None:
        self.memories = [object()]
        self._model_block = model_block

    def to_model_block(self) -> str:
        return self._model_block

    def to_manifest_dict(self) -> dict:
        return {
            "used_memories": [{"id": "memory-id"}],
            "budget": {"max_workspace_items": 6},
            "diagnostics": {
                "workspace_index_missing": 0,
                "user_index_missing": 0,
                "skipped_inactive": 0,
                "skipped_deleted": 0,
                "skipped_missing": 0,
                "skipped_invalid": 0,
                "skipped_over_budget": 2,
            },
        }


class _GatewayWithContextWindow:
    provider_name = "test-provider"

    def __init__(self, context_window_tokens: int) -> None:
        self._context_window_tokens = context_window_tokens

    def metadata(self):
        from haagent.models.types import ModelGatewayMetadata

        metadata = ModelGatewayMetadata(
            provider="test-provider",
            model="test-model",
            endpoint=None,
            base_url=None,
            profile_name=None,
        )
        object.__setattr__(metadata, "context_window_tokens", self._context_window_tokens)
        return metadata

    def generate(self, messages, tool_schemas):
        raise AssertionError("generate is not used")


class _InteractionResolver:
    def state_records(self) -> list[dict]:
        return []


def _latest_context_manifest(writer: EpisodeWriter) -> dict:
    return json.loads((writer.path / "context-manifest.json").read_text(encoding="utf-8"))


def _make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: test\n", encoding="utf-8")
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

