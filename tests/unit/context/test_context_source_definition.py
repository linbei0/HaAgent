"""
tests/unit/context/test_context_source_definition.py - ContextSourceDefinition 注册表
"""

from __future__ import annotations

from dataclasses import replace

from haagent.context.selection import (
    ContextCandidateInputs,
    ContextSelector,
    collect_context_candidates,
    compaction_sections_from_selection,
)
from haagent.context.source_definitions import (
    CONTEXT_SOURCE_DEFINITIONS,
    ContextSourceDefinition,
    get_source_definition,
)


def test_definitions_cover_production_sources() -> None:
    ids = {item.id for item in CONTEXT_SOURCE_DEFINITIONS}
    assert ids == {
        "soul",
        "project_instructions",
        "prompt_pack",
        "session_summary",
        "working_state",
        "task_ledger",
        "memory_index",
        "memory",
        "interaction_history",
        "skills",
    }


def test_collect_uses_definition_metadata_without_string_maps() -> None:
    inputs = ContextCandidateInputs(
        project_instructions="Use uv.",
        session_summary="Earlier: fixed bug.",
        memory_block="- prefer dark theme",
    )
    candidates = collect_context_candidates(inputs)
    by_id = {item.source_id: item for item in candidates}
    assert by_id["project_instructions"].priority == 10
    assert by_id["session_summary"].priority == 30
    assert by_id["memory"].priority == 50

    selection = ContextSelector().select(candidates)
    sections = compaction_sections_from_selection(selection)
    by_key = {item.key: item for item in sections}
    assert by_key["project_instructions"].source == "project"
    assert by_key["session_summary"].source == "session"
    assert by_key["project_instructions"].priority == 80


def test_extra_definition_only_needs_definition_and_content(monkeypatch) -> None:
    """验收：新增测试 source 只加 definition + 内容字段，不改字符串映射。"""
    extra = ContextSourceDefinition(
        id="test_source",
        placement="task",
        title="Test Source",
        reason="test_only",
        selection_priority=5,
        compaction_priority=99,
        content_field="working_state",
    )
    monkeypatch.setattr(
        "haagent.context.selection.CONTEXT_SOURCE_DEFINITIONS",
        (*CONTEXT_SOURCE_DEFINITIONS, extra),
    )
    # get_source_definition 也要看到新定义
    monkeypatch.setattr(
        "haagent.context.selection.get_source_definition",
        lambda source_id: (
            extra
            if source_id == "test_source"
            else get_source_definition(source_id)
        ),
    )
    candidates = collect_context_candidates(
        ContextCandidateInputs(working_state="goal: ship phase 4")
    )
    assert any(item.source_id == "test_source" for item in candidates)
    selection = ContextSelector().select(candidates)
    sections = compaction_sections_from_selection(selection)
    test_section = next(item for item in sections if item.key == "test_source")
    assert test_section.priority == 99
    assert test_section.source == "test_source"
