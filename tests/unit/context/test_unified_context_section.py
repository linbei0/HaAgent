from __future__ import annotations

from haagent.context.compression.sections import (
    ContextBudget,
    ContextSection as CompactionImportedSection,
    compact_context_sections,
)
from haagent.context.selection import ContextSelector, ContextSelectionBudget, compaction_sections_from_selection
from haagent.context.sources import ContextCandidate, ContextSection


def test_context_section_is_single_type() -> None:
    # 选择层与压缩层必须共用同一 ContextSection，禁止同名双类型。
    assert ContextSection is CompactionImportedSection


def test_selection_sections_are_directly_compactable() -> None:
    selection = ContextSelector(
        budget=ContextSelectionBudget(max_system_chars=10_000, max_task_chars=10_000, max_source_chars=10_000),
    ).select(
        [
            ContextCandidate(
                source_type="session_summary",
                source_id="session_summary",
                placement="system",
                title="Session Summary",
                content="keep me",
                reason="resumed_session",
                priority=30,
            ),
            ContextCandidate(
                source_type="memory",
                source_id="memory",
                placement="task",
                title="Relevant Memory",
                content="optional memory",
                reason="memory_retrieval_match",
                priority=50,
            ),
        ],
    )

    sections = compaction_sections_from_selection(selection)
    assert all(isinstance(section, ContextSection) for section in sections)
    assert {section.key for section in sections} == {"session_summary", "memory"}
    assert all(section.source_type is not None for section in sections)
    assert all(section.placement is not None for section in sections)

    # 适配函数不得再复制/改写字段：同一对象进入压缩。
    assert sections is not selection.system_sections  # 仍是拼接列表
    assert sections[0] is selection.system_sections[0]
    assert sections[1] is selection.task_sections[0]

    result = compact_context_sections(sections, ContextBudget(max_total_chars=100, max_section_chars=50))
    assert {section.key for section in result.sections} == {"session_summary", "memory"}
    assert all(record.decision == "selected" for record in result.diagnostics)
