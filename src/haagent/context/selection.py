"""
haagent/context/selection.py - 上下文选择 Module

把可用上下文候选转换为模型可见 section 和可审计 selection manifest。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from haagent.context.compression.sections import ContextBudget, collapse_text_head_tail
from haagent.context.source_definitions import (
    CONTEXT_SOURCE_DEFINITIONS,
    ContextSourceDefinition,
    get_source_definition,
)
from haagent.context.sources import ContextCandidate, ContextDecision, ContextSection



@dataclass(frozen=True)
class ContextSelectionBudget:
    max_system_chars: int = 9000
    max_task_chars: int = 12000
    max_source_chars: int = 4000
    collapse_head_chars: int = 600
    collapse_tail_chars: int = 300

    def to_dict(self) -> dict[str, int]:
        return {
            "max_system_chars": self.max_system_chars,
            "max_task_chars": self.max_task_chars,
            "max_source_chars": self.max_source_chars,
            "collapse_head_chars": self.collapse_head_chars,
            "collapse_tail_chars": self.collapse_tail_chars,
        }


@dataclass(frozen=True)
class ContextSelectionResult:
    system_sections: list[ContextSection]
    task_sections: list[ContextSection]
    selected: list[ContextDecision]
    skipped: list[ContextDecision]
    budget: ContextSelectionBudget
    used_system_chars: int
    used_task_chars: int

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "budget": {
                **self.budget.to_dict(),
                "used_system_chars": self.used_system_chars,
                "used_task_chars": self.used_task_chars,
            },
            "selected": [decision.to_dict() for decision in self.selected],
            "skipped": [decision.to_dict() for decision in self.skipped],
        }


@dataclass(frozen=True)
class ContextSelector:
    budget: ContextSelectionBudget = field(default_factory=ContextSelectionBudget)

    def select(self, candidates: list[ContextCandidate]) -> ContextSelectionResult:
        prepared = [_prepare_candidate(candidate, self.budget) for candidate in candidates]
        used = {"system": 0, "task": 0}
        limits = {"system": self.budget.max_system_chars, "task": self.budget.max_task_chars}
        selected: list[ContextDecision] = []
        skipped: list[ContextDecision] = []
        system_sections: list[ContextSection] = []
        task_sections: list[ContextSection] = []

        for candidate in sorted(prepared, key=lambda item: (item.priority, item.source_type, item.source_id)):
            content = candidate.content.strip()
            chars = len(content)
            placement = candidate.placement
            if not content:
                skipped.append(
                    _decision(
                        candidate,
                        chars=0,
                        selected=False,
                        skip_reason=candidate.skip_reason or "empty",
                    ),
                )
                continue
            if not candidate.hard_required and used[placement] + chars > limits[placement]:
                skipped.append(_decision(candidate, chars=chars, selected=False, skip_reason="over_budget"))
                continue
            used[placement] += chars
            decision = _decision(candidate, chars=chars, selected=True, skip_reason=None)
            selected.append(decision)
            # 直接产出压缩层可消费的统一 ContextSection，避免二次字段映射。
            definition = get_source_definition(candidate.source_type)
            section = ContextSection(
                key=candidate.source_id,
                title=candidate.title,
                content=content,
                source=(
                    definition.resolved_compaction_source()
                    if definition is not None
                    else candidate.source_type
                ),
                priority=(
                    definition.compaction_priority if definition is not None else 10
                ),
                kind=(
                    definition.resolved_compaction_kind()
                    if definition is not None
                    else candidate.source_type
                ),
                hard_required=candidate.hard_required,
                source_type=candidate.source_type,
                source_id=candidate.source_id,
                placement=placement,
                chars=chars,
            )
            if placement == "system":
                system_sections.append(section)
            else:
                task_sections.append(section)

        return ContextSelectionResult(
            system_sections=system_sections,
            task_sections=task_sections,
            selected=selected,
            skipped=skipped,
            budget=self.budget,
            used_system_chars=used["system"],
            used_task_chars=used["task"],
        )


@dataclass(frozen=True)
class ContextCandidateInputs:
    soul: str | None = None
    soul_skip_reason: str | None = None
    soul_metadata: dict[str, Any] = field(default_factory=dict)
    project_instructions: str | None = None
    prompt_packs: str | None = None
    prompt_pack_metadata: dict[str, Any] = field(default_factory=dict)
    session_summary: str | None = None
    working_state: str | None = None
    task_ledger: str | None = None
    memory_index: str | None = None
    memory_index_skip_reason: str | None = None
    memory_index_metadata: dict[str, Any] = field(default_factory=dict)
    memory_block: str | None = None
    interaction_state: str | None = None
    skills_block: str | None = None


def collect_context_candidates(inputs: ContextCandidateInputs) -> list[ContextCandidate]:
    """按固定 ContextSourceDefinition 列表收集候选；新增 source 只加 definition。"""
    candidates: list[ContextCandidate] = []
    for definition in CONTEXT_SOURCE_DEFINITIONS:
        content = getattr(inputs, definition.content_field, None)
        skip_reason = (
            getattr(inputs, definition.skip_reason_field, None)
            if definition.skip_reason_field
            else None
        )
        metadata = (
            getattr(inputs, definition.metadata_field, None)
            if definition.metadata_field
            else None
        )
        include_empty = definition.include_empty_on_skip and skip_reason is not None
        _append_candidate(
            candidates,
            definition=definition,
            content=content if isinstance(content, str) else None,
            include_empty=include_empty,
            skip_reason=skip_reason if isinstance(skip_reason, str) else None,
            metadata=metadata if isinstance(metadata, dict) else None,
        )
    return candidates


def selection_budget_for_initial_audit(_budget: ContextBudget) -> ContextSelectionBudget:
    return ContextSelectionBudget(
        max_system_chars=1_000_000,
        max_task_chars=1_000_000,
        max_source_chars=1_000_000,
    )


def compaction_sections_from_selection(selection: ContextSelectionResult) -> list[ContextSection]:
    """拼接已选 section；选择阶段已填齐压缩字段，不再复制改写。"""
    return [*selection.system_sections, *selection.task_sections]



def _prepare_candidate(candidate: ContextCandidate, budget: ContextSelectionBudget) -> ContextCandidate:
    content = candidate.content.strip()
    if len(content) <= budget.max_source_chars or candidate.hard_required:
        return replace(candidate, content=content)
    head_chars = min(budget.collapse_head_chars, max(1, budget.max_source_chars // 2))
    tail_chars = min(budget.collapse_tail_chars, max(0, budget.max_source_chars - head_chars))
    collapsed, collapsed_chars = collapse_text_head_tail(
        content,
        max_chars=budget.max_source_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )
    metadata = {
        **candidate.metadata,
        "truncated": collapsed_chars > 0,
        "original_chars": len(content),
    }
    return replace(candidate, content=collapsed, metadata=metadata)


def _append_candidate(
    candidates: list[ContextCandidate],
    *,
    definition: ContextSourceDefinition,
    content: str | None,
    include_empty: bool = False,
    skip_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if include_empty or (content and content.strip()):
        candidates.append(
            ContextCandidate(
                source_type=definition.source_type,
                source_id=definition.id,
                placement=definition.placement,
                title=definition.title,
                content=(content or "").strip(),
                reason=definition.reason,
                priority=definition.selection_priority,
                hard_required=definition.hard_required,
                skip_reason=skip_reason,
                metadata=dict(metadata or {}),
            ),
        )


def _decision(
    candidate: ContextCandidate,
    *,
    chars: int,
    selected: bool,
    skip_reason: str | None,
) -> ContextDecision:
    return ContextDecision(
        source_type=candidate.source_type,
        source_id=candidate.source_id,
        title=candidate.title,
        reason=candidate.reason,
        placement=candidate.placement,
        priority=candidate.priority,
        chars=chars,
        selected=selected,
        skip_reason=skip_reason,
        metadata=dict(candidate.metadata),
    )
