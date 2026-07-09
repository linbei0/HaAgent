"""
haagent/context/selection.py - 上下文选择 Module

把可用上下文候选转换为模型可见 section 和可审计 selection manifest。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from haagent.context.compression.sections import ContextBudget, collapse_text_head_tail
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
            section = ContextSection(
                key=candidate.source_id,
                title=candidate.title,
                content=content,
                source=_compaction_source(candidate.source_type),
                priority=_compaction_priority(candidate.source_type),
                kind=_compaction_kind(candidate.source_type),
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
    candidates: list[ContextCandidate] = []
    _append_candidate(
        candidates,
        source_type="project_instructions",
        source_id="project_instructions",
        placement="system",
        title="Project Instructions",
        content=inputs.project_instructions,
        reason="workspace_agents_md_found",
        priority=10,
    )
    _append_candidate(
        candidates,
        source_type="prompt_pack",
        source_id="prompt_pack",
        placement="system",
        title="Prompt Packs",
        content=inputs.prompt_packs,
        reason="explicit_prompt_command",
        priority=15,
        metadata=inputs.prompt_pack_metadata,
        hard_required=True,
    )
    _append_candidate(
        candidates,
        source_type="session_summary",
        source_id="session_summary",
        placement="system",
        title="Session Summary",
        content=inputs.session_summary,
        reason="resumed_session",
        priority=30,
    )
    _append_candidate(
        candidates,
        source_type="working_state",
        source_id="working_state",
        placement="task",
        title="Working State",
        content=inputs.working_state,
        reason="working_state_present",
        priority=20,
    )
    _append_candidate(
        candidates,
        source_type="task_ledger",
        source_id="task_ledger",
        placement="task",
        title="Task Ledger",
        content=inputs.task_ledger,
        reason="task_ledger_present",
        priority=18,
    )
    _append_candidate(
        candidates,
        source_type="memory_index",
        source_id="memory_index",
        placement="task",
        title="Memory/SOP Navigation Index",
        content=inputs.memory_index,
        reason="confirmed_memory_index_available",
        priority=45,
        include_empty=inputs.memory_index_skip_reason is not None,
        skip_reason=inputs.memory_index_skip_reason,
        metadata=inputs.memory_index_metadata,
    )
    _append_candidate(
        candidates,
        source_type="memory",
        source_id="memory",
        placement="task",
        title="Relevant Memory",
        content=inputs.memory_block,
        reason="memory_retrieval_match",
        priority=50,
    )
    _append_candidate(
        candidates,
        source_type="interaction_history",
        source_id="interaction_history",
        placement="task",
        title="Interaction History",
        content=inputs.interaction_state,
        reason="recent_interaction_state",
        priority=40,
    )
    _append_candidate(
        candidates,
        source_type="skills",
        source_id="skills",
        placement="system",
        title="Available Skills",
        content=inputs.skills_block,
        reason="allowed_skill_tools_present",
        priority=60,
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
    source_type: str,
    source_id: str,
    placement: str,
    title: str,
    content: str | None,
    reason: str,
    priority: int,
    include_empty: bool = False,
    skip_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    hard_required: bool = False,
) -> None:
    if include_empty or (content and content.strip()):
        candidates.append(
            ContextCandidate(
                source_type=source_type,
                source_id=source_id,
                placement=placement,  # type: ignore[arg-type]
                title=title,
                content=(content or "").strip(),
                reason=reason,
                priority=priority,
                hard_required=hard_required,
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


def _compaction_source(source_type: str) -> str:
    if source_type == "project_instructions":
        return "project"
    if source_type == "session_summary":
        return "session"
    if source_type == "interaction_history":
        return "interaction_state"
    if source_type == "memory_index":
        return "memory"
    return source_type


def _compaction_kind(source_type: str) -> str:
    if source_type == "interaction_history":
        return "interaction"
    if source_type == "memory_index":
        return "memory_index"
    return source_type


def _compaction_priority(source_type: str) -> int:
    return {
        "prompt_pack": 85,
        "project_instructions": 80,
        "task_ledger": 78,
        "working_state": 75,
        "session_summary": 70,
        "memory_index": 65,
        "memory": 60,
        "interaction_history": 55,
        "skills": 50,
    }.get(source_type, 10)
