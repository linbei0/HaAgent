"""
src/haagent/context/compaction.py - 确定性上下文压缩

在模型调用前按字符预算选择、折叠上下文 section，并返回机器可读诊断。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


ContextDecision = Literal["selected", "skipped", "collapsed"]


@dataclass(frozen=True)
class ContextBudget:
    max_total_chars: int = 12000
    max_section_chars: int = 4000
    max_tool_observation_chars: int = 1200
    keep_recent_observations: int = 4
    collapse_head_chars: int = 600
    collapse_tail_chars: int = 300


@dataclass(frozen=True)
class ContextSection:
    key: str
    title: str
    content: str
    source: str
    priority: int
    kind: str
    recent_rank: int | None = None
    hard_required: bool = False


@dataclass(frozen=True)
class ContextSelectionRecord:
    key: str
    source: str
    kind: str
    decision: ContextDecision
    reason: str
    original_chars: int
    final_chars: int
    priority: int


@dataclass(frozen=True)
class ContextCompactionResult:
    sections: list[ContextSection]
    diagnostics: list[ContextSelectionRecord]
    original_chars: int
    final_chars: int


def assess_compact_readiness(
    compaction: ContextCompactionResult,
    budget: ContextBudget = ContextBudget(),
) -> dict:
    """基于确定性压缩结果评估是否需要关注 full compact 候选。"""
    skipped_count = sum(1 for record in compaction.diagnostics if record.decision == "skipped")
    collapsed_count = sum(1 for record in compaction.diagnostics if record.decision == "collapsed")
    budget_pressure = _ratio(compaction.final_chars, budget.max_total_chars)
    saved_ratio = _ratio(compaction.original_chars - compaction.final_chars, compaction.original_chars)

    reasons: list[str] = []
    if budget_pressure >= 0.8:
        reasons.append("near_budget_limit")
    if skipped_count:
        reasons.append("skipped_context_present")
    if collapsed_count:
        reasons.append("collapsed_context_present")

    if budget_pressure >= 0.9 and (skipped_count or collapsed_count):
        status = "full_compact_candidate"
        recommendation = "evaluate_full_compact"
    elif reasons:
        status = "watch"
        recommendation = "keep_deterministic"
    else:
        status = "deterministic_sufficient"
        recommendation = "keep_deterministic"
        reasons = ["within_budget_after_compaction"]

    return {
        "status": status,
        "budget_pressure": budget_pressure,
        "saved_ratio": saved_ratio,
        "skipped_count": skipped_count,
        "collapsed_count": collapsed_count,
        "recommendation": recommendation,
        "reasons": reasons,
    }


def assess_auto_compact_trigger(
    *,
    compact_readiness: dict,
    compaction: ContextCompactionResult,
    budget: ContextBudget = ContextBudget(),
    tool_result_microcompact_count: int = 0,
    session_summary_count: int = 0,
    session_summary_chars: int = 0,
) -> dict:
    """根据结构化上下文压力决定是否触发确定性 session memory 压缩。"""
    skipped_count = sum(1 for record in compaction.diagnostics if record.decision == "skipped")
    collapsed_count = sum(1 for record in compaction.diagnostics if record.decision == "collapsed")
    budget_pressure = _ratio(compaction.final_chars, budget.max_total_chars)
    session_history_over_budget = (
        session_summary_count > 6
        or session_summary_chars > 1000
        or any(record.key == "session_summary" and record.decision != "selected" for record in compaction.diagnostics)
    )

    reasons: list[str] = []
    if budget_pressure >= 0.8:
        reasons.append("near_budget_limit")
    if skipped_count:
        reasons.append("skipped_context_present")
    if collapsed_count:
        reasons.append("collapsed_context_present")
    if tool_result_microcompact_count:
        reasons.append("tool_result_microcompact_present")
    if session_history_over_budget:
        reasons.append("session_history_over_budget")

    triggered = session_history_over_budget and (
        budget_pressure >= 0.9
        or skipped_count > 0
        or collapsed_count > 0
        or tool_result_microcompact_count > 0
        or session_summary_chars > 1000
    )
    if triggered:
        status = "triggered"
        recommendation = "apply_session_memory_compaction"
        trigger_kind = "session_memory"
    elif reasons:
        status = "watch"
        recommendation = "keep_deterministic"
        trigger_kind = None
    else:
        status = "not_needed"
        recommendation = "keep_deterministic"
        trigger_kind = None
        reasons = list(compact_readiness.get("reasons") or ["within_budget_after_compaction"])

    return {
        "triggered": triggered,
        "trigger_kind": trigger_kind,
        "status": status,
        "recommendation": recommendation,
        "reasons": reasons,
        "budget_pressure": budget_pressure,
        "skipped_count": skipped_count,
        "collapsed_count": collapsed_count,
        "tool_result_microcompact_count": max(0, tool_result_microcompact_count),
        "session_summary_count": max(0, session_summary_count),
        "session_summary_chars": max(0, session_summary_chars),
    }


def compact_context_sections(
    sections: list[ContextSection],
    budget: ContextBudget = ContextBudget(),
) -> ContextCompactionResult:
    """按 section 和总字符预算确定性选择上下文。"""
    candidates = [_candidate(index, section, budget) for index, section in enumerate(sections)]
    selected_indices: set[int] = set()
    skipped_indices: set[int] = set()
    running_chars = 0

    for index, _section, content, _decision, _reason in sorted(candidates, key=_selection_key):
        final_chars = len(content)
        if running_chars + final_chars <= budget.max_total_chars:
            selected_indices.add(index)
            running_chars += final_chars
        else:
            skipped_indices.add(index)

    output_sections: list[ContextSection] = []
    diagnostics: list[ContextSelectionRecord] = []
    for index, section, content, candidate_decision, candidate_reason in candidates:
        original_chars = len(section.content)
        if index in skipped_indices:
            diagnostics.append(
                ContextSelectionRecord(
                    key=section.key,
                    source=section.source,
                    kind=section.kind,
                    decision="skipped",
                    reason="over_total_budget",
                    original_chars=original_chars,
                    final_chars=0,
                    priority=section.priority,
                ),
            )
            continue
        output_sections.append(replace(section, content=content))
        diagnostics.append(
            ContextSelectionRecord(
                key=section.key,
                source=section.source,
                kind=section.kind,
                decision=candidate_decision,
                reason=candidate_reason,
                original_chars=original_chars,
                final_chars=len(content),
                priority=section.priority,
            ),
        )

    return ContextCompactionResult(
        sections=output_sections,
        diagnostics=diagnostics,
        original_chars=sum(len(section.content) for section in sections),
        final_chars=sum(len(section.content) for section in output_sections),
    )


def collapse_text_head_tail(
    text: str,
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> tuple[str, int]:
    """保留头尾并显式标记中间被折叠的字符数。"""
    if len(text) <= max_chars:
        return text, 0
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip() if tail_chars > 0 else ""
    collapsed_chars = len(text) - head_chars - tail_chars
    marker = f"...[collapsed {collapsed_chars} chars]..."
    if tail:
        return f"{head}\n{marker}\n{tail}", collapsed_chars
    return f"{head}\n{marker}", collapsed_chars


def _candidate(
    index: int,
    section: ContextSection,
    budget: ContextBudget,
) -> tuple[int, ContextSection, str, ContextDecision, str]:
    content = section.content
    decision: ContextDecision = "selected"
    reason = "within_budget"

    if not section.hard_required and _should_collapse_old_observation(section, budget):
        content, collapsed_chars = collapse_text_head_tail(
            content,
            max_chars=budget.max_tool_observation_chars,
            head_chars=budget.collapse_head_chars,
            tail_chars=budget.collapse_tail_chars,
        )
        if collapsed_chars > 0:
            decision = "collapsed"
            reason = "old_observation_over_budget"

    if not section.hard_required and len(content) > budget.max_section_chars:
        content, collapsed_chars = collapse_text_head_tail(
            content,
            max_chars=budget.max_section_chars,
            head_chars=budget.collapse_head_chars,
            tail_chars=budget.collapse_tail_chars,
        )
        if collapsed_chars > 0:
            decision = "collapsed"
            reason = "section_over_budget"

    return index, section, content, decision, reason


def _should_collapse_old_observation(section: ContextSection, budget: ContextBudget) -> bool:
    return (
        section.kind == "observation"
        and section.recent_rank is not None
        and section.recent_rank >= budget.keep_recent_observations
        and len(section.content) > budget.max_tool_observation_chars
    )


def _selection_key(candidate: tuple[int, ContextSection, str, ContextDecision, str]) -> tuple[int, int, int]:
    index, section, _content, _decision, _reason = candidate
    recent_rank = section.recent_rank if section.recent_rank is not None else 1_000_000
    return (-section.priority, recent_rank, index)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)
