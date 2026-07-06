"""
src/haagent/context/compression/session_memory.py - 会话记忆确定性压缩

折叠较早 turn summary，保留最近若干轮作为模型可见会话记忆。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.context.compression.budget import CompressionBudget, derive_compression_budget
from haagent.context.compression.sections import collapse_text_head_tail

DEFAULT_PRESERVED_RECENT_TURNS = 6
SESSION_MEMORY_CHAR_LIMIT = 2000


@dataclass(frozen=True)
class SessionMemoryCompactionResult:
    summary_text: str | None
    diagnostics: dict


def compact_session_memory(
    summaries: list[str],
    *,
    budget: CompressionBudget | None = None,
    keep_recent: int = DEFAULT_PRESERVED_RECENT_TURNS,
    memory_char_limit: int | None = None,
) -> SessionMemoryCompactionResult:
    """确定性折叠较早摘要，不调用模型。"""
    limit = memory_char_limit if memory_char_limit is not None else _memory_char_limit(budget)
    if not summaries:
        return SessionMemoryCompactionResult(
            summary_text=None,
            diagnostics={
                "decision": "empty",
                "original_turn_count": 0,
                "compacted_turn_count": 0,
                "preserved_recent_count": 0,
                "original_chars": 0,
                "final_chars": 0,
                "saved_chars": 0,
                "reason": "no_session_history",
            },
        )

    original_text = "\n".join(summaries)
    if len(summaries) <= keep_recent and len(original_text) <= limit:
        return SessionMemoryCompactionResult(
            summary_text=original_text,
            diagnostics={
                "decision": "kept",
                "original_turn_count": len(summaries),
                "compacted_turn_count": 0,
                "preserved_recent_count": len(summaries),
                "original_chars": len(original_text),
                "final_chars": len(original_text),
                "saved_chars": 0,
                "reason": "within_budget",
            },
        )

    older = summaries[:-keep_recent] if keep_recent > 0 else list(summaries)
    recent = summaries[-keep_recent:] if keep_recent > 0 else []
    memory = _session_memory_summary(older)
    summary_parts = [memory, *recent] if memory else recent
    summary_text = "\n".join(part for part in summary_parts if part)
    if len(summary_text) > limit:
        summary_text, _collapsed = collapse_text_head_tail(
            summary_text,
            max_chars=limit,
            head_chars=min(900, max(1, limit // 2)),
            tail_chars=min(500, max(0, limit // 3)),
        )

    return SessionMemoryCompactionResult(
        summary_text=summary_text,
        diagnostics={
            "decision": "compacted",
            "original_turn_count": len(summaries),
            "compacted_turn_count": len(older),
            "preserved_recent_count": len(recent),
            "original_chars": len(original_text),
            "final_chars": len(summary_text),
            "saved_chars": max(0, len(original_text) - len(summary_text)),
            "reason": "session_history_over_budget",
        },
    )


def _memory_char_limit(budget: CompressionBudget | None) -> int:
    active_budget = budget or derive_compression_budget(None)
    return max(SESSION_MEMORY_CHAR_LIMIT, min(12_000, active_budget.context_builder_max_tokens // 2))


def _session_memory_summary(older_summaries: list[str]) -> str:
    if not older_summaries:
        return ""
    statuses = _count_field_values(older_summaries, "status")
    verification = _count_field_values(older_summaries, "verification")
    lines = [f"[session_memory_compacted {len(older_summaries)} earlier turns]"]
    lines.append(f"total_turns: {len(older_summaries)}")
    lines.extend(_format_counts(statuses))
    if verification:
        lines.append("verification:")
        lines.extend(f"- {key}: {count}" for key, count in sorted(verification.items()))
    recent_requests = _recent_field_values(older_summaries, "user_request", limit=3)
    if recent_requests:
        lines.append("recent_compacted_requests:")
        lines.extend(f"- {request}" for request in recent_requests)
    return "\n".join(lines)


def _count_field_values(summaries: list[str], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary in summaries:
        value = _first_field_value(summary, field)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


def _recent_field_values(summaries: list[str], field: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for summary in reversed(summaries):
        value = _first_field_value(summary, field)
        if value:
            values.append(value)
        if len(values) >= limit:
            break
    return list(reversed(values))


def _first_field_value(summary: str, field: str) -> str:
    prefix = f"- {field}:"
    nested_prefix = f"  {field}:"
    for line in summary.splitlines():
        stripped = line.rstrip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
        if stripped.startswith(nested_prefix):
            return stripped[len(nested_prefix) :].strip()
    return ""


def _format_counts(counts: dict[str, int]) -> list[str]:
    return [f"{key}: {count}" for key, count in sorted(counts.items())]
