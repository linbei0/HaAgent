"""
src/haagent/runtime/session/memory_compaction.py - 确定性会话记忆压缩

把较早的 turn summary 折叠成结构化 session memory，同时保留最近的原文摘要。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.context.compaction import collapse_text_head_tail


DEFAULT_PRESERVED_RECENT_TURNS = 6
SESSION_MEMORY_CHAR_LIMIT = 2000


@dataclass(frozen=True)
class SessionMemoryCompactionResult:
    summary_text: str | None
    diagnostics: dict[str, object]


def compact_session_memory(
    summaries: list[str],
    *,
    keep_recent: int = DEFAULT_PRESERVED_RECENT_TURNS,
    memory_char_limit: int = SESSION_MEMORY_CHAR_LIMIT,
) -> SessionMemoryCompactionResult:
    """确定性折叠较早摘要，不调用模型。"""
    original_text = "\n".join(summaries)
    original_chars = len(original_text)
    if not summaries:
        return SessionMemoryCompactionResult(
            summary_text=None,
            diagnostics={
                "decision": "not_needed",
                "original_turn_count": 0,
                "compacted_turn_count": 0,
                "preserved_recent_count": 0,
                "original_chars": 0,
                "final_chars": 0,
                "saved_chars": 0,
                "reason": "no_session_history",
            },
        )

    preserved_count = min(max(keep_recent, 0), len(summaries))
    if len(summaries) <= preserved_count and original_chars <= memory_char_limit:
        return SessionMemoryCompactionResult(
            summary_text=original_text,
            diagnostics={
                "decision": "not_needed",
                "original_turn_count": len(summaries),
                "compacted_turn_count": 0,
                "preserved_recent_count": len(summaries),
                "original_chars": original_chars,
                "final_chars": original_chars,
                "saved_chars": 0,
                "reason": "within_session_memory_budget",
            },
        )

    older = summaries[:-preserved_count] if preserved_count else list(summaries)
    recent = summaries[-preserved_count:] if preserved_count else []
    memory = _session_memory_summary(older)
    parts = [memory]
    if recent:
        parts.append("[recent_turn_summaries]")
        parts.extend(recent)
    final_text = "\n".join(parts)
    reason = "session_history_over_budget"
    if len(final_text) > memory_char_limit:
        recent_text = "\n".join(["[recent_turn_summaries]", *recent]) if recent else ""
        memory_budget = memory_char_limit - len(recent_text) - 1
        if memory_budget > 80:
            memory, _ = collapse_text_head_tail(
                memory,
                max_chars=memory_budget,
                head_chars=max(1, memory_budget // 2),
                tail_chars=max(0, memory_budget // 4),
            )
            final_text = "\n".join([memory, recent_text]) if recent_text else memory
        else:
            final_text, _ = collapse_text_head_tail(
                final_text,
                max_chars=memory_char_limit,
                head_chars=max(1, memory_char_limit // 2),
                tail_chars=max(0, memory_char_limit // 4),
            )
        reason = "session_history_over_budget_head_tail_collapsed"

    final_chars = len(final_text)
    return SessionMemoryCompactionResult(
        summary_text=final_text,
        diagnostics={
            "decision": "compacted",
            "original_turn_count": len(summaries),
            "compacted_turn_count": len(older),
            "preserved_recent_count": len(recent),
            "original_chars": original_chars,
            "final_chars": final_chars,
            "saved_chars": max(0, original_chars - final_chars),
            "reason": reason,
        },
    )


def _session_memory_summary(older_summaries: list[str]) -> str:
    status_counts = _count_field_values(older_summaries, "status")
    verification_counts = _count_field_values(older_summaries, "verification")
    recent_goals = _recent_field_values(older_summaries, "- user_request", limit=3)
    episode_paths = _recent_field_values(older_summaries, "episode_path", limit=3)
    lines = [
        f"[session_memory_compacted {len(older_summaries)} earlier turns]",
        f"total_turns: {len(older_summaries)}",
        "status_counts:",
        *_format_counts(status_counts),
        "recent_prior_user_goals:",
        *[f"- {goal}" for goal in recent_goals],
        "recent_prior_episode_paths:",
        *[f"- {path}" for path in episode_paths],
        "verification_counts:",
        *_format_counts(verification_counts),
    ]
    return "\n".join(lines)


def _count_field_values(summaries: list[str], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for summary in summaries:
        value = _first_field_value(summary, field)
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return counts


def _recent_field_values(summaries: list[str], field: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for summary in reversed(summaries):
        value = _first_field_value(summary, field)
        if not value:
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return list(reversed(values))


def _first_field_value(summary: str, field: str) -> str:
    prefix = f"{field}:"
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _format_counts(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["- none: 0"]
    return [f"- {name}: {count}" for name, count in sorted(counts.items())]
