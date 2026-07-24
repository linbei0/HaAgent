"""
src/haagent/context/compression/session_memory.py - 会话记忆确定性压缩

折叠较早 turn summary，并把最近若干轮完整问答原文作为模型可见会话记忆。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haagent.context.compression.budget import CompressionBudget, derive_compression_budget
from haagent.context.compression.sections import collapse_text_head_tail

DEFAULT_PRESERVED_RECENT_TURNS = 6
# 会话记忆预算：只在此一处做整轮丢弃截断，builder 不再叠加更小的硬上限。
# 12000 字约覆盖最近 6 轮中等长度问答；超出时按整轮丢弃最旧完整块。
SESSION_MEMORY_CHAR_LIMIT = 12000


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
    recent_turns: list[dict[str, Any]] | None = None,
) -> SessionMemoryCompactionResult:
    """确定性折叠较早摘要，不调用模型。

    recent_turns 提供最近若干轮的完整问答原文；优先于截断摘要进入模型输入，
    只有整体超出 memory_char_limit 时才按整轮丢弃最旧的完整块。
    """
    limit = memory_char_limit if memory_char_limit is not None else _memory_char_limit(budget)
    full_recent = _render_recent_full_turns(recent_turns, keep_recent)
    original_text = "\n".join(summaries)
    original_chars = len(original_text) + len(full_recent)

    if not summaries and not full_recent:
        return SessionMemoryCompactionResult(
            summary_text=None,
            diagnostics={
                "decision": "empty",
                "original_turn_count": 0,
                "compacted_turn_count": 0,
                "preserved_recent_count": 0,
                "recent_full_turns": 0,
                "dropped_recent_full_turns": 0,
                "original_chars": 0,
                "final_chars": 0,
                "saved_chars": 0,
                "reason": "no_session_history",
            },
        )

    if not summaries:
        fitted_recent, dropped = _fit_recent_turns(recent_turns or [], keep_recent, limit)
        recent_block = _render_recent_full_turns(fitted_recent, keep_recent)
        return SessionMemoryCompactionResult(
            summary_text=recent_block or None,
            diagnostics={
                "decision": "kept" if dropped == 0 else "compacted",
                "original_turn_count": len(recent_turns or []),
                "compacted_turn_count": 0,
                "preserved_recent_count": len(fitted_recent),
                "recent_full_turns": len(fitted_recent),
                "dropped_recent_full_turns": dropped,
                "original_chars": original_chars,
                "final_chars": len(recent_block),
                "saved_chars": max(0, original_chars - len(recent_block)),
                "reason": "within_budget" if dropped == 0 else "recent_turns_over_budget",
            },
        )

    if len(summaries) <= keep_recent and (len(original_text) + len(full_recent)) <= limit:
        combined = "\n".join(part for part in [original_text, full_recent] if part)
        return SessionMemoryCompactionResult(
            summary_text=combined,
            diagnostics={
                "decision": "kept",
                "original_turn_count": len(summaries),
                "compacted_turn_count": 0,
                "preserved_recent_count": len(summaries),
                "recent_full_turns": len(recent_turns or []),
                "dropped_recent_full_turns": 0,
                "original_chars": original_chars,
                "final_chars": len(combined),
                "saved_chars": 0,
                "reason": "within_budget",
            },
        )

    older = summaries[:-keep_recent] if keep_recent > 0 else list(summaries)
    recent_summaries = summaries[-keep_recent:] if keep_recent > 0 else []
    memory = _session_memory_summary(older)
    # 预算不足时按整轮丢弃最旧的完整块，保最新轮，不从回答中间截断。
    fitted_recent, dropped = _fit_recent_turns(recent_turns or [], keep_recent, limit - len(memory))
    recent_block = _render_recent_full_turns(fitted_recent, keep_recent)
    if not recent_block:
        recent_block = "\n".join(recent_summaries)
    summary_parts = [memory, recent_block] if memory else [recent_block]
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
            "preserved_recent_count": len(fitted_recent) if recent_turns else len(recent_summaries),
            "recent_full_turns": len(fitted_recent),
            "dropped_recent_full_turns": dropped,
            "original_chars": original_chars,
            "final_chars": len(summary_text),
            "saved_chars": max(0, original_chars - len(summary_text)),
            "reason": "session_history_over_budget",
        },
    )


def _render_recent_full_turns(recent_turns: list[dict[str, Any]] | None, keep_recent: int) -> str:
    """把最近若干轮完整问答渲染成模型可见文本；为空返回空串。"""
    if not recent_turns:
        return ""
    selected = recent_turns[-keep_recent:] if keep_recent > 0 else []
    blocks: list[str] = []
    for turn in selected:
        user = str(turn.get("request") or turn.get("user") or "").strip()
        assistant = str(
            turn.get("assistant_display_text") or turn.get("assistant") or "",
        ).strip()
        lines: list[str] = []
        if user:
            lines.append(f"user: {user}")
        if assistant:
            lines.append(f"assistant: {assistant}")
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _fit_recent_turns(
    recent_turns: list[dict[str, Any]],
    keep_recent: int,
    char_budget: int,
) -> tuple[list[dict[str, Any]], int]:
    """预算内按整轮保留最新完整块；返回 (保留列表, 丢弃块数)。"""
    if not recent_turns or char_budget <= 0:
        return [], len(recent_turns or [])
    candidates = recent_turns[-keep_recent:] if keep_recent > 0 else []
    kept: list[dict[str, Any]] = []
    running = 0
    dropped = 0
    for turn in reversed(candidates):
        block_len = len(_render_recent_full_turns([turn], keep_recent=1))
        if running + block_len > char_budget and kept:
            dropped += 1
            continue
        if running + block_len > char_budget and not kept:
            # 单块也超预算：仍保留最新一轮，交由后续 head/tail 兜底。
            kept.insert(0, turn)
            running += block_len
            continue
        kept.insert(0, turn)
        running += block_len
    dropped += len(candidates) - len(kept) - dropped
    return kept, max(0, len(candidates) - len(kept))


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
