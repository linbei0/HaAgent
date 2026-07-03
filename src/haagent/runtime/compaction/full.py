"""
src/src/haagent/runtime/compaction/full.py - full compact 执行器

在 full compact contract 允许时调用 ModelGateway 生成结构化 summary，并重建压缩后的消息窗口。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from haagent.models.gateway import ModelCallError, ModelGateway
from haagent.runtime.compaction.contract import (
    REQUIRED_FULL_COMPACT_SUMMARY_FIELDS,
    FullCompactEligibility,
    plan_full_compact_window,
    validate_full_compact_summary,
)


@dataclass(frozen=True)
class FullCompactResult:
    applied: bool
    reason: str
    pre_message_count: int
    post_message_count: int
    older_message_count: int
    preserved_recent_count: int
    summary_chars: int
    messages: list[dict[str, Any]]
    manifest: dict[str, Any]


def maybe_full_compact_messages(
    *,
    messages: list[dict[str, Any]],
    eligibility: FullCompactEligibility,
    gateway: ModelGateway,
    preserve_recent: int = 6,
) -> FullCompactResult:
    """按 full compact 契约尝试压缩 messages；失败时返回原消息拷贝。"""
    original_messages = copy.deepcopy(messages)
    pre_message_count = len(messages)
    if not eligibility.eligible:
        return _not_applied(
            reason=eligibility.reason,
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=0,
            preserved_recent_count=min(max(0, preserve_recent), pre_message_count),
        )

    plan = plan_full_compact_window(messages, preserve_recent=preserve_recent)
    if plan.blocked_reason is not None or not plan.tool_pair_safe:
        return _not_applied(
            reason=plan.blocked_reason or "tool_pair_boundary_unsafe",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
        )

    older_messages = copy.deepcopy(messages[: plan.preserved_start_index])
    preserved_recent = copy.deepcopy(messages[plan.preserved_start_index :])
    try:
        response = gateway.generate(
            messages=_build_summary_request_messages(older_messages),
            tool_schemas=[],
        )
    except ModelCallError as error:
        return _not_applied(
            reason="model_call_failed",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
            extra_manifest={"error": str(error)},
        )

    if response.tool_calls:
        return _not_applied(
            reason="summary_returned_tool_calls",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
        )
    if not response.content.strip():
        return _not_applied(
            reason="summary_empty",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
        )
    try:
        summary = json.loads(response.content)
    except json.JSONDecodeError as error:
        return _not_applied(
            reason="summary_json_invalid",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
            extra_manifest={"error": str(error)},
        )
    if not isinstance(summary, dict):
        return _not_applied(
            reason="summary_schema_not_object",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
        )

    schema_errors = validate_full_compact_summary(summary)
    if schema_errors:
        return _not_applied(
            reason="schema_invalid",
            original_messages=original_messages,
            pre_message_count=pre_message_count,
            older_message_count=plan.older_message_count,
            preserved_recent_count=plan.preserved_recent_count,
            extra_manifest={"schema_errors": schema_errors},
        )

    summary_text = _format_summary_message(summary)
    compacted_messages = [
        {
            "role": "user",
            "content": (
                "[full_compact_boundary "
                f"older_messages={plan.older_message_count} "
                f"preserved_recent={plan.preserved_recent_count}]"
            ),
        },
        {"role": "user", "content": summary_text},
        *preserved_recent,
    ]
    manifest = _manifest(
        applied=True,
        reason="applied",
        pre_message_count=pre_message_count,
        post_message_count=len(compacted_messages),
        older_message_count=plan.older_message_count,
        preserved_recent_count=plan.preserved_recent_count,
        summary_chars=len(summary_text),
    )
    return FullCompactResult(
        applied=True,
        reason="applied",
        pre_message_count=pre_message_count,
        post_message_count=len(compacted_messages),
        older_message_count=plan.older_message_count,
        preserved_recent_count=plan.preserved_recent_count,
        summary_chars=len(summary_text),
        messages=compacted_messages,
        manifest=manifest,
    )


def _build_summary_request_messages(older_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    fields = ", ".join(REQUIRED_FULL_COMPACT_SUMMARY_FIELDS)
    return [
        {
            "role": "system",
            "content": (
                "You are HaAgent full compact summarizer. Return only one JSON object, no markdown. "
                f"The JSON object must contain exactly these required fields: {fields}. "
                "task_focus must be a string. All other fields must be arrays of concise strings."
            ),
        },
        {
            "role": "user",
            "content": (
                "Summarize the older conversation messages below for future task continuity. "
                "Preserve concrete files, decisions, constraints, tool outcomes, verification, and risks.\n\n"
                + json.dumps(older_messages, ensure_ascii=False, indent=2)
            ),
        },
    ]


def _format_summary_message(summary: dict[str, Any]) -> str:
    lines = ["Full Compact Summary:", f"task_focus: {summary['task_focus']}"]
    for field in REQUIRED_FULL_COMPACT_SUMMARY_FIELDS:
        if field == "task_focus":
            continue
        lines.append(f"{field}:")
        values = summary[field]
        if not values:
            lines.append("- none")
            continue
        for value in values:
            lines.append(f"- {value}")
    return "\n".join(lines)


def _not_applied(
    *,
    reason: str,
    original_messages: list[dict[str, Any]],
    pre_message_count: int,
    older_message_count: int,
    preserved_recent_count: int,
    extra_manifest: dict[str, Any] | None = None,
) -> FullCompactResult:
    manifest = _manifest(
        applied=False,
        reason=reason,
        pre_message_count=pre_message_count,
        post_message_count=len(original_messages),
        older_message_count=older_message_count,
        preserved_recent_count=preserved_recent_count,
        summary_chars=0,
    )
    if extra_manifest:
        manifest.update(extra_manifest)
    return FullCompactResult(
        applied=False,
        reason=reason,
        pre_message_count=pre_message_count,
        post_message_count=len(original_messages),
        older_message_count=older_message_count,
        preserved_recent_count=preserved_recent_count,
        summary_chars=0,
        messages=copy.deepcopy(original_messages),
        manifest=manifest,
    )


def _manifest(
    *,
    applied: bool,
    reason: str,
    pre_message_count: int,
    post_message_count: int,
    older_message_count: int,
    preserved_recent_count: int,
    summary_chars: int,
) -> dict[str, Any]:
    return {
        "applied": applied,
        "reason": reason,
        "pre_message_count": pre_message_count,
        "post_message_count": post_message_count,
        "older_message_count": older_message_count,
        "preserved_recent_count": preserved_recent_count,
        "summary_chars": summary_chars,
    }
