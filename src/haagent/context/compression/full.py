"""
src/haagent/context/compression/full.py - 统一 full compact 流水线

在确定性压缩后仍高压时，通过 ModelGateway 生成结构化摘要并保留最近消息。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from haagent.context.compression.budget import CompressionBudget, estimate_message_tokens
from haagent.models.types import ModelCallError, ModelGateway

DEFAULT_FULL_COMPACT_PRESERVE_RECENT = 6
REQUIRED_FULL_COMPACT_SUMMARY_FIELDS = (
    "task_focus",
    "completed_work",
    "open_issues",
    "important_files",
    "tool_results",
    "constraints",
    "verification",
    "risks",
)


@dataclass(frozen=True)
class FullCompactEligibility:
    eligible: bool
    reason: str
    trigger_kind: str | None
    required_preserve_recent: int

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "trigger_kind": self.trigger_kind,
            "required_preserve_recent": self.required_preserve_recent,
        }


@dataclass(frozen=True)
class FullCompactPlan:
    older_message_count: int
    preserved_recent_count: int
    tool_pair_safe: bool
    blocked_reason: str | None
    preserved_start_index: int


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


def assess_full_compact_need(
    messages: list[dict[str, Any]],
    budget: CompressionBudget,
    diagnostics: list[object],
) -> FullCompactEligibility:
    del diagnostics
    message_tokens = estimate_message_tokens(messages)
    if message_tokens < int(budget.available_input_tokens * 0.90):
        return FullCompactEligibility(
            eligible=False,
            reason="deterministic_context_sufficient",
            trigger_kind=None,
            required_preserve_recent=budget.full_compact_preserve_recent,
        )
    if len(messages) <= budget.full_compact_preserve_recent:
        return FullCompactEligibility(
            eligible=False,
            reason="insufficient_compressible_history",
            trigger_kind=None,
            required_preserve_recent=budget.full_compact_preserve_recent,
        )
    return FullCompactEligibility(
        eligible=True,
        reason="high_pressure_after_deterministic_compression",
        trigger_kind="auto_full_compact",
        required_preserve_recent=budget.full_compact_preserve_recent,
    )


def assess_full_compact_eligibility(
    *,
    auto_compact_trigger: dict[str, Any] | None,
    compact_readiness: dict[str, Any] | None,
    session_compaction: dict[str, Any] | None,
    message_count: int = 0,
    summary_count: int = 0,
    recent_microcompact: bool = False,
    preserve_recent: int = DEFAULT_FULL_COMPACT_PRESERVE_RECENT,
) -> FullCompactEligibility:
    """只判断 full compact 候选资格，不执行 compact。"""
    required_preserve_recent = max(0, preserve_recent)
    if max(message_count, summary_count) <= required_preserve_recent:
        return FullCompactEligibility(
            eligible=False,
            reason="insufficient_compressible_history",
            trigger_kind=None,
            required_preserve_recent=required_preserve_recent,
        )

    readiness_status = _string_field(compact_readiness, "status")
    trigger_status = _string_field(auto_compact_trigger, "status")
    trigger_kind = _string_field(auto_compact_trigger, "trigger_kind") or None
    budget_pressure = _float_field(compact_readiness, "budget_pressure")

    if readiness_status == "deterministic_sufficient":
        return FullCompactEligibility(
            eligible=False,
            reason="deterministic_context_sufficient",
            trigger_kind=None,
            required_preserve_recent=required_preserve_recent,
        )

    if readiness_status == "full_compact_candidate":
        return FullCompactEligibility(
            eligible=True,
            reason="full_compact_candidate_after_deterministic_compaction",
            trigger_kind="full_compact_candidate",
            required_preserve_recent=required_preserve_recent,
        )

    if trigger_status == "triggered" and trigger_kind == "session_memory":
        if _session_compaction_sufficient(session_compaction, budget_pressure):
            return FullCompactEligibility(
                eligible=False,
                reason="deterministic_session_memory_sufficient",
                trigger_kind=None,
                required_preserve_recent=required_preserve_recent,
            )
        if budget_pressure >= 0.9 or recent_microcompact:
            return FullCompactEligibility(
                eligible=True,
                reason="auto_trigger_still_high_pressure_after_session_memory",
                trigger_kind="session_memory",
                required_preserve_recent=required_preserve_recent,
            )
        return FullCompactEligibility(
            eligible=False,
            reason="deterministic_session_memory_sufficient",
            trigger_kind=None,
            required_preserve_recent=required_preserve_recent,
        )

    return FullCompactEligibility(
        eligible=False,
        reason="no_full_compact_trigger",
        trigger_kind=None,
        required_preserve_recent=required_preserve_recent,
    )


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


def plan_full_compact_window(
    messages: list[dict],
    preserve_recent: int = DEFAULT_FULL_COMPACT_PRESERVE_RECENT,
) -> FullCompactPlan:
    """规划 older/preserved recent 窗口，保护 assistant tool_calls 与 tool result 配对。"""
    message_count = len(messages)
    requested_recent = min(max(0, preserve_recent), message_count)
    boundary = message_count - requested_recent
    boundary = _adjust_boundary_for_tool_pairs(messages, boundary)
    older_count = boundary
    preserved_count = message_count - boundary
    blocked_reason = None
    if older_count <= 0:
        blocked_reason = "insufficient_older_messages_after_tool_pair_adjustment"
    return FullCompactPlan(
        older_message_count=older_count,
        preserved_recent_count=preserved_count,
        tool_pair_safe=_tool_pair_boundary_safe(messages, boundary),
        blocked_reason=blocked_reason,
        preserved_start_index=boundary,
    )


def validate_full_compact_summary(summary: dict) -> list[str]:
    """验证 future summarizer 输出 schema；只报错，不修复。"""
    errors: list[str] = []
    for field in REQUIRED_FULL_COMPACT_SUMMARY_FIELDS:
        if field not in summary:
            errors.append(f"missing required field: {field}")
            continue
        value = summary[field]
        if field == "task_focus":
            if not isinstance(value, str):
                errors.append("field task_focus must be str")
        elif not isinstance(value, list):
            errors.append(f"field {field} must be list")
    return errors


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


def _session_compaction_sufficient(session_compaction: dict[str, Any] | None, budget_pressure: float) -> bool:
    if budget_pressure >= 0.9:
        return False
    if not isinstance(session_compaction, dict):
        return False
    decision = session_compaction.get("decision")
    saved_chars = session_compaction.get("saved_chars")
    return decision == "compacted" and isinstance(saved_chars, int) and saved_chars > 0


def _adjust_boundary_for_tool_pairs(messages: list[dict], boundary: int) -> int:
    adjusted = boundary
    while adjusted > 0 and not _tool_pair_boundary_safe(messages, adjusted):
        adjusted -= 1
    return adjusted


def _tool_pair_boundary_safe(messages: list[dict], boundary: int) -> bool:
    if boundary <= 0 or boundary >= len(messages):
        return True
    older_tool_call_ids = set()
    preserved_tool_call_ids = set()
    older_tool_result_ids = set()
    preserved_tool_result_ids = set()
    for index, message in enumerate(messages):
        target_tool_calls = older_tool_call_ids if index < boundary else preserved_tool_call_ids
        target_tool_results = older_tool_result_ids if index < boundary else preserved_tool_result_ids
        for call_id in _assistant_tool_call_ids(message):
            target_tool_calls.add(call_id)
        tool_call_id = message.get("tool_call_id")
        if message.get("role") == "tool" and isinstance(tool_call_id, str):
            target_tool_results.add(tool_call_id)
    return not (older_tool_call_ids & preserved_tool_result_ids or preserved_tool_call_ids & older_tool_result_ids)


def _assistant_tool_call_ids(message: dict) -> list[str]:
    if message.get("role") != "assistant":
        return []
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    ids: list[str] = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str):
            ids.append(str(tool_call["id"]))
    return ids


def _string_field(value: dict[str, Any] | None, field: str) -> str:
    if not isinstance(value, dict):
        return ""
    raw = value.get(field)
    return raw if isinstance(raw, str) else ""


def _float_field(value: dict[str, Any] | None, field: str) -> float:
    if not isinstance(value, dict):
        return 0.0
    raw = value.get(field)
    return float(raw) if isinstance(raw, int | float) else 0.0
