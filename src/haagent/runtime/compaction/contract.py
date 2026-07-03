"""
src/haagent/runtime/compaction/contract.py - full compact 工程契约

定义 full compact 的候选资格、消息窗口和 summary schema；本模块不调用模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
