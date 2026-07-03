"""
tests/unit/runtime/test_full_compact_contract.py - full compact 契约测试

验证 full compact 的候选资格、窗口切分、tool pair 保护和 summary schema。
"""

from __future__ import annotations

from haagent.runtime.compaction.contract import (
    REQUIRED_FULL_COMPACT_SUMMARY_FIELDS,
    assess_full_compact_eligibility,
    plan_full_compact_window,
    validate_full_compact_summary,
)


def test_full_compact_eligibility_false_when_deterministic_sufficient() -> None:
    eligibility = assess_full_compact_eligibility(
        auto_compact_trigger={"status": "not_needed", "triggered": False},
        compact_readiness={"status": "deterministic_sufficient", "budget_pressure": 0.42},
        session_compaction={"decision": "not_needed", "saved_chars": 0},
        message_count=10,
        summary_count=10,
    )

    assert eligibility.eligible is False
    assert eligibility.reason == "deterministic_context_sufficient"
    assert eligibility.trigger_kind is None
    assert eligibility.required_preserve_recent == 6


def test_full_compact_eligibility_true_for_full_compact_candidate_after_session_compaction() -> None:
    eligibility = assess_full_compact_eligibility(
        auto_compact_trigger={"status": "triggered", "triggered": True, "trigger_kind": "session_memory"},
        compact_readiness={"status": "full_compact_candidate", "budget_pressure": 0.96},
        session_compaction={"decision": "compacted", "saved_chars": 6200},
        message_count=18,
        summary_count=18,
        recent_microcompact=True,
    )

    assert eligibility.eligible is True
    assert eligibility.reason == "full_compact_candidate_after_deterministic_compaction"
    assert eligibility.trigger_kind == "full_compact_candidate"


def test_full_compact_eligibility_false_when_session_compaction_sufficient() -> None:
    eligibility = assess_full_compact_eligibility(
        auto_compact_trigger={"status": "triggered", "triggered": True, "trigger_kind": "session_memory"},
        compact_readiness={"status": "watch", "budget_pressure": 0.74},
        session_compaction={"decision": "compacted", "saved_chars": 6200},
        message_count=18,
        summary_count=18,
    )

    assert eligibility.eligible is False
    assert eligibility.reason == "deterministic_session_memory_sufficient"


def test_full_compact_window_preserves_recent_messages() -> None:
    messages = [{"role": "user", "content": f"message {index}"} for index in range(10)]

    plan = plan_full_compact_window(messages, preserve_recent=4)

    assert plan.older_message_count == 6
    assert plan.preserved_recent_count == 4
    assert plan.preserved_start_index == 6
    assert plan.tool_pair_safe is True
    assert plan.blocked_reason is None


def test_full_compact_window_adjusts_boundary_to_preserve_tool_pair() -> None:
    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "file_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
        {"role": "assistant", "content": "after tool"},
    ]

    plan = plan_full_compact_window(messages, preserve_recent=2)

    assert plan.older_message_count == 1
    assert plan.preserved_recent_count == 3
    assert plan.preserved_start_index == 1
    assert plan.tool_pair_safe is True
    assert plan.blocked_reason is None


def test_full_compact_window_blocks_when_no_safe_older_messages_remain() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "file_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
    ]

    plan = plan_full_compact_window(messages, preserve_recent=1)

    assert plan.older_message_count == 0
    assert plan.preserved_recent_count == 2
    assert plan.tool_pair_safe is True
    assert plan.blocked_reason == "insufficient_older_messages_after_tool_pair_adjustment"


def test_validate_full_compact_summary_accepts_complete_schema() -> None:
    summary = {field: [] for field in REQUIRED_FULL_COMPACT_SUMMARY_FIELDS}
    summary["task_focus"] = "finish phase seven"

    assert validate_full_compact_summary(summary) == []


def test_validate_full_compact_summary_reports_missing_and_wrong_types() -> None:
    summary = {field: [] for field in REQUIRED_FULL_COMPACT_SUMMARY_FIELDS if field != "risks"}
    summary["task_focus"] = ["wrong"]
    summary["verification"] = "pytest"

    errors = validate_full_compact_summary(summary)

    assert "missing required field: risks" in errors
    assert "field task_focus must be str" in errors
    assert "field verification must be list" in errors
