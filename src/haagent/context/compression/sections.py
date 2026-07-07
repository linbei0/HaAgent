"""
src/haagent/context/compression/sections.py - 上下文 section 与 observation 压缩

集中承接 ContextBuilder 的确定性 section 选择、折叠和工具 observation 摘要。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from haagent.context.compression.budget import CompressionBudget
from haagent.runtime.execution.command import redact_secret_like_text


ContextDecision = Literal["selected", "skipped", "collapsed"]
OBSERVATION_MICROCOMPACT_CHAR_LIMIT = 1200
OBSERVATION_MICROCOMPACT_HEAD_CHARS = 600
OBSERVATION_MICROCOMPACT_TAIL_CHARS = 300


@dataclass(frozen=True)
class ContextBudget:
    max_total_chars: int = 12000
    max_section_chars: int = 4000
    max_tool_observation_chars: int = 1200
    keep_recent_observations: int = 4
    collapse_head_chars: int = 600
    collapse_tail_chars: int = 300
    max_total_tokens: int | None = None


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


@dataclass(frozen=True)
class ObservationCompactionRecord:
    tool_name: str
    kind: str
    decision: ContextDecision
    reason: str
    original_chars: int
    final_chars: int


def context_budget_from_compression_budget(budget: CompressionBudget) -> ContextBudget:
    max_total_chars = budget.context_builder_max_tokens * 4
    return ContextBudget(
        max_total_chars=max_total_chars,
        max_section_chars=max(4_000, min(24_000, max_total_chars // 4)),
        max_tool_observation_chars=OBSERVATION_MICROCOMPACT_CHAR_LIMIT,
        keep_recent_observations=budget.full_compact_preserve_recent,
        collapse_head_chars=budget.historical_collapse_head_chars,
        collapse_tail_chars=budget.historical_collapse_tail_chars,
        max_total_tokens=budget.context_builder_max_tokens,
    )


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
    historical_tool_compression_count: int = 0,
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
    if historical_tool_compression_count:
        reasons.append("historical_tool_compression_present")
    if session_history_over_budget:
        reasons.append("session_history_over_budget")

    triggered = session_history_over_budget and (
        budget_pressure >= 0.9
        or skipped_count > 0
        or collapsed_count > 0
        or historical_tool_compression_count > 0
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
        "historical_tool_compression_count": max(0, historical_tool_compression_count),
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


def compact_observation_with_record(
    observation: dict[str, object],
    *,
    max_chars: int = OBSERVATION_MICROCOMPACT_CHAR_LIMIT,
    head_chars: int = OBSERVATION_MICROCOMPACT_HEAD_CHARS,
    tail_chars: int = OBSERVATION_MICROCOMPACT_TAIL_CHARS,
) -> tuple[str, ObservationCompactionRecord]:
    """返回 observation 的确定性 microcompact 文本和机器可读记录。"""
    tool_name = observation_tool_name(observation)
    summary = observation_summary(observation)
    raw = json.dumps(raw_observation_summary(observation), ensure_ascii=False)
    compacted_summary, field_collapsed = _microcompact_summary_fields(
        summary,
        observation,
        max_chars=max_chars,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )
    compacted = json.dumps(compacted_summary, ensure_ascii=False)
    if len(compacted) > max_chars and not field_collapsed:
        compacted, _collapsed_chars = collapse_text_head_tail(
            compacted,
            max_chars=max_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        field_collapsed = True
    decision: ContextDecision = "collapsed" if field_collapsed or len(compacted) > max_chars else "selected"
    reason = "observation_over_budget" if decision == "collapsed" else "within_budget"
    original_chars = len(raw) if decision == "collapsed" else len(compacted)
    return compacted, ObservationCompactionRecord(
        tool_name=tool_name,
        kind="observation",
        decision=decision,
        reason=reason,
        original_chars=original_chars,
        final_chars=len(compacted),
    )


def observation_tool_name(observation: dict[str, object]) -> str:
    return str(observation.get("tool_name") or "unknown_tool")


def observation_summary(observation: dict[str, object]) -> dict[str, object]:
    tool_name = observation_tool_name(observation)
    args = _dict_or_empty(observation.get("args"))
    result = _dict_or_empty(observation.get("result"))
    if tool_name == "file_read":
        return _file_read_observation_summary(args, result)
    if tool_name == "file_write":
        return _file_write_observation_summary(args, result)
    if tool_name == "request_user_input":
        return _request_user_input_observation_summary(args, result)
    if tool_name == "file_list":
        return _file_list_observation_summary(args, result)
    if tool_name == "grep":
        return _grep_observation_summary(args, result)
    if tool_name == "shell":
        return _shell_observation_summary(args, result)
    if tool_name == "code_run":
        return _code_run_observation_summary(args, result)
    if tool_name == "apply_patch":
        return _apply_patch_observation_summary(args, result)
    if tool_name == "apply_patch_set":
        return _apply_patch_set_observation_summary(args, result)
    if tool_name == "verification":
        return _verification_observation_summary(args, result)
    if tool_name == "loop_suggestion":
        return _loop_suggestion_observation_summary(args, result)
    return _generic_observation_summary(args, result)


def raw_observation_summary(observation: dict[str, object]) -> dict[str, object]:
    return _generic_observation_summary(
        _dict_or_empty(observation.get("args")),
        _dict_or_empty(observation.get("result")),
    )


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


def _file_read_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "path": _first_present_string(result.get("path"), args.get("path")),
        "start_line": result.get("start_line"),
        "end_line": result.get("end_line"),
        "line_count": result.get("line_count"),
        "truncated": result.get("truncated", False),
        "content": _compact_excerpt(_first_present_string(result.get("content"), result.get("excerpt")))[0],
    }


def _file_write_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "path": _first_present_string(result.get("path"), args.get("path")),
        "mode": args.get("mode"),
        "bytes_written": result.get("bytes_written"),
        "created": result.get("created"),
        "truncated": result.get("truncated", False),
    }


def _request_user_input_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    answer = _first_present_string(result.get("answer"), result.get("answer_excerpt"))
    return {
        "status": result.get("status", "unknown"),
        "question": _first_present_string(result.get("question"), args.get("question")),
        "answer_excerpt": _compact_excerpt(answer)[0],
        "answer_chars": result.get("answer_chars", len(answer)),
    }


def _file_list_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    tree = _first_present_string(result.get("tree"), result.get("content"))
    return {
        "status": result.get("status", "unknown"),
        "path": _first_present_string(result.get("path"), args.get("path"), "."),
        "entry_count": result.get("entry_count"),
        "truncated": result.get("truncated", False),
        "tree": _compact_excerpt(tree)[0],
    }


def _grep_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    matches = result.get("matches")
    formatted_matches = []
    if isinstance(matches, list):
        formatted_matches = [_format_search_match(match) for match in matches[:8]]
    return {
        "status": result.get("status", "unknown"),
        "pattern": _first_present_string(result.get("pattern"), args.get("pattern")),
        "match_count": result.get("match_count", len(formatted_matches)),
        "truncated": result.get("truncated", False),
        "matches": formatted_matches,
    }


def _shell_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    stdout = _first_present_string(result.get("stdout_excerpt"), result.get("stdout"))
    stderr = _first_present_string(result.get("stderr_excerpt"), result.get("stderr"))
    return {
        "status": result.get("status", "unknown"),
        "command": _first_present_string(result.get("command"), args.get("command")),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout", False),
        "stdout": _compact_excerpt(stdout)[0],
        "stderr": _compact_excerpt(stderr)[0],
        "truncated": result.get("truncated", False),
    }


def _code_run_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    stdout = _first_present_string(result.get("stdout_excerpt"), result.get("stdout"))
    stderr = _first_present_string(result.get("stderr_excerpt"), result.get("stderr"))
    return {
        "status": result.get("status", "unknown"),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout", False),
        "stdout": _compact_excerpt(stdout)[0],
        "stderr": _compact_excerpt(stderr)[0],
        "truncated": result.get("truncated", False),
    }


def _apply_patch_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    patch = _first_present_string(result.get("patch"), args.get("patch"))
    return {
        "status": result.get("status", "unknown"),
        "path": _first_present_string(result.get("path"), args.get("path")),
        "changed": result.get("changed"),
        "patch": _compact_excerpt(patch)[0],
    }


def _apply_patch_set_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "replacement_count": result.get("replacement_count", _patch_set_arg_count(args)),
        "changed_paths": result.get("changed_paths", []),
        "summary": _compact_excerpt(_first_present_string(result.get("summary")))[0],
    }


def _verification_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "command": _first_present_string(result.get("command"), args.get("command")),
        "exit_code": result.get("exit_code"),
        "stdout": _compact_excerpt(_first_present_string(result.get("stdout")))[0],
        "stderr": _compact_excerpt(_first_present_string(result.get("stderr")))[0],
    }


def _loop_suggestion_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    del args
    return {
        "status": result.get("status", "unknown"),
        "suggested_tool": result.get("suggested_tool"),
        "reason": _compact_excerpt(_first_present_string(result.get("reason")))[0],
    }


def _format_search_match(match: object) -> str:
    if isinstance(match, dict):
        path = _first_present_string(match.get("path"), match.get("file"))
        line = match.get("line") or match.get("line_number")
        text = _compact_excerpt(_first_present_string(match.get("text"), match.get("line_text")))[0]
        return f"{path}:{line}: {text}"
    return _compact_excerpt(str(match))[0]


def _generic_observation_summary(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "args": args,
        "result": result,
    }


def _compact_excerpt(value: str) -> tuple[str, bool]:
    redacted, changed = redact_secret_like_text(value)
    compacted, collapsed_chars = collapse_text_head_tail(redacted, max_chars=600, head_chars=300, tail_chars=160)
    return compacted, changed or collapsed_chars > 0


def _microcompact_summary_fields(
    summary: dict[str, object],
    observation: dict[str, object],
    *,
    max_chars: int,
    head_chars: int,
    tail_chars: int,
) -> tuple[dict[str, object], bool]:
    updated = dict(summary)
    changed = False
    for key in _long_text_summary_keys().get(observation_tool_name(observation), "").split(","):
        if not key or not isinstance(updated.get(key), str):
            continue
        value = str(updated[key])
        field_max_chars = max_chars
        compacted, collapsed_chars = collapse_text_head_tail(
            value,
            max_chars=field_max_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        if collapsed_chars:
            updated[key] = compacted
            changed = True
        elif "...[collapsed " in value:
            changed = True
    return updated, changed


def _long_text_summary_keys() -> dict[str, str]:
    return {
        "file_read": "content",
        "file_list": "tree",
        "shell": "stdout,stderr",
        "code_run": "stdout,stderr",
        "apply_patch": "patch",
        "verification": "stdout,stderr",
    }


def _dict_or_empty(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_present(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def _string_value(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _first_present_string(*values: object) -> str:
    value = _first_present(*values)
    return _string_value(value)


def _patch_set_arg_count(args: dict[str, Any]) -> int:
    replacements = args.get("replacements")
    return len(replacements) if isinstance(replacements, list) else 0
