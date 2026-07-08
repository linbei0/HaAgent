"""
haagent/cli_inspect.py - episode inspect 摘要渲染

提供 CLI inspect 子命令使用的人类可读 episode package 摘要格式化逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haagent.runtime.episodes.validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)
from haagent.runtime.session.task_ledger import TaskLedgerError, task_ledger_from_dict


class EpisodeInspectError(RuntimeError):
    """Raised when an episode package cannot be inspected safely."""


def render_episode_summary(episode_path: Path) -> str:
    """读取 episode package，并生成面向人的审计摘要。"""
    try:
        package_view = load_inspect_episode_package(episode_path)
    except EpisodeValidationError as error:
        raise EpisodeInspectError(str(error)) from error
    episode_metadata = package_view.episode_metadata
    context_manifest = package_view.context_manifest
    plan = package_view.plan
    transcript = package_view.transcript
    tool_calls = package_view.tool_calls
    verification = package_view.verification_commands
    verification_reached = package_view.verification_reached
    failure_record = package_view.failure_record
    environment = package_view.environment
    cost = package_view.cost
    sandbox = package_view.sandbox
    workspace_preflight = package_view.workspace_preflight

    failure_attribution = (episode_path / "failure-attribution.md").read_text(encoding="utf-8").strip()

    state_flow = [
        record["status"]
        for record in transcript
        if record.get("event") == "state_transition"
    ]
    final_status = state_flow[-1] if state_flow else "unknown"
    final_status = episode_metadata.get("status", final_status)
    model_calls = cost.get("model_calls", [])

    lines = [
        "Run Summary",
        f"- episode_path: {episode_path}",
        f"- episode_version: {episode_metadata.get('episode_version', 'unknown')}",
        f"- status: {final_status}",
        f"- provider: {_summary_provider(episode_metadata)}",
        f"- context_count: {context_manifest.get('context_count', 0)}",
        "",
        "State Flow",
        f"- {' -> '.join(state_flow) if state_flow else 'none'}",
        "",
        "Contexts",
    ]
    lines.extend(_format_contexts(episode_path, context_manifest.get("contexts", [])))
    lines.extend(["", "Plan"])
    lines.extend(_format_plan(plan))
    lines.extend(["", "Environment"])
    lines.extend(_format_environment(environment))
    lines.extend(["", "Cost"])
    lines.extend(_format_cost(cost))
    lines.extend(["", "Sandbox"])
    lines.extend(_format_sandbox(sandbox))
    lines.extend(["", "Workspace Preflight"])
    lines.extend(_format_workspace_preflight(workspace_preflight))
    lines.extend(["", "Task Ledger"])
    lines.extend(_format_task_ledger_for_episode(episode_path))
    lines.extend(["", "Next Actions"])
    lines.extend(_format_next_actions(episode_path, context_manifest.get("contexts", [])))
    lines.extend(["", "Model Calls"])
    lines.extend(_format_model_calls(model_calls))
    lines.extend(["", "Final Response"])
    lines.extend(_format_final_response(transcript))
    lines.extend(["", "Tool Calls"])
    lines.extend(_format_tool_calls(tool_calls))
    lines.extend(["", "Human Interactions"])
    lines.extend(_format_human_interactions(transcript))
    lines.extend(["", "Compression Diagnostics"])
    lines.extend(_format_compression_diagnostics(transcript))
    lines.extend(["", "Approval Summary"])
    lines.extend(_format_approval_summary(tool_calls))
    lines.extend(["", "Tool Argument Errors"])
    lines.extend(_format_tool_argument_errors(tool_calls))
    lines.extend(["", "Verification"])
    lines.extend(_format_verification(verification, verification_reached))
    lines.extend(["", "Structured Failure"])
    lines.extend(_format_failure_record(failure_record))
    lines.extend(["", "Failure Attribution", failure_attribution])
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_task_ledger_for_episode(episode_path: Path) -> list[str]:
    ledger_path = _find_task_ledger_for_episode(episode_path)
    if ledger_path is None:
        return ["- none"]
    try:
        ledger = task_ledger_from_dict(_read_json(ledger_path))
    except (OSError, json.JSONDecodeError, TaskLedgerError) as error:
        return [f"- invalid: {error}"]
    active = ledger.active_step()
    blocked = [step for step in ledger.steps if step.status == "blocked"]
    completed = [step for step in ledger.steps if step.status == "completed"]
    lines = [
        f"- session_ledger: {ledger_path}",
        f"- status: {ledger.status}",
        f"- current_step_id: {ledger.current_step_id or 'none'}",
        f"- steps: total={len(ledger.steps)} completed={len(completed)} blocked={len(blocked)}",
        f"- checkpoints: {len(ledger.checkpoints)}",
    ]
    if active is not None:
        lines.append(
            f"- active_step: {active.id} evidence={len(active.evidence_refs)} "
            f"checkpoints={len(active.checkpoint_ids)} [{active.status}/{active.owner}] {active.title}"
        )
        if active.blocker:
            category = active.blocker.get("category", "blocked")
            reason = active.blocker.get("reason", "")
            suggested_action = active.blocker.get("suggested_action", "")
            recovery = f"- recovery: {category} {reason}".strip()
            if suggested_action:
                recovery = f"{recovery} suggested_action={suggested_action}"
            lines.append(recovery)
    return lines


def _find_task_ledger_for_episode(episode_path: Path) -> Path | None:
    runs_root = _find_runs_root_for_episode(episode_path)
    if runs_root is None:
        return None
    sessions_root = runs_root / "sessions"
    if not sessions_root.exists():
        return None
    targets = {str(episode_path)}
    try:
        targets.add(str(episode_path.resolve()))
    except OSError:
        pass
    for turns_path in sessions_root.glob("*/turns.jsonl"):
        try:
            lines = turns_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and str(record.get("episode_path", "")) in targets:
                ledger_path = turns_path.parent / "task-ledger.json"
                return ledger_path if ledger_path.exists() else None
    return None


def _find_runs_root_for_episode(episode_path: Path) -> Path | None:
    for candidate in (episode_path.parent, *episode_path.parents):
        if candidate.name == ".runs":
            return candidate
    return None


def _summary_provider(episode_metadata: dict[str, Any]) -> str:
    return str(episode_metadata.get("provider", "unknown"))


def _format_contexts(episode_path: Path, contexts: list[dict[str, Any]]) -> list[str]:
    if not contexts:
        return ["- none"]
    lines: list[str] = []
    for context in contexts:
        lines.append(
            f"- {context['context_id']}: "
            f"{context['model_input_path']} | {context['manifest_path']}",
        )
        lines.extend(_format_context_compaction(episode_path, context))
    return lines


def _format_context_compaction(episode_path: Path, context: dict[str, Any]) -> list[str]:
    manifest_path = context.get("manifest_path")
    if not isinstance(manifest_path, str):
        return []
    try:
        context_manifest = _read_json(episode_path / manifest_path)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    compaction = context_manifest.get("compaction")
    if not isinstance(compaction, dict):
        return []
    lines = [
        "  compaction: "
        f"original={compaction.get('original_chars', 0)} "
        f"final={compaction.get('final_chars', 0)} "
        f"saved={compaction.get('saved_chars', 0)} "
        f"selected={compaction.get('selected_count', 0)} "
        f"collapsed={compaction.get('collapsed_count', 0)} "
        f"skipped={compaction.get('skipped_count', 0)}",
    ]
    skipped_reasons = compaction.get("skipped_reasons")
    if isinstance(skipped_reasons, dict) and skipped_reasons:
        reason_parts = [f"{reason}={count}" for reason, count in sorted(skipped_reasons.items())]
        lines.append(f"  skipped_reasons: {', '.join(reason_parts)}")
    lines.extend(_format_source_diagnostics(context_manifest.get("source_diagnostics")))
    lines.extend(_format_compact_readiness(context_manifest.get("compact_readiness")))
    lines.extend(_format_auto_compact_trigger(context_manifest.get("auto_compact_trigger")))
    lines.extend(_format_session_compaction(context_manifest.get("session_compaction")))
    lines.extend(_format_full_compact_contract(context_manifest.get("full_compact_contract")))
    lines.extend(_format_full_compact(context_manifest.get("full_compact")))
    return lines


def _format_auto_compact_trigger(auto_compact_trigger: Any) -> list[str]:
    if not isinstance(auto_compact_trigger, dict):
        return []
    kind = auto_compact_trigger.get("trigger_kind") or "none"
    return [
        "  auto_compact_trigger: "
        f"status={auto_compact_trigger.get('status', 'unknown')} "
        f"kind={kind} "
        f"recommendation={auto_compact_trigger.get('recommendation', 'unknown')}",
    ]


def _format_session_compaction(session_compaction: Any) -> list[str]:
    if not isinstance(session_compaction, dict):
        return []
    return [
        "  session_compaction: "
        f"decision={session_compaction.get('decision', 'unknown')} "
        f"original_turns={session_compaction.get('original_turn_count', 0)} "
        f"compacted_turns={session_compaction.get('compacted_turn_count', 0)} "
        f"preserved_recent={session_compaction.get('preserved_recent_count', 0)} "
        f"saved={session_compaction.get('saved_chars', 0)}",
    ]


def _format_full_compact_contract(full_compact_contract: Any) -> list[str]:
    if not isinstance(full_compact_contract, dict):
        return []
    return [
        "  full_compact_contract: "
        f"eligible={_format_bool(full_compact_contract.get('eligible'))} "
        f"reason={full_compact_contract.get('reason', 'unknown')} "
        f"preserve_recent={full_compact_contract.get('required_preserve_recent', 0)}",
    ]


def _format_full_compact(full_compact: Any) -> list[str]:
    if not isinstance(full_compact, dict):
        return []
    applied = full_compact.get("applied")
    if applied is True:
        return [
            "  full_compact: "
            "applied=true "
            f"older={full_compact.get('older_message_count', 0)} "
            f"preserved={full_compact.get('preserved_recent_count', 0)} "
            f"summary_chars={full_compact.get('summary_chars', 0)} "
            f"reason={full_compact.get('reason', 'unknown')}",
        ]
    return [
        "  full_compact: "
        f"applied={_format_bool(applied)} "
        f"reason={full_compact.get('reason', 'unknown')}",
    ]


def _format_compact_readiness(compact_readiness: Any) -> list[str]:
    if not isinstance(compact_readiness, dict):
        return []
    lines = [
        "  compact_readiness: "
        f"status={compact_readiness.get('status', 'unknown')} "
        f"pressure={compact_readiness.get('budget_pressure', 0)} "
        f"saved_ratio={compact_readiness.get('saved_ratio', 0)} "
        f"recommendation={compact_readiness.get('recommendation', 'unknown')}",
    ]
    reasons = compact_readiness.get("reasons")
    if isinstance(reasons, list) and reasons:
        reason_text = ", ".join(str(reason) for reason in reasons)
        lines.append(f"  readiness_reasons: {reason_text}")
    return lines


def _format_source_diagnostics(source_diagnostics: Any) -> list[str]:
    if not isinstance(source_diagnostics, dict):
        return []
    lines: list[str] = []
    session = source_diagnostics.get("session_summary")
    if isinstance(session, dict):
        lines.append(
            "  source_diagnostics: session_summary "
            f"included={_format_bool(session.get('included'))} "
            f"chars={session.get('model_input_chars', 0)}/{session.get('limit', 0)}",
        )
    memory = source_diagnostics.get("memory")
    if isinstance(memory, dict):
        lines.append(
            "  source_diagnostics: memory "
            f"used={memory.get('used_count', 0)} "
            f"skipped_over_budget={memory.get('skipped_over_budget', 0)} "
            f"included={_format_bool(memory.get('included_in_model_input'))}",
        )
    observations = source_diagnostics.get("observations")
    if isinstance(observations, dict):
        lines.append(
            "  source_diagnostics: observations "
            f"included={_format_bool(observations.get('included_in_model_input'))} "
            f"sections={observations.get('observation_section_count', 0)} "
            f"compacted={observations.get('compacted_count', 0)} "
            f"truncated={observations.get('truncated_count', 0)} "
            f"saved={observations.get('saved_chars', 0)}",
        )
    return lines


def _format_plan(plan: dict[str, Any]) -> list[str]:
    planned_steps = plan.get("planned_steps", [])
    if not planned_steps:
        return ["- none"]
    return [f"- {step}" for step in planned_steps]


def _format_environment(environment: dict[str, Any]) -> list[str]:
    if not environment:
        return ["- none"]
    model = environment.get("model", {})
    tools = environment.get("tools", {})
    haagent = environment.get("haagent", {})
    if not isinstance(model, dict):
        model = {}
    if not isinstance(tools, dict):
        tools = {}
    if not isinstance(haagent, dict):
        haagent = {}
    provider = model.get("provider", "unknown")
    model_name = model.get("model") or "unknown"
    return [
        f"- python: {environment.get('python', 'unknown')}",
        f"- platform: {environment.get('platform', 'unknown')}",
        f"- haagent_version: {haagent.get('package_version', 'unknown')}",
        f"- model: {provider}/{model_name}",
        f"- endpoint: {model.get('endpoint') or 'unknown'}",
        f"- allowed_tool_count: {tools.get('allowed_tool_count', 'unknown')}",
    ]


def _format_cost(cost: dict[str, Any]) -> list[str]:
    if not cost:
        return ["- none"]
    totals = cost.get("totals", {})
    if not isinstance(totals, dict):
        totals = {}
    estimated_cost = cost.get("estimated_cost")
    currency = cost.get("currency")
    if estimated_cost is None:
        estimated = "unavailable"
    elif currency:
        estimated = f"{estimated_cost} {currency}"
    else:
        estimated = str(estimated_cost)
    return [
        f"- usage_available: {_format_bool(cost.get('usage_available'))}",
        f"- pricing_available: {_format_bool(cost.get('pricing_available'))}",
        f"- model_call_count: {totals.get('model_call_count', 'unknown')}",
        f"- input_tokens: {_format_optional_count(totals.get('input_tokens'))}",
        f"- output_tokens: {_format_optional_count(totals.get('output_tokens'))}",
        f"- total_tokens: {_format_optional_count(totals.get('total_tokens'))}",
        f"- estimated_cost: {estimated}",
        f"- reason: {cost.get('reason') or 'none'}",
    ]


def _format_sandbox(sandbox: dict[str, Any]) -> list[str]:
    resource_limits = sandbox.get("resource_limits", {})
    if not isinstance(resource_limits, dict):
        resource_limits = {}
    isolation = sandbox.get("isolation", {})
    if not isinstance(isolation, dict):
        isolation = {}
    availability = sandbox.get("availability", {})
    if not isinstance(availability, dict):
        availability = {}
    return [
        f"- backend: {sandbox.get('backend', 'unknown')}",
        f"- filesystem_boundary: {sandbox.get('filesystem_boundary', 'unknown')}",
        f"- network_policy: {sandbox.get('network_policy', 'unknown')}",
        f"- process_policy: {sandbox.get('process_policy', 'unknown')}",
        f"- credential_policy: {sandbox.get('credential_policy', 'unknown')}",
        (
            "- command_timeout_seconds: "
            f"{resource_limits.get('command_timeout_seconds', 'unknown')}"
        ),
        f"- cpu_limit: {resource_limits.get('cpu_limit', 'unknown')}",
        f"- memory_limit: {resource_limits.get('memory_limit', 'unknown')}",
        f"- pids_limit: {resource_limits.get('pids_limit', 'unknown')}",
        f"- user: {isolation.get('user', 'unknown')}",
        f"- privileged: {isolation.get('privileged', 'unknown')}",
        f"- degraded: {availability.get('degraded', 'unknown')}",
        f"- availability_reason: {availability.get('reason', '')}",
    ]


def _format_workspace_preflight(preflight: dict[str, Any]) -> list[str]:
    if not preflight:
        return ["- none"]
    lines = [
        f"- workspace_root: {preflight.get('workspace_root', 'unknown')}",
        f"- exists: {_format_bool(preflight.get('exists'))}",
        f"- git_status: {preflight.get('git_status', 'unknown')}",
        f"- is_git_repo: {_format_bool(preflight.get('is_git_repo'))}",
        f"- git_branch: {preflight.get('git_branch') or 'none'}",
        f"- git_dirty: {_format_bool(preflight.get('git_dirty'))}",
    ]
    summary = preflight.get("git_dirty_summary")
    if isinstance(summary, dict):
        lines.append(
            (
                "- git_dirty_summary: "
                f"total={summary.get('total', 0)} "
                f"modified={summary.get('modified', 0)} "
                f"untracked={summary.get('untracked', 0)} "
                f"deleted={summary.get('deleted', 0)} "
                f"renamed={summary.get('renamed', 0)} "
                f"other={summary.get('other', 0)}"
            ),
        )
    lines.append(
        (
            "- modifies_original_workspace: "
            f"{_format_bool(preflight.get('modifies_original_workspace'))}"
        ),
    )
    return lines


def _format_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _format_optional_count(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return "unavailable"


def _format_next_actions(episode_path: Path, contexts: list[dict[str, Any]]) -> list[str]:
    if not contexts:
        return ["- none"]
    return [
        f"- {context.get('context_id', 'unknown')}: messages accumulated in conversation history"
        for context in contexts
    ]


def _format_model_calls(model_calls: list[dict[str, Any]]) -> list[str]:
    if not model_calls:
        return ["- none"]
    lines = []
    for call in model_calls:
        token_text = ""
        total = call.get("total_tokens")
        if total is not None:
            token_text = f" total_tokens={total}"
        lines.append(
            (
                f"- turn={call.get('turn', '?')} "
                f"provider={call.get('provider', 'unknown')} "
                f"model={call.get('model', 'unknown')}"
                f"{token_text}"
            ),
        )
    return lines


def _format_final_response(transcript: list[dict[str, Any]]) -> list[str]:
    response = _last_model_response(transcript)
    if response is None:
        return ["- none"]
    tool_calls = response.get("tool_calls", [])
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else 0
    content = str(response.get("content", ""))
    return [
        (
            f"- provider={response.get('provider', 'unknown')} "
            f"turn={response.get('turn', 'unknown')} "
            f"tool_call_count={tool_call_count}"
        ),
        f"- content: {_excerpt(content)}",
    ]


def _last_model_response(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return record
    return None


def _excerpt(content: str, limit: int = 500) -> str:
    if len(content) <= limit:
        return content
    return content[:limit] + "... [truncated]"


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    if not tool_calls:
        return ["- none"]
    return [
        f"- {call.get('tool_name', 'unknown')}: {call.get('status', 'unknown')}"
        for call in tool_calls
    ]


def _format_human_interactions(transcript: list[dict[str, Any]]) -> list[str]:
    interaction_events = [
        record
        for record in transcript
        if record.get("event")
        in {
            "user_input_requested",
            "user_input_received",
            "approval_requested",
            "approval_granted",
            "approval_denied",
        }
    ]
    if not interaction_events:
        return ["- none"]
    lines = []
    for record in interaction_events:
        event = record.get("event", "unknown")
        tool_name = record.get("tool_name", "unknown")
        question = _excerpt(str(record.get("question", "")), 160)
        if event == "user_input_received":
            lines.append(f"- {event}: tool={tool_name} answer_chars={record.get('answer_chars', 0)}")
        elif event in {"approval_granted", "approval_denied"}:
            approved = str(record.get("approved")).lower()
            lines.append(f"- {event}: tool={tool_name} approved={approved} question={question}")
        else:
            lines.append(f"- {event}: tool={tool_name} question={question}")
    return lines


def _format_compression_diagnostics(transcript: list[dict[str, Any]]) -> list[str]:
    records = [record for record in transcript if record.get("event") == "compression_diagnostic"]
    if not records:
        return ["- none"]
    lines: list[str] = []
    for record in records:
        stage = str(record.get("stage", "unknown"))
        subject = str(record.get("subject") or record.get("tool_name") or "unknown")
        original_chars = record.get("original_chars")
        final_chars = record.get("final_chars")
        artifact = record.get("artifact_path")
        saved = ""
        if isinstance(original_chars, int) and isinstance(final_chars, int):
            saved = f" chars={original_chars}->{final_chars}"
        artifact_text = f" artifact={artifact}" if isinstance(artifact, str) and artifact else ""
        lines.append(
            f"- {stage}: subject={subject} decision={record.get('decision', 'unknown')} "
            f"reason={record.get('reason', 'unknown')}{saved}{artifact_text}",
        )
    return lines


def _format_approval_summary(tool_calls: list[dict[str, Any]]) -> list[str]:
    if not tool_calls:
        return ["- none"]
    lines = []
    for call in tool_calls:
        tool_name = call.get("tool_name", "unknown")
        policy = call.get("policy")
        if policy is None and _policy_not_evaluated(call):
            error = call.get("error") if isinstance(call.get("error"), dict) else {}
            lines.append(f"- {tool_name}: policy=not_evaluated reason={error.get('message', '')}")
            continue
        approval = policy["approval"]
        required = "true" if approval.get("required") is True else "false"
        lines.append(
            (
                f"- {tool_name}: action={policy['action']} "
                f"approval.required={required} "
                f"approval.status={approval['status']} "
                f"approval.reason={approval['reason']}"
            ),
        )
    return lines


def _policy_not_evaluated(call: dict[str, Any]) -> bool:
    error = call.get("error")
    return (
        call.get("status") == "error"
        and isinstance(error, dict)
        and error.get("type") in {"tool_not_allowed", "unknown_tool", "tool_call_skipped"}
    )


def _format_tool_argument_errors(tool_calls: list[dict[str, Any]]) -> list[str]:
    errors = []
    for call in tool_calls:
        error = call.get("error")
        if isinstance(error, dict) and error.get("type") == "tool_argument_invalid":
            errors.append(
                f"- {call.get('tool_name', 'unknown')}: {error.get('message', '')}",
            )
    if not errors:
        return ["- none"]
    return errors


def _format_verification(
    commands: list[dict[str, Any]],
    verification_reached: bool = True,
) -> list[str]:
    if not verification_reached:
        return ["- not reached"]
    if not commands:
        return ["- none"]
    lines = []
    for command in commands:
        lines.append(
            (
                f"- {command.get('command', '')}: {command.get('status', 'unknown')} "
                f"(exit_code={command.get('exit_code')})"
            ),
        )
        if command.get("timeout"):
            lines.append("  timeout: true")
        if command.get("stdout_excerpt"):
            lines.append(f"  stdout: {command['stdout_excerpt']}")
        if command.get("stderr_excerpt"):
            lines.append(f"  stderr: {command['stderr_excerpt']}")
    return lines


def _format_failure_record(record: dict[str, Any]) -> list[str]:
    if record.get("status") == "success":
        return ["- status: success"]
    failure = record.get("failure") or {}
    return [
        f"- status: {record.get('status', 'unknown')}",
        f"- category: {failure.get('category', 'unknown')}",
        f"- stage: {failure.get('stage', 'unknown')}",
        f"- evidence: {failure.get('evidence', '')}",
    ]
