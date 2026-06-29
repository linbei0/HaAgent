"""
haagent/cli_inspect.py - episode inspect 摘要渲染

提供 CLI inspect 子命令使用的人类可读 episode package 摘要格式化逻辑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    load_inspect_episode_package,
)


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
    model_calls = [
        record
        for record in transcript
        if record.get("event") == "model_call"
    ]

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
    lines.extend(["", "Sandbox"])
    lines.extend(_format_sandbox(sandbox))
    lines.extend(["", "Workspace Preflight"])
    lines.extend(_format_workspace_preflight(workspace_preflight))
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
    return lines


def _format_plan(plan: dict[str, Any]) -> list[str]:
    planned_steps = plan.get("planned_steps", [])
    if not planned_steps:
        return ["- none"]
    return [f"- {step}" for step in planned_steps]


def _format_sandbox(sandbox: dict[str, Any]) -> list[str]:
    resource_limits = sandbox.get("resource_limits", {})
    if not isinstance(resource_limits, dict):
        resource_limits = {}
    return [
        f"- filesystem_boundary: {sandbox.get('filesystem_boundary', 'unknown')}",
        f"- network_policy: {sandbox.get('network_policy', 'unknown')}",
        f"- process_policy: {sandbox.get('process_policy', 'unknown')}",
        f"- credential_policy: {sandbox.get('credential_policy', 'unknown')}",
        (
            "- command_timeout_seconds: "
            f"{resource_limits.get('command_timeout_seconds', 'unknown')}"
        ),
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
    return [
        (
            f"- provider={call.get('provider', 'unknown')} "
            f"context_id={call.get('context_id', 'unknown')}"
        )
        for call in model_calls
    ]


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
        and error.get("type") in {"tool_not_allowed", "unknown_tool"}
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
