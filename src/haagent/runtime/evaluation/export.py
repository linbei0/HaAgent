"""
src/haagent/runtime/evaluation/export.py - Eval Case 导出器

把已校验的 episode package 转换为可审计、可序列化的最小 eval case 字典。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.episodes.validator import load_validated_episode_package
from haagent.runtime.contracts.task import load_task


EVAL_CASE_VERSION = "1.0"


def export_eval_case(episode_path: Path) -> dict[str, Any]:
    """导出单个 episode 的 eval case；入口先执行完整 package 校验。"""
    package_view = load_validated_episode_package(episode_path)

    episode_metadata = package_view.episode_metadata
    failure_record = package_view.failure_record
    task = load_task(episode_path / "task.yaml")

    return {
        "eval_case_version": EVAL_CASE_VERSION,
        "episode_version": episode_metadata["episode_version"],
        "task": {
            "goal": task.goal,
            "constraints": task.constraints,
            "allowed_tools": task.allowed_tools,
            "acceptance_criteria": task.acceptance_criteria,
            "verification_commands": task.verification_commands,
            "policy": task.policy,
        },
        "workspace_root": episode_metadata["workspace_root"],
        "final_status": episode_metadata["status"],
        "expected_tool_uses": _tool_names_used(package_view.tool_calls),
        "expectations": _expectations_summary(episode_metadata, failure_record, package_view.transcript),
        "failure": _failure_summary(failure_record),
        "verification": _verification_summary(package_view.verification_commands),
        "sandbox_summary": _sandbox_summary(package_view.sandbox),
        "environment_summary": _environment_summary(package_view.environment),
        "cost_summary": _cost_summary(package_view.cost),
        "tool_names_used": _tool_names_used(package_view.tool_calls),
        "tool_argument_errors": _tool_argument_errors(package_view.tool_calls),
        "approval_summary": _approval_summary(package_view.tool_calls),
        "human_interactions": _human_interactions_summary(package_view.transcript),
        "final_response": _final_response_summary(package_view.transcript),
    }


def _expectations_summary(
    episode_metadata: dict[str, Any],
    failure_record: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    final_response = _final_response_summary(transcript)
    failure = _failure_summary(failure_record)
    return {
        "final_status": episode_metadata["status"],
        "failure_category": failure["category"] if failure else None,
        "final_response": {
            "mode": "contains",
            "value": final_response["content"] if final_response else "",
        },
    }


def _failure_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    if record["status"] == "success":
        return None
    failure = record["failure"]
    return {
        "category": failure["category"],
        "stage": failure["stage"],
        "evidence": failure["evidence"],
    }


def _verification_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "command": record["command"],
            "status": record["status"],
            "exit_code": record["exit_code"],
            "timeout": record["timeout"],
            "stdout_excerpt": record["stdout_excerpt"],
            "stderr_excerpt": record["stderr_excerpt"],
            "stdout_truncated": record["stdout_truncated"],
            "stderr_truncated": record["stderr_truncated"],
            "stdout_original_length": record["stdout_original_length"],
            "stderr_original_length": record["stderr_original_length"],
            "redacted": record["redacted"],
        }
        for record in records
    ]


def _sandbox_summary(sandbox: dict[str, Any]) -> dict[str, Any]:
    resource_limits = sandbox["resource_limits"]
    availability = sandbox.get("availability", {})
    isolation = sandbox.get("isolation", {})
    return {
        "workspace_root": sandbox["workspace_root"],
        "filesystem_boundary": sandbox["filesystem_boundary"],
        "backend": sandbox["backend"],
        "network_policy": sandbox["network_policy"],
        "process_policy": sandbox["process_policy"],
        "credential_policy": sandbox["credential_policy"],
        "command_timeout_seconds": resource_limits["command_timeout_seconds"],
        "cpu_limit": resource_limits.get("cpu_limit"),
        "memory_limit": resource_limits.get("memory_limit"),
        "pids_limit": resource_limits.get("pids_limit"),
        "degraded": availability.get("degraded"),
        "availability_reason": availability.get("reason"),
        "sandbox_user": isolation.get("user"),
        "privileged": isolation.get("privileged"),
    }


def _environment_summary(environment: dict[str, Any]) -> dict[str, Any]:
    haagent = environment.get("haagent", {})
    model = environment.get("model", {})
    tools = environment.get("tools", {})
    if not isinstance(haagent, dict):
        haagent = {}
    if not isinstance(model, dict):
        model = {}
    if not isinstance(tools, dict):
        tools = {}
    return {
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "haagent_version": haagent.get("package_version"),
        "model_provider": model.get("provider"),
        "model": model.get("model"),
        "endpoint": model.get("endpoint"),
        "allowed_tool_count": tools.get("allowed_tool_count"),
    }


def _cost_summary(cost: dict[str, Any]) -> dict[str, Any]:
    totals = cost.get("totals", {})
    if not isinstance(totals, dict):
        totals = {}
    return {
        "usage_available": cost.get("usage_available"),
        "pricing_available": cost.get("pricing_available"),
        "model_call_count": totals.get("model_call_count"),
        "input_tokens": totals.get("input_tokens"),
        "output_tokens": totals.get("output_tokens"),
        "total_tokens": totals.get("total_tokens"),
        "estimated_cost": cost.get("estimated_cost"),
        "currency": cost.get("currency"),
        "reason": cost.get("reason"),
    }


def _tool_names_used(records: list[dict[str, Any]]) -> list[str]:
    names = {str(record["tool_name"]) for record in records}
    return sorted(names)


def _tool_argument_errors(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    errors = []
    for record in records:
        error = record.get("error")
        if isinstance(error, dict) and error.get("type") == "tool_argument_invalid":
            errors.append(
                {
                    "tool_name": str(record.get("tool_name", "unknown")),
                    "message": str(error.get("message", "")),
                },
            )
    return errors


def _approval_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_approval_summary_record(record) for record in records]


def _approval_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(record["tool_name"])
    policy = record.get("policy")
    if policy is None and _policy_not_evaluated(record):
        error = record.get("error") if isinstance(record.get("error"), dict) else {}
        return {
            "tool_name": tool_name,
            "action": "not_evaluated",
            "approval_required": False,
            "approval_status": "not_evaluated",
            "approval_reason": str(error.get("message", "")),
        }
    approval = policy["approval"]
    return {
        "tool_name": tool_name,
        "action": policy["action"],
        "approval_required": approval["required"],
        "approval_status": approval["status"],
        "approval_reason": approval["reason"],
    }


def _policy_not_evaluated(record: dict[str, Any]) -> bool:
    error = record.get("error")
    return (
        record.get("status") == "error"
        and isinstance(error, dict)
        and error.get("type") in {"tool_not_allowed", "unknown_tool", "tool_call_skipped"}
    )


def _human_interactions_summary(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for record in transcript:
        event = record.get("event")
        if event not in {
            "user_input_requested",
            "user_input_received",
            "approval_requested",
            "approval_granted",
            "approval_denied",
        }:
            continue
        summary = {
            "event": event,
            "tool_name": str(record.get("tool_name", "unknown")),
            "question": str(record.get("question", "")),
            "approved": record.get("approved"),
        }
        if event == "user_input_received":
            summary["answer_chars"] = record.get("answer_chars")
        events.append(summary)
    return events


def _final_response_summary(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    response = _last_model_response(transcript)
    if response is None:
        return None
    tool_calls = response.get("tool_calls", [])
    return {
        "provider": str(response.get("provider", "unknown")),
        "turn": response.get("turn"),
        "content": str(response.get("content", "")),
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
    }


def _last_model_response(transcript: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(transcript):
        if record.get("event") == "model_response":
            return record
    return None
