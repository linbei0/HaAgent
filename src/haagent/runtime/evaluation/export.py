"""
src/haagent/runtime/evaluation/export.py - Eval Case 导出器

把已校验的 typed EpisodePackage 转换为可审计、可序列化的最小 eval case 字典。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.contracts.task import load_task
from haagent.runtime.episodes.package_types import (
    CostRecord,
    EpisodeMetadata,
    EpisodePackage,
    EnvironmentRecord,
    FailureRecord,
    VerificationCommandRecord,
)
from haagent.runtime.episodes.validator import load_validated_episode_package


EVAL_CASE_VERSION = "1.0"


def export_eval_case(episode_path: Path) -> dict[str, Any]:
    """导出单个 episode 的 eval case；入口先执行完整 package 校验。"""
    package = load_validated_episode_package(episode_path)
    task = load_task(episode_path / "task.yaml")
    final_response = _final_response_summary(package)

    return {
        "eval_case_version": EVAL_CASE_VERSION,
        "episode_version": package.metadata.episode_version,
        "task": {
            "goal": task.goal,
            "constraints": task.constraints,
            "allowed_tools": task.allowed_tools,
            "acceptance_criteria": task.acceptance_criteria,
            "verification_commands": task.verification_commands,
            "policy": task.policy,
        },
        "workspace_root": package.metadata.workspace_root,
        "final_status": package.metadata.status,
        "expected_tool_uses": package.tool_names_used(),
        "expectations": _expectations_summary(package.metadata, package.failure, final_response),
        "failure": _failure_summary(package.failure),
        "verification": _verification_summary(package.verification_commands),
        "sandbox_summary": _sandbox_summary(package.sandbox),
        "environment_summary": _environment_summary(package.environment),
        "cost_summary": _cost_summary(package.cost),
        "tool_names_used": package.tool_names_used(),
        "tool_argument_errors": package.tool_argument_errors(),
        "tool_reliability_metrics": package.tool_reliability_metrics(),
        "approval_summary": package.approval_summaries(),
        "human_interactions": _human_interactions_summary(package.transcript),
        "final_response": final_response,
    }


def _expectations_summary(
    metadata: EpisodeMetadata,
    failure_record: FailureRecord,
    final_response: dict[str, Any] | None,
) -> dict[str, Any]:
    failure = _failure_summary(failure_record)
    return {
        "final_status": metadata.status,
        "failure_category": failure["category"] if failure else None,
        "final_response": {
            "mode": "contains",
            "value": final_response["content"] if final_response else "",
        },
    }


def _failure_summary(record: FailureRecord) -> dict[str, Any] | None:
    if record.is_success or record.failure is None:
        return None
    return {
        "category": record.failure.category,
        "stage": record.failure.stage,
        "evidence": record.failure.evidence,
    }


def _verification_summary(records: list[VerificationCommandRecord]) -> list[dict[str, Any]]:
    return [
        {
            "command": record.command,
            "status": record.status,
            "exit_code": record.exit_code,
            "timeout": record.timeout,
            "stdout_excerpt": record.stdout_excerpt,
            "stderr_excerpt": record.stderr_excerpt,
            "stdout_truncated": record.stdout_truncated,
            "stderr_truncated": record.stderr_truncated,
            "stdout_original_length": record.stdout_original_length,
            "stderr_original_length": record.stderr_original_length,
            "redacted": record.redacted,
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


def _environment_summary(environment: EnvironmentRecord) -> dict[str, Any]:
    return {
        "python": environment.python,
        "platform": environment.platform,
        "haagent_version": environment.haagent_version,
        "model_provider": environment.model.provider,
        "model": environment.model.model,
        "endpoint": environment.model.endpoint,
        "allowed_tool_count": environment.tools.allowed_tool_count,
    }


def _cost_summary(cost: CostRecord) -> dict[str, Any]:
    return {
        "usage_available": cost.usage_available,
        "pricing_available": cost.pricing_available,
        "model_call_count": cost.totals.model_call_count,
        "input_tokens": cost.totals.input_tokens,
        "output_tokens": cost.totals.output_tokens,
        "total_tokens": cost.totals.total_tokens,
        "estimated_cost": cost.estimated_cost,
        "currency": cost.currency,
        "reason": cost.reason,
    }


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


def _final_response_summary(package: EpisodePackage) -> dict[str, Any] | None:
    response = package.last_model_response()
    if response is None:
        return None
    tool_calls = response.get("tool_calls", [])
    return {
        "provider": str(response.get("provider", "unknown")),
        "turn": response.get("turn"),
        "content": str(response.get("content", "")),
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
    }
