"""
haagent/tools/contributions/jobs.py - 后台 job 静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.contribution_helpers import (
    compact_excerpt,
    first_present_string,
    interaction_summary_value,
    shell_guardrail,
    summary_value,
)
from haagent.tools.jobs import job_cancel, job_logs, job_start, job_status


def _bind_job_start(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return job_start(
            args,
            workspace_root=deps.workspace_root,
            path_policy=deps.path_policy,
            job_manager=deps.job_manager,
            execution_context=context,
        )

    return handler


def _bind_job_status(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        del context
        return job_status(args, job_manager=deps.job_manager)

    return handler


def _bind_job_logs(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        del context
        return job_logs(args, job_manager=deps.job_manager)

    return handler


def _bind_job_cancel(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        del context
        return job_cancel(args, job_manager=deps.job_manager)

    return handler


def _job_start_interaction(args: dict[str, Any]) -> dict[str, object]:
    return {
        "command": interaction_summary_value(str(args.get("command", "")), 160),
        "cwd": str(args.get("cwd", ".")),
        "timeout_seconds": args.get("timeout_seconds"),
    }


def _job_id_interaction(args: dict[str, Any]) -> dict[str, object]:
    return {"job_id": str(args.get("job_id", ""))}


def _job_start_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "pid": result.get("pid"),
        "timeout_seconds": result.get("timeout_seconds"),
    }


def _job_status_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout"),
        "duration_seconds": result.get("duration_seconds"),
        "waited_seconds": result.get("waited_seconds"),
        "stdout_excerpt": summary_value(str(result.get("stdout_excerpt", "")), 300),
        "stderr_excerpt": summary_value(str(result.get("stderr_excerpt", "")), 300),
        "logs_truncated": bool(result.get("logs_truncated")),
    }


def _job_logs_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "stdout_excerpt": summary_value(str(result.get("stdout_excerpt", result.get("stdout", ""))), 300),
        "stderr_excerpt": summary_value(str(result.get("stderr_excerpt", result.get("stderr", ""))), 300),
        "truncated": bool(result.get("truncated")),
    }


def _job_cancel_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "exit_code": result.get("exit_code"),
    }


def _job_start_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    return {
        "status": result.get("status", "unknown"),
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "command": first_present_string(result.get("command"), args.get("command")),
        "pid": result.get("pid"),
    }


def _job_status_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    del args
    return {
        "status": result.get("status", "unknown"),
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout", False),
        "duration_seconds": result.get("duration_seconds"),
        "waited_seconds": result.get("waited_seconds"),
        "stdout": compact_excerpt(str(result.get("stdout_excerpt", "")))[0],
        "stderr": compact_excerpt(str(result.get("stderr_excerpt", "")))[0],
        "logs_truncated": result.get("logs_truncated", False),
    }


def _job_logs_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    del args
    stdout = first_present_string(result.get("stdout_excerpt"), result.get("stdout"))
    stderr = first_present_string(result.get("stderr_excerpt"), result.get("stderr"))
    return {
        "status": result.get("status", "unknown"),
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "stdout": compact_excerpt(stdout)[0],
        "stderr": compact_excerpt(stderr)[0],
        "truncated": result.get("truncated", False),
    }


def _job_cancel_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    del args
    return {
        "status": result.get("status", "unknown"),
        "job_id": result.get("job_id"),
        "job_status": result.get("job_status"),
        "exit_code": result.get("exit_code"),
    }


JOB_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="job_start",
        description=(
            "Start a long-running shell command in the background and return job_id immediately. "
            "Use for builds, full test suites, downloads, or any work that may exceed the 120s shell timeout. "
            "Keep short commands on shell/code_run. After start, call job_status; it waits before returning while "
            "the job is running and includes recent logs at terminal state. Use job_logs only when live output is "
            "needed for diagnosis. Stop with job_cancel when needed. Set cwd instead of embedding cd."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "shell command to run in the background under cwd; quote paths with spaces; "
                        "avoid embedding cd"
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'absolute or relative to workspace_root working directory; use "." or omit '
                        "for workspace_root; external directories require user permission"
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": (
                        "optional wall-clock timeout for the background job; defaults to 3600 and must be <= 7200"
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_job_start,
        interaction_args_summary=_job_start_interaction,
        summarize_result=_job_start_result,
        project_observation=_job_start_observation,
        guardrail=shell_guardrail,
        display_name_zh="后台启动",
    ),
    ToolContribution(
        name="job_status",
        description=(
            "Wait for and check a background job started by job_start. By default this blocks for up to 30 seconds "
            "while the job is running, returns early on terminal state, and includes recent terminal logs. Repeat only "
            "when job_status is still running. Use wait_seconds=0 only for an explicit immediate snapshot. Routine "
            "wait checks do not need separate narration because the activity UI already shows progress."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "job_id returned by job_start",
                },
                "wait_seconds": {
                    "type": "number",
                    "description": (
                        "seconds to wait for terminal state; defaults to 30, maximum 60; use 0 for an immediate snapshot"
                    ),
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_default"}),
        bind_handler=_bind_job_status,
        interaction_args_summary=_job_id_interaction,
        summarize_result=_job_status_result,
        project_observation=_job_status_observation,
        display_name_zh="任务状态",
        observation_long_text_keys=("stdout", "stderr"),
    ),
    ToolContribution(
        name="job_logs",
        description=(
            "Read recent stdout/stderr from a background job. Output is redacted and truncated; "
            "use only when live output is needed or terminal excerpts from job_status are insufficient."
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "job_id returned by job_start",
                },
                "stream": {
                    "type": "string",
                    "enum": ["stdout", "stderr", "both"],
                    "description": "which stream to read; defaults to both",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "max characters to return from the tail of each stream; default 4000",
                },
                "offset": {
                    "type": "integer",
                    "description": "optional character offset into the log file before taking the tail",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        execution_effect="read_only",
        replay_safety=ReplaySafety.SAFE_TO_REPLAY,
        tags=frozenset({"chat_default"}),
        bind_handler=_bind_job_logs,
        interaction_args_summary=_job_id_interaction,
        summarize_result=_job_logs_result,
        project_observation=_job_logs_observation,
        display_name_zh="任务日志",
        observation_long_text_keys=("stdout", "stderr"),
    ),
    ToolContribution(
        name="job_cancel",
        description=(
            "Cancel a background job started by job_start and stop its process tree. "
            "Safe to call if the job already finished."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "job_id returned by job_start",
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_job_cancel,
        interaction_args_summary=_job_id_interaction,
        summarize_result=_job_cancel_result,
        project_observation=_job_cancel_observation,
        display_name_zh="取消任务",
    ),
]
