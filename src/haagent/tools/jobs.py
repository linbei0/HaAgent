"""
haagent/tools/jobs.py - 后台 job 工具

提供 job_start / job_status / job_logs / job_cancel；长任务后台跑，
单次 shell 保持短超时。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from haagent.runtime.execution.command import CWD_GUIDANCE
from haagent.runtime.execution.jobs import (
    JobManager,
    default_job_manager,
    normalize_job_timeout,
    normalize_job_wait,
)
from haagent.runtime.execution.path_policy import PathPolicy, classify_path_access, default_path_policy
from haagent.tools.base import ToolExecutionContext, tool_error
from haagent.tools.path_access import resolve_tool_paths
from haagent.tools.shell_paths import collect_shell_paths


def job_start(
    args: dict[str, Any],
    *,
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    job_manager: JobManager | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """启动后台命令，立即返回 job_id；不阻塞等待进程结束。"""
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return tool_error("tool_argument_invalid", "command must be a non-empty string")

    cwd_arg = args.get("cwd")
    if cwd_arg is not None and not isinstance(cwd_arg, str):
        return tool_error("tool_argument_invalid", f"cwd must be a string; {CWD_GUIDANCE}")
    policy = path_policy or default_path_policy(workspace_root)
    cwd_path = "." if cwd_arg in (None, ".") else cwd_arg
    cwd_preflight = classify_path_access(cwd_path, policy, "full")
    if cwd_preflight.path is None:
        return tool_error(cwd_preflight.error_type, cwd_preflight.message, retryable=False)
    cwd_result = cwd_preflight.path
    if not cwd_result.exists():
        return tool_error("tool_argument_invalid", f"cwd does not exist: {cwd_arg}; {CWD_GUIDANCE}")
    if not cwd_result.is_dir():
        return tool_error("tool_argument_invalid", f"cwd must be a directory: {cwd_arg}; {CWD_GUIDANCE}")

    command_paths = collect_shell_paths(command, cwd=cwd_result)
    command_scope = resolve_tool_paths([cwd_result, *command_paths], policy, "full", execution_context)
    if isinstance(command_scope, dict):
        return command_scope
    cwd_result = command_scope[0]

    timeout_result = normalize_job_timeout(args.get("timeout_seconds"))
    if isinstance(timeout_result, str):
        return tool_error("tool_argument_invalid", timeout_result)

    manager = job_manager or default_job_manager()
    record = manager.start(
        command=command,
        cwd=cwd_result,
        workspace_root=workspace_root,
        timeout_seconds=timeout_result,
    )
    if record.status == "failed" and record.pid is None:
        return {
            **tool_error(
                "job_start_failed",
                record.error_message or "failed to start background job",
                retryable=False,
            ),
            "job_id": record.job_id,
            "job_status": record.status,
            "command": record.command,
            "cwd": record.cwd,
            "timeout_seconds": record.timeout_seconds,
        }
    return {
        "status": "success",
        "job_id": record.job_id,
        "job_status": record.status,
        "command": record.command,
        "cwd": record.cwd,
        "pid": record.pid,
        "timeout_seconds": record.timeout_seconds,
        "message": "background job started; wait with job_status; terminal status includes recent logs",
    }


def job_status(
    args: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """等待并查询后台 job 状态。"""
    job_id = args.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        return tool_error("tool_argument_invalid", "job_id must be a non-empty string")
    wait_result = normalize_job_wait(args.get("wait_seconds"))
    if isinstance(wait_result, str):
        return tool_error("tool_argument_invalid", wait_result)
    manager = job_manager or default_job_manager()
    started = time.monotonic()
    record = manager.status(job_id.strip(), wait_seconds=wait_result)
    waited_seconds = max(0.0, time.monotonic() - started)
    if record is None:
        return tool_error("job_not_found", f"unknown job_id: {job_id}", retryable=False)
    public = record.to_public_dict()
    result = {
        "status": "success",
        "job_id": public["job_id"],
        "job_status": public["status"],
        "command": public["command"],
        "cwd": public["cwd"],
        "pid": public["pid"],
        "exit_code": public["exit_code"],
        "timeout": public["timeout"],
        "timeout_seconds": public["timeout_seconds"],
        "duration_seconds": public["duration_seconds"],
        "created_at": public["created_at"],
        "updated_at": public["updated_at"],
        "finished_at": public["finished_at"],
        "error_message": public["error_message"],
        "wait_seconds": wait_result,
        "waited_seconds": waited_seconds,
    }
    if public["status"] in {"finished", "failed", "timeout", "cancelled"}:
        # 终态直接附带有界日志摘要，避免模型再发一次 job_logs 才能形成最终回答。
        logs = manager.logs(job_id.strip(), max_chars=4000)
        if logs is not None:
            result.update(
                {
                    "stdout_excerpt": logs["stdout_excerpt"],
                    "stderr_excerpt": logs["stderr_excerpt"],
                    "logs_truncated": logs["truncated"],
                    "logs_redacted": logs["redacted"],
                },
            )
    return result


def job_logs(
    args: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """读取后台 job 日志尾部（脱敏摘要）。"""
    job_id = args.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        return tool_error("tool_argument_invalid", "job_id must be a non-empty string")
    stream = args.get("stream", "both")
    if stream is not None and stream not in {"stdout", "stderr", "both"}:
        return tool_error("tool_argument_invalid", "stream must be stdout, stderr, or both")
    max_chars = args.get("max_chars", 4000)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        return tool_error("tool_argument_invalid", "max_chars must be an integer")
    offset = args.get("offset", 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return tool_error("tool_argument_invalid", "offset must be a non-negative integer")

    manager = job_manager or default_job_manager()
    payload = manager.logs(
        job_id.strip(),
        stream=str(stream or "both"),
        max_chars=max_chars,
        offset=offset,
    )
    if payload is None:
        return tool_error("job_not_found", f"unknown job_id: {job_id}", retryable=False)
    return {
        "status": "success",
        **payload,
    }


def job_cancel(
    args: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """取消后台 job（杀进程树）。"""
    job_id = args.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        return tool_error("tool_argument_invalid", "job_id must be a non-empty string")
    manager = job_manager or default_job_manager()
    record = manager.cancel(job_id.strip())
    if record is None:
        return tool_error("job_not_found", f"unknown job_id: {job_id}", retryable=False)
    return {
        "status": "success",
        "job_id": record.job_id,
        "job_status": record.status,
        "exit_code": record.exit_code,
        "timeout": record.timeout,
        "message": "background job cancelled" if record.status == "cancelled" else f"job already {record.status}",
    }
