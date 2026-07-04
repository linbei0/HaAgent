"""
haagent/tools/shell.py - shell 本地工具

执行命令并捕获 timeout、exit_code、stdout 和 stderr。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.execution.command import CWD_GUIDANCE, normalize_timeout, resolve_execution_cwd, run_command
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, resolve_cwd_for_execution
from haagent.runtime.sandbox.base import SandboxBackend, SandboxCommand
from haagent.tools.base import tool_error


def shell(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    cancellation_token: CancellationToken | None = None,
    sandbox_backend: SandboxBackend | None = None,
) -> dict[str, Any]:
    """运行 shell 命令，捕获 stdout/stderr/exit_code，并把失败结构化返回。"""
    command = args.get("command")
    if not isinstance(command, str) or not command:
        return tool_error("tool_argument_invalid", "command must be a non-empty string")

    cwd_arg = args.get("cwd")
    if cwd_arg is not None and not isinstance(cwd_arg, str):
        return tool_error("tool_argument_invalid", f"cwd must be a string; {CWD_GUIDANCE}")
    if path_policy is None:
        cwd_result = resolve_execution_cwd(cwd_arg, workspace_root)
    else:
        cwd_result = resolve_cwd_for_execution(cwd_arg, path_policy or default_path_policy(workspace_root))
    if isinstance(cwd_result, str):
        error_type = "path_policy_denied" if path_policy is not None else "tool_argument_invalid"
        return tool_error(error_type, cwd_result)

    timeout_result = normalize_timeout(args.get("timeout_seconds"))
    if isinstance(timeout_result, str):
        return tool_error("tool_argument_invalid", timeout_result)

    if sandbox_backend is None:
        command_result = run_command(command, cwd_result, timeout_result, cancellation_token=cancellation_token)
    else:
        command_result = sandbox_backend.run_shell(
            SandboxCommand(
                command=command,
                cwd=cwd_result,
                timeout_seconds=timeout_result,
                cancellation_token=cancellation_token,
            ),
        )
    result = {
        "status": "success" if command_result.status == "success" else "error",
        "exit_code": command_result.exit_code,
        "stdout_excerpt": command_result.stdout_excerpt,
        "stderr_excerpt": command_result.stderr_excerpt,
        "stdout_truncated": command_result.stdout_truncated,
        "stderr_truncated": command_result.stderr_truncated,
        "truncated": command_result.truncated,
        "timeout": command_result.timeout,
        "redacted": command_result.redacted,
        "timeout_seconds": command_result.timeout_seconds,
    }
    if command_result.status == "timeout":
        result["error"] = {
            "type": "timeout",
            "message": f"command timed out after {timeout_result} seconds",
        }
    elif command_result.status == "cancelled":
        result["error"] = {
            "type": "cancelled",
            "message": "command cancelled by user",
        }
    elif command_result.status == "failed":
        result["error"] = {
            "type": "command_failed",
            "message": f"command exited with code {command_result.exit_code}",
        }
    return result
