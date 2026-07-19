"""
haagent/tools/shell.py - shell 本地工具

执行命令并捕获 timeout、exit_code、stdout 和 stderr。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.execution.command import CWD_GUIDANCE, normalize_timeout, run_command
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.path_policy import PathPolicy, classify_path_access, default_path_policy
from haagent.runtime.sandbox.base import SandboxBackend, SandboxCommand
from haagent.tools.base import ToolExecutionContext, tool_error
from haagent.tools.path_access import resolve_tool_paths
from haagent.tools.shell_paths import collect_shell_paths


def shell(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    cancellation_token: CancellationToken | None = None,
    sandbox_backend: SandboxBackend | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    """运行 shell 命令，捕获 stdout/stderr/exit_code，并把失败结构化返回。"""
    command = args.get("command")
    if not isinstance(command, str) or not command:
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

    # 与 OpenCode 相同，这只是常见文件命令的静态预检；真正隔离仍由 sandbox backend 提供。
    command_paths = collect_shell_paths(command, cwd=cwd_result)
    command_scope = resolve_tool_paths([cwd_result, *command_paths], policy, "full", execution_context)
    if isinstance(command_scope, dict):
        return command_scope
    cwd_result = command_scope[0]

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
        result["execution_state"] = "unknown"
        result["error"] = {
            "type": "timeout",
            "message": f"command timed out after {timeout_result} seconds",
        }
    elif command_result.status == "cancelled":
        result["execution_state"] = "unknown"
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
