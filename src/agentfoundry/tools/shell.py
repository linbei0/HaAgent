"""
agentfoundry/tools/shell.py - shell 本地工具

执行命令并捕获 timeout、exit_code、stdout 和 stderr。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentfoundry.runtime.command import run_command
from agentfoundry.tools.base import tool_error
from agentfoundry.tools.file_tools import resolve_workspace_path


def shell(args: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """运行 shell 命令，捕获 stdout/stderr/exit_code，并把失败结构化返回。"""
    command = args.get("command")
    if not isinstance(command, str) or not command:
        return tool_error("invalid_arguments", "command must be a non-empty string")

    cwd_arg = args.get("cwd", ".")
    if not isinstance(cwd_arg, str):
        return tool_error("invalid_arguments", "cwd must be a string")
    cwd = resolve_workspace_path(cwd_arg, workspace_root)
    if cwd is None:
        return tool_error("path_outside_workspace", "cwd must be inside workspace")

    timeout_seconds = float(args.get("timeout_seconds", 60))
    command_result = run_command(command, cwd, timeout_seconds)
    result = {
        "status": "success" if command_result.status == "success" else "error",
        "exit_code": command_result.exit_code,
        "stdout": command_result.stdout,
        "stderr": command_result.stderr,
    }
    if command_result.status == "timeout":
        result["error"] = {
            "type": "timeout",
            "message": f"command timed out after {timeout_seconds} seconds",
        }
    elif command_result.status == "failed":
        result["error"] = {
            "type": "command_failed",
            "message": f"command exited with code {command_result.exit_code}",
        }
    return result
