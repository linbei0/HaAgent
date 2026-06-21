"""
haagent/tools/shell.py - shell 本地工具

执行命令并捕获 timeout、exit_code、stdout 和 stderr。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.command import run_command
from haagent.tools.base import tool_error
from haagent.tools.file_tools import resolve_workspace_path


CWD_GUIDANCE = (
    'cwd is relative to workspace_root; use "." or omit cwd for workspace root'
)


def shell(args: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """运行 shell 命令，捕获 stdout/stderr/exit_code，并把失败结构化返回。"""
    command = args.get("command")
    if not isinstance(command, str) or not command:
        return tool_error("invalid_arguments", "command must be a non-empty string")

    cwd_arg = args.get("cwd")
    if cwd_arg is not None and not isinstance(cwd_arg, str):
        return tool_error("invalid_arguments", f"cwd must be a string; {CWD_GUIDANCE}")
    cwd_result = _resolve_cwd(cwd_arg, workspace_root)
    if isinstance(cwd_result, dict):
        return cwd_result
    cwd = cwd_result

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


def _resolve_cwd(cwd_arg: str | None, workspace_root: Path) -> Path | dict[str, Any]:
    if cwd_arg in (None, "."):
        cwd_arg = "."
    elif Path(cwd_arg).is_absolute():
        return tool_error(
            "tool_argument_invalid",
            f"cwd must be relative to workspace_root; {CWD_GUIDANCE}",
        )

    cwd = resolve_workspace_path(cwd_arg, workspace_root)
    if cwd is None:
        return tool_error(
            "tool_argument_invalid",
            f"cwd must stay inside workspace_root; {CWD_GUIDANCE}",
        )
    if not cwd.exists():
        return tool_error("tool_argument_invalid", f"cwd does not exist; {CWD_GUIDANCE}")
    if not cwd.is_dir():
        return tool_error("tool_argument_invalid", f"cwd must be a directory; {CWD_GUIDANCE}")
    return cwd
