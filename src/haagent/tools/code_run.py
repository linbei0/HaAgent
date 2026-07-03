"""
haagent/tools/code_run.py - Python 脚本执行工具

把多行 Python 代码写入工作区临时脚本后执行，避免 shell 转义复杂脚本。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.command import (
    CWD_GUIDANCE,
    normalize_timeout,
    resolve_execution_cwd,
    run_process,
)
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy, resolve_cwd_for_execution
from haagent.tools.base import tool_error


def code_run(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    code = args.get("code")
    if not isinstance(code, str) or not code:
        return tool_error("tool_argument_invalid", "code must be a non-empty string")

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

    root = workspace_root.resolve()
    tmp_dir = root / ".haagent-tmp"
    tmp_dir.mkdir(exist_ok=True)
    script_path = tmp_dir / f"code-run-{uuid.uuid4().hex[:12]}.py"
    script_path.write_text(code, encoding="utf-8")

    command_result = run_process(
        command=f"{sys.executable} {script_path}",
        popen_args=[sys.executable, str(script_path)],
        shell=False,
        cwd=cwd_result,
        timeout_seconds=timeout_result,
        cancellation_token=cancellation_token,
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
        "script_path": script_path.relative_to(root).as_posix(),
    }
    if command_result.status == "timeout":
        result["error"] = {
            "type": "timeout",
            "message": f"python code timed out after {timeout_result} seconds",
        }
    elif command_result.status == "cancelled":
        result["error"] = {
            "type": "cancelled",
            "message": "python code cancelled by user",
        }
    elif command_result.status == "failed":
        result["error"] = {
            "type": "code_run_failed",
            "message": f"python code exited with code {command_result.exit_code}",
        }
    return result
