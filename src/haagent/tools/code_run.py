"""
haagent/tools/code_run.py - Python 脚本执行工具

把多行 Python 代码写入系统临时脚本后执行，避免 shell 转义复杂脚本，且不污染 workspace。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.command import (
    CWD_GUIDANCE,
    build_python_utf8_environment,
    normalize_timeout,
    run_process,
)
from haagent.runtime.execution.path_policy import PathPolicy, default_path_policy
from haagent.runtime.sandbox.base import SandboxBackend, SandboxCommand
from haagent.tools.base import RecoveryAction, ToolExecutionContext, ToolFailureCategory, tool_error
from haagent.tools.path_access import resolve_tool_paths


def code_run(
    args: dict[str, Any],
    workspace_root: Path,
    path_policy: PathPolicy | None = None,
    cancellation_token: CancellationToken | None = None,
    sandbox_backend: SandboxBackend | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, Any]:
    code = args.get("code")
    if not isinstance(code, str) or not code:
        return tool_error("tool_argument_invalid", "code must be a non-empty string")

    cwd_arg = args.get("cwd")
    if cwd_arg is not None and not isinstance(cwd_arg, str):
        return tool_error("tool_argument_invalid", f"cwd must be a string; {CWD_GUIDANCE}")
    external_directories = args.get("external_directories", [])
    if not isinstance(external_directories, list) or any(
        not isinstance(path, str) or not path for path in external_directories
    ):
        return tool_error(
            "tool_argument_invalid",
            "external_directories must be a list of non-empty paths",
        )
    policy = path_policy or default_path_policy(workspace_root)
    requested_paths = ["." if cwd_arg in (None, ".") else cwd_arg, *external_directories]
    scope = resolve_tool_paths(requested_paths, policy, "full", execution_context)
    if isinstance(scope, dict):
        return scope
    cwd_result = scope[0]
    if not cwd_result.exists():
        return tool_error("tool_argument_invalid", f"cwd does not exist: {cwd_arg}; {CWD_GUIDANCE}")
    if not cwd_result.is_dir():
        return tool_error("tool_argument_invalid", f"cwd must be a directory: {cwd_arg}; {CWD_GUIDANCE}")
    for directory in scope[1:]:
        if not directory.exists() or not directory.is_dir():
            return tool_error(
                "tool_argument_invalid",
                f"declared external directory does not exist: {directory}",
            )

    timeout_result = normalize_timeout(args.get("timeout_seconds"))
    if isinstance(timeout_result, str):
        return tool_error("tool_argument_invalid", timeout_result)

    # 任意 Python 代码无法可靠静态分析；调用方必须显式声明外部目录，真实隔离由 sandbox 提供。
    # 脚本放系统 temp，cwd 仍指向 workspace，避免污染用户项目树。
    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="haagent-code-run-",
            suffix=".py",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as handle:
            handle.write(code)
            script_path = Path(handle.name)

        if sandbox_backend is None:
            command_result = run_process(
                command=f"{sys.executable} -X utf8 {script_path}",
                popen_args=[sys.executable, "-X", "utf8", str(script_path)],
                shell=False,
                cwd=cwd_result,
                timeout_seconds=timeout_result,
                cancellation_token=cancellation_token,
                env=build_python_utf8_environment(),
            )
        else:
            command_result = sandbox_backend.run_python(
                script_path,
                SandboxCommand(
                    command=f"python -X utf8 {script_path}",
                    cwd=cwd_result,
                    timeout_seconds=timeout_result,
                    cancellation_token=cancellation_token,
                    env=build_python_utf8_environment(inherit=False),
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
            # 诊断字段：绝对路径；成功失败后文件均已删除，复盘依赖 episode 中的 code 参数。
            "script_path": str(script_path.resolve()),
        }
        if command_result.status == "timeout":
            result.update(
                tool_error("timeout", f"python code timed out after {timeout_result} seconds", retryable=False, execution_state="unknown"),
            )
        elif command_result.status == "cancelled":
            result.update(tool_error("cancelled", "python code cancelled by user", retryable=False, execution_state="unknown"))
        elif command_result.status == "failed":
            result.update(
                tool_error(
                    "code_run_failed",
                    f"python code exited with code {command_result.exit_code}",
                    category=ToolFailureCategory.EXECUTION,
                    retryable=False,
                    recovery=RecoveryAction(
                        "correct_arguments",
                        "代码已完成但返回非零；查看 stderr 后修改 code，不要原样重试。",
                    ),
                    execution_state="completed",
                ),
            )
        return result
    finally:
        # 成功与失败都清理临时脚本，避免污染系统 temp 与 workspace。
        if script_path is not None:
            script_path.unlink(missing_ok=True)
