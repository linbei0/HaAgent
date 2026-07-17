"""
haagent/tools/contributions/shell.py - 命令执行静态工具 contribution
"""

from __future__ import annotations

from typing import Any

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools.base import ToolExecutionContext, ToolHandler
from haagent.tools.catalog import ToolContribution, ToolRuntimeDeps
from haagent.tools.code_run import code_run
from haagent.tools.contribution_helpers import (
    code_run_guardrail,
    compact_excerpt,
    first_present_string,
    interaction_summary_value,
    shell_guardrail,
    summary_value,
)
from haagent.tools.shell import shell


def _bind_code_run(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return code_run(
            args,
            deps.workspace_root,
            deps.path_policy,
            cancellation_token=deps.cancellation_token,
            sandbox_backend=deps.sandbox_backend,
        )

    return handler


def _bind_shell(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], _context: ToolExecutionContext) -> dict[str, Any]:
        return shell(
            args,
            deps.workspace_root,
            deps.path_policy,
            cancellation_token=deps.cancellation_token,
            sandbox_backend=deps.sandbox_backend,
        )

    return handler


def _code_run_interaction(args: dict[str, Any]) -> dict[str, object]:
    code = str(args.get("code", ""))
    return {
        "code_chars": len(code),
        "cwd": str(args.get("cwd", ".")),
        "timeout_seconds": args.get("timeout_seconds"),
    }


def _code_run_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "exit_code": result.get("exit_code"),
        "stdout_excerpt": summary_value(str(result.get("stdout_excerpt", "")), 300),
        "stderr_excerpt": summary_value(str(result.get("stderr_excerpt", "")), 300),
        "stdout_chars": len(str(result.get("stdout_excerpt", ""))),
        "stderr_chars": len(str(result.get("stderr_excerpt", ""))),
        "truncated": bool(result.get("truncated")),
    }


def _code_run_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    del args
    stdout = first_present_string(result.get("stdout_excerpt"), result.get("stdout"))
    stderr = first_present_string(result.get("stderr_excerpt"), result.get("stderr"))
    return {
        "status": result.get("status", "unknown"),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout", False),
        "stdout": compact_excerpt(stdout)[0],
        "stderr": compact_excerpt(stderr)[0],
        "truncated": result.get("truncated", False),
    }


def _shell_interaction(args: dict[str, Any]) -> dict[str, object]:
    return {
        "command": interaction_summary_value(str(args.get("command", "")), 160),
        "cwd": str(args.get("cwd", ".")),
        "timeout_seconds": args.get("timeout_seconds"),
    }


def _shell_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "exit_code": result.get("exit_code"),
        "stdout_excerpt": summary_value(str(result.get("stdout_excerpt", "")), 300),
        "stderr_excerpt": summary_value(str(result.get("stderr_excerpt", "")), 300),
        "stdout_chars": len(str(result.get("stdout_excerpt", ""))),
        "stderr_chars": len(str(result.get("stderr_excerpt", ""))),
        "timeout": bool(result.get("timeout")),
        "truncated": bool(result.get("truncated")),
    }


def _shell_observation(args: dict[str, Any], result: dict[str, Any]) -> dict[str, object]:
    stdout = first_present_string(result.get("stdout_excerpt"), result.get("stdout"))
    stderr = first_present_string(result.get("stderr_excerpt"), result.get("stderr"))
    return {
        "status": result.get("status", "unknown"),
        "command": first_present_string(result.get("command"), args.get("command")),
        "exit_code": result.get("exit_code"),
        "timeout": result.get("timeout", False),
        "stdout": compact_excerpt(stdout)[0],
        "stderr": compact_excerpt(stderr)[0],
        "truncated": result.get("truncated", False),
    }


SHELL_CONTRIBUTIONS: list[ToolContribution] = [
    ToolContribution(
        name="code_run",
        description="run a multiline Python script from a temporary workspace file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to write to a temporary script and execute",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds; defaults to 60 and must be <= 120",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'working directory relative to workspace_root; use "." or omit '
                        "for workspace root"
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_code_run,
        interaction_args_summary=_code_run_interaction,
        summarize_result=_code_run_result,
        project_observation=_code_run_observation,
        guardrail=code_run_guardrail,
        display_name_zh="运行代码",
        observation_long_text_keys=("stdout", "stderr"),
    ),
    ToolContribution(
        name="shell",
        description="run a shell command with timeout and captured output",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'working directory relative to workspace_root; use "." or omit '
                        "for workspace root"
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds; defaults to 60 and must be <= 120",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
        replay_safety=ReplaySafety.NEVER_REPLAY,
        tags=frozenset({"chat_default", "chat_approval"}),
        bind_handler=_bind_shell,
        interaction_args_summary=_shell_interaction,
        summarize_result=_shell_result,
        project_observation=_shell_observation,
        guardrail=shell_guardrail,
        display_name_zh="运行命令",
        observation_long_text_keys=("stdout", "stderr"),
    ),
]
