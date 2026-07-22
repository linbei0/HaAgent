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
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return code_run(
            args,
            deps.workspace_root,
            deps.path_policy,
            cancellation_token=deps.cancellation_token,
            sandbox_backend=deps.sandbox_backend,
            execution_context=context,
        )

    return handler


def _bind_shell(deps: ToolRuntimeDeps) -> ToolHandler:
    def handler(args: dict[str, Any], context: ToolExecutionContext) -> dict[str, Any]:
        return shell(
            args,
            deps.workspace_root,
            deps.path_policy,
            cancellation_token=deps.cancellation_token,
            sandbox_backend=deps.sandbox_backend,
            execution_context=context,
        )

    return handler


def _code_run_interaction(args: dict[str, Any]) -> dict[str, object]:
    code = str(args.get("code", ""))
    summary: dict[str, object] = {
        "code_chars": len(code),
        "cwd": str(args.get("cwd", ".")),
        "timeout_seconds": args.get("timeout_seconds"),
    }
    external_directories = args.get("external_directories")
    if isinstance(external_directories, list) and external_directories:
        summary["external_directories"] = list(external_directories)
    return summary


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
        description=(
            "Run multiline Python for calculations, data transformation, or Python-specific automation. "
            "Use shell for tests, builds, git, package managers, and existing project scripts. Use file_list, "
            "grep, file_read, and patch tools for file discovery and editing. Declare every workspace-external "
            "directory the code will access; do not retry a failed write-like script without inspecting its state."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "complete Python program to write to a system temporary file and execute "
                        "(cleaned up after the run); print the concise result or diagnostics needed by the next step"
                    ),
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds; defaults to 60 and must be <= 120",
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        'absolute or relative to workspace_root working directory; use "." or omit '
                        "for workspace_root; external directories require user permission"
                    ),
                },
                "external_directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "workspace-external directories the Python code will access; "
                        "HaAgent requests directory permission before execution"
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
        description=(
            "Run tests, builds, package managers, git, or an existing project command with captured output. "
            "Timeout defaults to 60s and must be <= 120s. For work that may take longer, use job_start then "
            "job_status/job_logs instead of raising timeout_seconds. Set cwd instead of embedding cd. Prefer "
            "file_list, grep, file_read, file_write, and patch tools for file operations. Quote paths containing "
            "spaces. Independent read-only commands may be issued in parallel; after failure, inspect exit_code "
            "and stderr before changing or retrying it."
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "single shell command to execute in cwd; quote paths containing spaces and avoid an "
                        "embedded cd because cwd selects the working directory; use the runtime shell contract "
                        "provided in this tool description"
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
