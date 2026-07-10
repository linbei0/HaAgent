"""
haagent/tools/handler_factory.py - 静态工具 handler 组合

将已解析的运行时依赖绑定为 ToolRouter 使用的静态 handler 映射。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.path_policy import PathPolicy
from haagent.runtime.sandbox.base import SandboxBackend
from haagent.skills import SkillSettings
from haagent.tools.base import ToolHandler
from haagent.tools.code_run import code_run
from haagent.tools.file_tools import apply_patch, apply_patch_set, file_list, file_read, file_write, grep
from haagent.tools.mcp_tools import list_mcp_resources, read_mcp_resource
from haagent.tools.shell import shell
from haagent.tools.skill_market import skill_market_search
from haagent.tools.skills import skill_list, skill_read
from haagent.tools.web import web_fetch, web_search


def build_static_tool_handlers(
    *,
    workspace_root: Path,
    path_policy: PathPolicy,
    skill_settings: SkillSettings | None,
    cancellation_token: CancellationToken | None,
    mcp_runtime: Any | None,
    sandbox_backend: SandboxBackend | None,
    router_handlers: dict[str, ToolHandler],
) -> dict[str, ToolHandler]:
    """组合静态工具 handler；策略、审批和审计仍由 router 负责。"""
    return {
        "fake_tool": router_handlers["fake_tool"],
        "load_image_attachment": router_handlers["load_image_attachment"],
        "agent": router_handlers["agent"],
        "send_message": router_handlers["send_message"],
        "task_stop": router_handlers["task_stop"],
        "task_get": router_handlers["task_get"],
        "task_list": router_handlers["task_list"],
        "task_output": router_handlers["task_output"],
        "file_list": lambda args: file_list(args, workspace_root, path_policy),
        "grep": lambda args: grep(args, workspace_root, path_policy),
        "file_read": lambda args: file_read(args, workspace_root, path_policy),
        "request_user_input": router_handlers["request_user_input"],
        "start_memory_update": router_handlers["start_memory_update"],
        "skill_list": lambda args: skill_list(args, workspace_root, skill_settings),
        "skill_read": lambda args: skill_read(args, workspace_root, skill_settings),
        "skill_market_search": skill_market_search,
        "web_search": web_search,
        "web_fetch": web_fetch,
        "list_mcp_resources": lambda args: list_mcp_resources(args, mcp_runtime),
        "read_mcp_resource": lambda args: read_mcp_resource(args, mcp_runtime),
        "file_write": lambda args: file_write(args, workspace_root, path_policy),
        "code_run": lambda args: code_run(
            args,
            workspace_root,
            path_policy,
            cancellation_token=cancellation_token,
            sandbox_backend=sandbox_backend,
        ),
        "apply_patch": lambda args: apply_patch(args, workspace_root, path_policy),
        "apply_patch_set": lambda args: apply_patch_set(args, workspace_root, path_policy),
        "shell": lambda args: shell(
            args,
            workspace_root,
            path_policy,
            cancellation_token=cancellation_token,
            sandbox_backend=sandbox_backend,
        ),
    }
