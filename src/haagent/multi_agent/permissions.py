"""
src/haagent/multi_agent/permissions.py - worker 工具权限分层

根据 worker 类型生成最小工具集合，避免子智能体绕过 HaAgent 的 ToolRouter 边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WorkerType = Literal["explorer", "worker", "verification"]


@dataclass(frozen=True)
class WorkerToolPolicy:
    allowed_tools: list[str]
    approval_allowed_tools: list[str]
    approved_tools: list[str]


def worker_tool_policy(
    subagent_type: WorkerType,
    *,
    inherited_allowed_tools: list[str],
    inherited_approval_allowed_tools: list[str],
    inherited_approved_tools: list[str],
    web_enabled: bool,
    mcp_tool_names: list[str],
) -> WorkerToolPolicy:
    if subagent_type == "worker":
        return WorkerToolPolicy(
            allowed_tools=list(inherited_allowed_tools),
            approval_allowed_tools=list(inherited_approval_allowed_tools),
            approved_tools=list(inherited_approved_tools),
        )
    if subagent_type == "explorer":
        allowed = ["file_list", "file_search", "file_read", "skill_list", "skill_read"]
        if web_enabled:
            allowed.extend(["web_search", "web_fetch"])
        if mcp_tool_names:
            allowed.extend(mcp_tool_names)
            allowed.extend(["list_mcp_resources", "read_mcp_resource"])
        return WorkerToolPolicy(
            allowed_tools=_dedupe(allowed),
            approval_allowed_tools=[],
            approved_tools=[],
        )
    if subagent_type == "verification":
        allowed = ["file_read", "file_search", "shell", "code_run"]
        approval_allowed = [
            tool
            for tool in inherited_approval_allowed_tools
            if tool in {"shell", "code_run"}
        ]
        approved = [
            tool
            for tool in inherited_approved_tools
            if tool in {"shell", "code_run"}
        ]
        return WorkerToolPolicy(
            allowed_tools=allowed,
            approval_allowed_tools=approval_allowed,
            approved_tools=approved,
        )
    raise ValueError(f"unknown subagent_type: {subagent_type}")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
