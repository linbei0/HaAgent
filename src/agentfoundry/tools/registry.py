"""
agentfoundry/tools/registry.py - Tool Registry v1

集中维护工具的可审计定义，供 ContextBuilder 和 ToolRouter 对齐工具集合。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: str
    parameters: dict[str, Any]


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "fake_tool": ToolDefinition(
        name="fake_tool",
        description="deterministic test tool",
        risk_level="low",
        parameters={"type": "object", "additionalProperties": True},
    ),
    "file_search": ToolDefinition(
        name="file_search",
        description="search workspace text using ripgrep when available",
        risk_level="low",
        parameters={"query": "text to search for in workspace files"},
    ),
    "file_read": ToolDefinition(
        name="file_read",
        description="read a workspace text file with offset and limit",
        risk_level="low",
        parameters={
            "path": "workspace-relative file path",
            "offset": "optional zero-based line offset",
            "limit": "optional maximum number of lines",
        },
    ),
    "apply_patch": ToolDefinition(
        name="apply_patch",
        description="replace unique text inside a workspace file",
        risk_level="high",
        parameters={
            "path": "workspace-relative file path",
            "old_text": "unique text to replace",
            "new_text": "replacement text",
        },
    ),
    "shell": ToolDefinition(
        name="shell",
        description="run a shell command with timeout and captured output",
        risk_level="high",
        parameters={
            "command": "shell command to execute",
            "cwd": "optional workspace-relative working directory",
            "timeout_seconds": "optional timeout in seconds",
        },
    ),
}
