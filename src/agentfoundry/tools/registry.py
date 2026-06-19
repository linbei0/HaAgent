"""
agentfoundry/tools/registry.py - Tool Registry v1

集中维护工具的可审计定义，并导出模型网关可用的最小 JSON Schema。
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

    def to_model_schema(self) -> dict[str, Any]:
        """导出模型网关需要的稳定字段，不暴露运行时内部风险元数据。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "fake_tool": ToolDefinition(
        name="fake_tool",
        description="deterministic test tool",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        },
    ),
    "file_search": ToolDefinition(
        name="file_search",
        description="search workspace text using ripgrep when available",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "text to search for in workspace files",
                },
                "root": {
                    "type": "string",
                    "description": "optional workspace-relative directory to search",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "file_read": ToolDefinition(
        name="file_read",
        description="read a workspace text file with offset and limit",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "offset": {
                    "type": "integer",
                    "description": "optional zero-based line offset",
                },
                "limit": {
                    "type": "integer",
                    "description": "optional maximum number of lines",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    "apply_patch": ToolDefinition(
        name="apply_patch",
        description="replace unique text inside a workspace file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "old_text": {
                    "type": "string",
                    "description": "unique text to replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "replacement text",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    ),
    "shell": ToolDefinition(
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
                    "description": "optional workspace-relative working directory",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "optional timeout in seconds",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    ),
}


def get_tool_definition(name: str) -> ToolDefinition:
    """按名称读取工具定义；未知工具必须显式失败，避免静默漏导 schema。"""
    try:
        return TOOL_REGISTRY[name]
    except KeyError as error:
        raise KeyError(f"unknown tool: {name}") from error


def allowed_tool_definitions(names: list[str]) -> list[ToolDefinition]:
    return [get_tool_definition(name) for name in names]


def export_tool_schemas(names: list[str]) -> list[dict[str, Any]]:
    return [
        definition.to_model_schema()
        for definition in allowed_tool_definitions(names)
    ]
