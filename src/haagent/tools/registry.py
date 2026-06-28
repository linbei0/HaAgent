"""
haagent/tools/registry.py - Tool Registry v1

集中维护工具的可审计定义，并导出模型网关可用的最小 JSON Schema。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haagent.memory.prompts import START_MEMORY_UPDATE_TOOL_DESCRIPTION


ALLOWED_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


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
    "file_list": ToolDefinition(
        name="file_list",
        description="list a compact workspace file tree for project discovery",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": 'optional workspace-relative directory to list; defaults to "."',
                },
                "max_depth": {
                    "type": "integer",
                    "description": "optional maximum directory depth; defaults to 2",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "optional maximum entries to return; defaults to 100",
                },
            },
            "required": [],
            "additionalProperties": False,
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
    "context_find": ToolDefinition(
        name="context_find",
        description=(
            "primary choice for locating relevant workspace files and snippets from a natural language "
            "query when the user describes functionality without paths"
        ),
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "natural language query describing the context to find",
                },
                "file_glob": {
                    "type": "string",
                    "description": "optional file glob filter such as *.py or docs/*.md",
                },
                "max_results": {
                    "type": "integer",
                    "description": "optional maximum number of candidates; defaults to 5",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "optional total excerpt character budget; defaults to 1600",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "file_read": ToolDefinition(
        name="file_read",
        description="read a workspace text file with offset, limit, or keyword context",
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
                "keyword": {
                    "type": "string",
                    "description": "optional keyword; read lines near the first match",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    "request_user_input": ToolDefinition(
        name="request_user_input",
        description="ask the user for missing information before continuing the task",
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "question to ask the user",
                },
                "reason": {
                    "type": "string",
                    "description": "short reason why the information is needed",
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
    "start_memory_update": ToolDefinition(
        name="start_memory_update",
        description=START_MEMORY_UPDATE_TOOL_DESCRIPTION,
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "short reason describing the durable information that may be worth settlement",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    "file_write": ToolDefinition(
        name="file_write",
        description="create, overwrite, or append a workspace text file",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "workspace-relative file path",
                },
                "content": {
                    "type": "string",
                    "description": "text content to write",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create", "overwrite", "append"],
                    "description": "write mode: create, overwrite, or append",
                },
            },
            "required": ["path", "content", "mode"],
            "additionalProperties": False,
        },
    ),
    "code_run": ToolDefinition(
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
    "apply_patch_set": ToolDefinition(
        name="apply_patch_set",
        description=(
            "apply multiple unique text replacements atomically after reading current file context; "
            "no files are written if any replacement does not match exactly once. "
            "Prefer this over repeated apply_patch calls for related multi-file or multi-site edits"
        ),
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {
                "replacements": {
                    "type": "array",
                    "description": (
                        "non-empty list of replacements; each item has workspace-relative path, "
                        "old_text, and new_text"
                    ),
                },
            },
            "required": ["replacements"],
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
    ),
}


def validate_tool_registry(registry: dict[str, ToolDefinition] | None = None) -> None:
    """自检 Tool Registry，启动时显式暴露 schema 配置错误。"""
    registry = TOOL_REGISTRY if registry is None else registry
    for name, definition in registry.items():
        if definition.risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError(f"{name} risk_level is invalid: {definition.risk_level}")
        _validate_parameters_schema(name, definition.parameters)


def _validate_parameters_schema(tool_name: str, schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError(f"{tool_name} parameters must be an object schema")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError(f"{tool_name} required must be a list of strings")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"{tool_name} properties must be a dict")
    additional_properties = schema.get("additionalProperties")
    if additional_properties is not None and not isinstance(additional_properties, bool):
        raise ValueError(f"{tool_name} additionalProperties must be a bool")
    for property_name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            raise ValueError(f"{tool_name}.{property_name} schema must be a dict")
        schema_type = property_schema.get("type")
        if schema_type is not None and schema_type not in ALLOWED_JSON_SCHEMA_TYPES:
            raise ValueError(
                f"{tool_name}.{property_name} has unsupported schema type: {schema_type}",
            )


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
