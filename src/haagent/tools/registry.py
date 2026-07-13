"""
haagent/tools/registry.py - 工具注册表组合入口

合并领域工具定义，校验 schema，并导出模型网关使用的稳定接口。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from haagent.runtime.execution.retry import ReplaySafety


ALLOWED_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}
ExecutionEffect = Literal[
    "read_only",
    "workspace_write",
    "external_effect",
    "interaction",
]
ALLOWED_EXECUTION_EFFECTS = {
    "read_only",
    "workspace_write",
    "external_effect",
    "interaction",
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: str
    parameters: dict[str, Any]
    execution_effect: ExecutionEffect
    replay_safety: ReplaySafety = ReplaySafety.NEVER_REPLAY

    def to_model_schema(self) -> dict[str, Any]:
        """导出模型网关需要的稳定字段，不暴露运行时内部风险与 effect 元数据。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def merge_tool_registry_fragments(
    *fragments: dict[str, ToolDefinition],
) -> dict[str, ToolDefinition]:
    """合并领域注册表片段，重复工具名必须显式失败。"""
    merged: dict[str, ToolDefinition] = {}
    for fragment in fragments:
        duplicate_names = sorted(set(merged).intersection(fragment))
        if duplicate_names:
            raise ValueError(f"duplicate tool definition: {duplicate_names[0]}")
        merged.update(fragment)
    return merged


@dataclass(frozen=True)
class ToolRuntimeRegistry:
    static_tools: dict[str, ToolDefinition]
    dynamic_tools: dict[str, ToolDefinition]
    _schema_version: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # registry 在构造后按只读 snapshot 使用；version 只序列化一次。
        object.__setattr__(self, "_schema_version", self._compute_schema_version())

    def get(self, name: str) -> ToolDefinition:
        if name in self.dynamic_tools:
            return self.dynamic_tools[name]
        try:
            return self.static_tools[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def has(self, name: str) -> bool:
        return name in self.dynamic_tools or name in self.static_tools

    def allowed_definitions(self, names: list[str]) -> list[ToolDefinition]:
        return [self.get(name) for name in names]

    @property
    def schema_version(self) -> str:
        """静态/动态工具 name/description/parameters 的 canonical JSON hash。"""

        return self._schema_version

    def _compute_schema_version(self) -> str:
        items: list[dict[str, Any]] = []
        for name in sorted(set(self.static_tools) | set(self.dynamic_tools)):
            definition = self.get(name)
            items.append(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                }
            )
        raw = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


from haagent.tools.registry_fragments.agent import AGENT_TOOL_REGISTRY
from haagent.tools.registry_fragments.core import CORE_TOOL_REGISTRY
from haagent.tools.registry_fragments.files import FILE_TOOL_REGISTRY
from haagent.tools.registry_fragments.mcp import MCP_TOOL_REGISTRY
from haagent.tools.registry_fragments.shell import SHELL_TOOL_REGISTRY
from haagent.tools.registry_fragments.skills import SKILL_TOOL_REGISTRY
from haagent.tools.registry_fragments.web import WEB_TOOL_REGISTRY


TOOL_REGISTRY = merge_tool_registry_fragments(
    CORE_TOOL_REGISTRY,
    AGENT_TOOL_REGISTRY,
    FILE_TOOL_REGISTRY,
    SKILL_TOOL_REGISTRY,
    WEB_TOOL_REGISTRY,
    MCP_TOOL_REGISTRY,
    SHELL_TOOL_REGISTRY,
)


def default_tool_runtime_registry(
    dynamic_tools: dict[str, ToolDefinition] | None = None,
) -> ToolRuntimeRegistry:
    return ToolRuntimeRegistry(
        static_tools=TOOL_REGISTRY,
        dynamic_tools=dict(dynamic_tools or {}),
    )


def validate_tool_registry(registry: dict[str, ToolDefinition] | None = None) -> None:
    """自检 Tool Registry，启动时显式暴露 schema 配置错误。"""
    registry = TOOL_REGISTRY if registry is None else registry
    for name, definition in registry.items():
        if definition.risk_level not in ALLOWED_RISK_LEVELS:
            raise ValueError(f"{name} risk_level is invalid: {definition.risk_level}")
        if definition.execution_effect not in ALLOWED_EXECUTION_EFFECTS:
            raise ValueError(
                f"{name} execution_effect is invalid: {definition.execution_effect}",
            )
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


def allowed_tool_definitions(
    names: list[str],
    registry: ToolRuntimeRegistry | None = None,
) -> list[ToolDefinition]:
    runtime_registry = registry or default_tool_runtime_registry()
    return runtime_registry.allowed_definitions(names)


def export_tool_schemas(
    names: list[str],
    registry: ToolRuntimeRegistry | None = None,
    *,
    cache: Any | None = None,
    diagnostics_sink: Any | None = None,
) -> list[dict[str, Any]]:
    runtime_registry = registry or default_tool_runtime_registry()
    if cache is None:
        from haagent.tools.schema_cache import default_tool_schema_cache

        cache = default_tool_schema_cache()
    return cache.export(names, runtime_registry, diagnostics_sink=diagnostics_sink)
