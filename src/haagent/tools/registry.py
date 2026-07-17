"""
haagent/tools/registry.py - 工具注册表组合入口

提供 ToolDefinition 与运行时 registry；静态定义由 ToolCatalog 唯一生成。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
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
    # 动态 MCP 可沿用默认；静态 contribution 必须显式声明。
    replay_safety: ReplaySafety = ReplaySafety.NEVER_REPLAY

    def to_model_schema(self) -> dict[str, Any]:
        """导出模型网关需要的稳定字段，不暴露运行时内部风险与 effect 元数据。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ToolRuntimeRegistry:
    static_tools: Mapping[str, ToolDefinition]
    dynamic_tools: Mapping[str, ToolDefinition]
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


def default_tool_runtime_registry(
    dynamic_tools: Mapping[str, ToolDefinition] | None = None,
) -> ToolRuntimeRegistry:
    return ToolRuntimeRegistry(
        static_tools=TOOL_REGISTRY,
        dynamic_tools=dict(dynamic_tools or {}),
    )


def validate_tool_registry(registry: Mapping[str, ToolDefinition] | None = None) -> None:
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


def _static_tool_registry() -> Mapping[str, ToolDefinition]:
    # 延迟加载避免 registry <-> catalog 循环导入。
    from haagent.tools.catalog import default_tool_catalog

    return default_tool_catalog().definitions


class _ToolRegistryProxy(Mapping[str, ToolDefinition]):
    """只读静态 registry 代理：完整 Mapping 接口，首次访问时从 catalog 加载。

    不用 dict 子类：未覆盖的 copy/update/| 等继承路径会在未加载时看到空表，
    或允许就地修改破坏全局 registry。
    """

    __slots__ = ("_loaded",)

    def __init__(self) -> None:
        self._loaded: Mapping[str, ToolDefinition] | None = None

    def _ensure(self) -> Mapping[str, ToolDefinition]:
        if self._loaded is None:
            # 失败边界：catalog 加载失败时保持未缓存，便于下次重试。
            self._loaded = _static_tool_registry()
        return self._loaded

    def __getitem__(self, key: str) -> ToolDefinition:
        return self._ensure()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())

    def __repr__(self) -> str:
        return repr(dict(self._ensure()))


TOOL_REGISTRY: Mapping[str, ToolDefinition] = _ToolRegistryProxy()
