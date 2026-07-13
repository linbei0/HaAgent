"""
tests/unit/tools/test_tool_schema_cache.py - ToolSchemaCache 合同测试

覆盖 schema 导出去重、深拷贝隔离、version 稳定与动态工具变更。
"""

from __future__ import annotations

from haagent.runtime.execution.retry import ReplaySafety
from haagent.tools import registry as registry_module
from haagent.tools.registry import ToolDefinition, ToolRuntimeRegistry
from haagent.tools.schema_cache import ToolSchemaCache


def _def(name: str, description: str = "d") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        risk_level="low",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "enum": ["a", "b"]}},
            "required": ["path"],
        },
        execution_effect="read_only",
            replay_safety=ReplaySafety.SAFE_TO_REPLAY,
    )


def test_schema_cache_exports_once_per_version_and_names() -> None:
    registry = ToolRuntimeRegistry(static_tools={"file_read": _def("file_read")}, dynamic_tools={})
    cache = ToolSchemaCache()
    first = cache.export(["file_read"], registry)
    second = cache.export(["file_read"], registry)
    assert first == second
    first[0]["parameters"]["properties"].clear()
    third = cache.export(["file_read"], registry)
    assert third[0]["parameters"]["properties"]
    assert "path" in third[0]["parameters"]["properties"]


def test_schema_version_stable_under_dict_insertion_order() -> None:
    a = ToolRuntimeRegistry(
        static_tools={"b": _def("b"), "a": _def("a")},
        dynamic_tools={},
    )
    b = ToolRuntimeRegistry(
        static_tools={"a": _def("a"), "b": _def("b")},
        dynamic_tools={},
    )
    assert a.schema_version == b.schema_version


def test_schema_version_changes_when_dynamic_tool_changes() -> None:
    base = ToolRuntimeRegistry(static_tools={"file_read": _def("file_read")}, dynamic_tools={})
    changed = ToolRuntimeRegistry(
        static_tools={"file_read": _def("file_read")},
        dynamic_tools={"mcp_x": _def("mcp_x", description="v2")},
    )
    assert base.schema_version != changed.schema_version


def test_schema_version_is_computed_once_per_registry(monkeypatch) -> None:
    registry = ToolRuntimeRegistry(static_tools={"file_read": _def("file_read")}, dynamic_tools={})
    first = registry.schema_version

    def unexpected_serialize(*args, **kwargs):
        raise AssertionError("schema version was recomputed")

    monkeypatch.setattr(registry_module.json, "dumps", unexpected_serialize)

    assert registry.schema_version == first


def test_export_preserves_property_and_enum_order() -> None:
    registry = ToolRuntimeRegistry(static_tools={"t": _def("t")}, dynamic_tools={})
    cache = ToolSchemaCache()
    exported = cache.export(["t"], registry)
    props = list(exported[0]["parameters"]["properties"].keys())
    assert props == ["path"]
    assert exported[0]["parameters"]["properties"]["path"]["enum"] == ["a", "b"]
    assert exported[0]["parameters"]["required"] == ["path"]
