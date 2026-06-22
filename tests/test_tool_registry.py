"""
tests/test_tool_registry.py - Tool Registry v1 测试

验证工具注册表包含本阶段全部工具定义、风险级别和可导出的 JSON Schema。
"""

import pytest

from haagent.tools.registry import (
    TOOL_REGISTRY,
    ToolDefinition,
    allowed_tool_definitions,
    export_tool_schemas,
    get_tool_definition,
    validate_tool_registry,
)


def test_tool_registry_contains_mvp_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "fake_tool",
        "file_list",
        "file_search",
        "file_read",
        "request_user_input",
        "file_write",
        "code_run",
        "apply_patch",
        "shell",
    }
    assert all(isinstance(definition, ToolDefinition) for definition in TOOL_REGISTRY.values())


def test_tool_registry_definitions_have_required_metadata() -> None:
    fake_tool = TOOL_REGISTRY["fake_tool"]

    assert fake_tool.name == "fake_tool"
    assert fake_tool.description == "deterministic test tool"
    assert fake_tool.risk_level == "low"
    assert isinstance(fake_tool.parameters, dict)
    assert fake_tool.parameters["type"] == "object"
    assert fake_tool.parameters["properties"] == {}
    assert fake_tool.parameters["required"] == []
    assert fake_tool.parameters["additionalProperties"] is True


def test_export_fake_tool_schema() -> None:
    schemas = export_tool_schemas(["fake_tool"])

    assert schemas == [
        {
            "name": "fake_tool",
            "description": "deterministic test tool",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            },
        },
    ]


def test_export_tool_schemas_only_exports_allowed_tools() -> None:
    schemas = export_tool_schemas(["file_list", "file_read", "shell"])

    assert [schema["name"] for schema in schemas] == ["file_list", "file_read", "shell"]
    assert [definition.name for definition in allowed_tool_definitions(["file_read"])] == ["file_read"]


def test_export_file_list_schema_describes_discovery_defaults() -> None:
    schemas = export_tool_schemas(["file_list"])
    schema = schemas[0]

    assert schema["description"] == "list a compact workspace file tree for project discovery"
    assert schema["parameters"]["required"] == []
    assert set(schema["parameters"]["properties"]) == {"path", "max_depth", "max_entries"}


def test_export_shell_schema_describes_cwd_relative_to_workspace_root() -> None:
    schemas = export_tool_schemas(["shell"])
    cwd_description = schemas[0]["parameters"]["properties"]["cwd"]["description"]

    assert "workspace_root" in cwd_description
    assert "." in cwd_description
    assert "omit" in cwd_description


def test_file_read_schema_supports_keyword() -> None:
    schemas = export_tool_schemas(["file_read"])
    schema = schemas[0]

    assert "keyword" in schema["parameters"]["properties"]
    assert schema["parameters"]["properties"]["keyword"]["type"] == "string"


def test_request_user_input_schema_requires_question() -> None:
    schemas = export_tool_schemas(["request_user_input"])
    schema = schemas[0]

    assert schema["description"] == "ask the user for missing information before continuing the task"
    assert schema["parameters"]["required"] == ["question"]
    assert schema["parameters"]["properties"]["question"]["type"] == "string"
    assert schema["parameters"]["properties"]["reason"]["type"] == "string"


def test_file_write_schema_describes_modes() -> None:
    schemas = export_tool_schemas(["file_write"])
    mode_schema = schemas[0]["parameters"]["properties"]["mode"]

    assert mode_schema["enum"] == ["create", "overwrite", "append"]
    assert schemas[0]["parameters"]["required"] == ["path", "content", "mode"]


def test_code_run_schema_describes_timeout_and_cwd() -> None:
    schemas = export_tool_schemas(["code_run"])
    properties = schemas[0]["parameters"]["properties"]

    assert "timeout_seconds" in properties
    assert "cwd" in properties
    assert "workspace_root" in properties["cwd"]["description"]


def test_tool_registry_rejects_unknown_tool() -> None:
    with pytest.raises(KeyError, match="unknown tool: mystery_tool"):
        get_tool_definition("mystery_tool")

    with pytest.raises(KeyError, match="unknown tool: mystery_tool"):
        export_tool_schemas(["fake_tool", "mystery_tool"])


def test_mutating_tools_are_high_risk() -> None:
    assert TOOL_REGISTRY["apply_patch"].risk_level == "high"
    assert TOOL_REGISTRY["shell"].risk_level == "high"
    assert TOOL_REGISTRY["file_write"].risk_level == "high"
    assert TOOL_REGISTRY["code_run"].risk_level == "high"


def test_current_tool_registry_self_check_passes() -> None:
    validate_tool_registry()


def test_tool_registry_self_check_rejects_unknown_schema_type() -> None:
    registry = {
        "bad": ToolDefinition(
            name="bad",
            description="bad",
            risk_level="low",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "mystery"}},
                "required": [],
            },
        ),
    }

    with pytest.raises(ValueError, match="bad.value has unsupported schema type: mystery"):
        validate_tool_registry(registry)


def test_tool_registry_self_check_rejects_required_not_list() -> None:
    registry = {
        "bad": ToolDefinition(
            name="bad",
            description="bad",
            risk_level="low",
            parameters={"type": "object", "properties": {}, "required": "value"},
        ),
    }

    with pytest.raises(ValueError, match="bad required must be a list of strings"):
        validate_tool_registry(registry)


def test_tool_registry_self_check_rejects_properties_not_dict() -> None:
    registry = {
        "bad": ToolDefinition(
            name="bad",
            description="bad",
            risk_level="low",
            parameters={"type": "object", "properties": [], "required": []},
        ),
    }

    with pytest.raises(ValueError, match="bad properties must be a dict"):
        validate_tool_registry(registry)


def test_tool_registry_self_check_rejects_invalid_risk_level() -> None:
    registry = {
        "bad": ToolDefinition(
            name="bad",
            description="bad",
            risk_level="extreme",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
    }

    with pytest.raises(ValueError, match="bad risk_level is invalid: extreme"):
        validate_tool_registry(registry)
