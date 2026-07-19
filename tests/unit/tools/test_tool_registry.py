"""
tests/unit/tools/test_tool_registry.py - Tool Registry v1 测试

验证工具注册表包含本阶段全部工具定义、风险级别和可导出的 JSON Schema。
"""

from collections.abc import Mapping

import pytest

from haagent.tools.registry import (
    ALLOWED_EXECUTION_EFFECTS,
    TOOL_REGISTRY,
    ToolDefinition,
    allowed_tool_definitions,
    default_tool_runtime_registry,
    export_tool_schemas,
    validate_tool_registry,
)


def test_tool_registry_contains_mvp_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "fake_tool",
        "load_image_attachment",
        "file_list",
        "grep",
        "file_read",
        "request_user_input",
        "start_memory_update",
        "skill_list",
        "skill_read",
        "skill_market_search",
        "web_search",
        "web_fetch",
        "list_mcp_resources",
        "read_mcp_resource",
        "file_write",
        "code_run",
        "apply_patch",
        "apply_patch_set",
        "shell",
        "agent",
        "send_message",
        "task_stop",
        "task_get",
        "task_list",
        "task_output",
    }
    assert all(isinstance(definition, ToolDefinition) for definition in TOOL_REGISTRY.values())


def test_tool_registry_static_execution_effects() -> None:
    expected = {
        "fake_tool": "read_only",
        "load_image_attachment": "read_only",
        "request_user_input": "interaction",
        "start_memory_update": "external_effect",
        "agent": "external_effect",
        "send_message": "external_effect",
        "task_stop": "external_effect",
        "task_get": "read_only",
        "task_list": "read_only",
        "task_output": "read_only",
        "file_list": "read_only",
        "grep": "read_only",
        "file_read": "read_only",
        "file_write": "workspace_write",
        "apply_patch": "workspace_write",
        "apply_patch_set": "workspace_write",
        "skill_list": "read_only",
        "skill_read": "read_only",
        "skill_market_search": "read_only",
        "web_search": "read_only",
        "web_fetch": "read_only",
        "list_mcp_resources": "read_only",
        "read_mcp_resource": "read_only",
        "code_run": "external_effect",
        "shell": "external_effect",
    }
    assert {name: TOOL_REGISTRY[name].execution_effect for name in expected} == expected
    assert ALLOWED_EXECUTION_EFFECTS == {
        "read_only",
        "workspace_write",
        "external_effect",
        "interaction",
    }


def test_execution_effect_is_not_exported_to_model_schema() -> None:
    schema = TOOL_REGISTRY["file_read"].to_model_schema()
    assert set(schema) == {"name", "description", "parameters"}
    assert "execution_effect" not in schema
    assert "risk_level" not in schema


def test_tool_definition_requires_execution_effect() -> None:
    with pytest.raises(TypeError):
        ToolDefinition(  # type: ignore[call-arg]
            name="missing_effect",
            description="missing",
            risk_level="low",
            parameters={"type": "object", "properties": {}, "required": []},
        )


def test_tool_registry_is_read_only_mapping_not_dict_subclass() -> None:
    # 不全量实现 dict 子类会让 copy/update 等继承路径在未加载时返回空集；只读 Mapping 更安全。
    assert isinstance(TOOL_REGISTRY, Mapping)
    assert not isinstance(TOOL_REGISTRY, dict)
    assert "file_read" in TOOL_REGISTRY
    assert len(TOOL_REGISTRY) == len(dict(TOOL_REGISTRY))
    with pytest.raises(TypeError):
        TOOL_REGISTRY["file_read"] = TOOL_REGISTRY["file_read"]  # type: ignore[index]


def test_tool_registry_is_built_from_static_tool_catalog() -> None:
    from haagent.tools.catalog import default_tool_catalog

    assert dict(TOOL_REGISTRY) == default_tool_catalog().definitions



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

    assert schema["description"] == "list a compact file tree by absolute or workspace-relative path"
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


def test_grep_schema_stays_deterministic() -> None:
    schemas = export_tool_schemas(["grep"])
    schema = schemas[0]

    assert "regular expression" in schema["description"]
    assert schema["parameters"]["required"] == ["pattern"]
    assert set(schema["parameters"]["properties"]) == {
        "pattern",
        "root",
        "file_glob",
        "case_sensitive",
        "max_matches",
        "timeout_seconds",
    }
    assert "directory or file" in schema["parameters"]["properties"]["root"]["description"]
    assert "defaults to **/*" not in schema["parameters"]["properties"]["file_glob"]["description"]
    assert TOOL_REGISTRY["grep"].risk_level == "low"


def test_task_tools_are_low_risk_worker_inspection_tools() -> None:
    get_schema, list_schema, output_schema = export_tool_schemas(["task_get", "task_list", "task_output"])

    assert get_schema["parameters"]["required"] == ["task_id"]
    assert set(get_schema["parameters"]["properties"]) == {"task_id"}
    assert list_schema["parameters"]["required"] == []
    assert set(list_schema["parameters"]["properties"]) == {"status"}
    assert output_schema["parameters"]["required"] == ["task_id"]
    assert set(output_schema["parameters"]["properties"]) == {"task_id", "max_chars"}
    assert TOOL_REGISTRY["task_get"].risk_level == "low"
    assert TOOL_REGISTRY["task_list"].risk_level == "low"
    assert TOOL_REGISTRY["task_output"].risk_level == "low"


def test_request_user_input_schema_requires_question() -> None:
    schemas = export_tool_schemas(["request_user_input"])
    schema = schemas[0]

    assert schema["description"] == "ask the user for missing information before continuing the task"
    assert schema["parameters"]["required"] == ["question"]
    assert schema["parameters"]["properties"]["question"]["type"] == "string"
    assert schema["parameters"]["properties"]["reason"]["type"] == "string"


def test_start_memory_update_schema_is_low_risk_internal_signal() -> None:
    schemas = export_tool_schemas(["start_memory_update"])
    schema = schemas[0]

    assert schema["name"] == "start_memory_update"
    assert "不直接写正式记忆" in schema["description"]
    assert schema["parameters"]["required"] == []
    assert schema["parameters"]["properties"]["reason"]["type"] == "string"
    assert TOOL_REGISTRY["start_memory_update"].risk_level == "low"


def test_skill_tool_schemas_are_low_risk_and_do_not_require_body_in_list() -> None:
    list_schema, read_schema = export_tool_schemas(["skill_list", "skill_read"])

    assert list_schema["parameters"]["required"] == []
    assert set(list_schema["parameters"]["properties"]) == {"query", "source", "max_results"}
    assert read_schema["parameters"]["required"] == ["name"]
    assert set(read_schema["parameters"]["properties"]) == {"name"}
    assert TOOL_REGISTRY["skill_list"].risk_level == "low"
    assert TOOL_REGISTRY["skill_read"].risk_level == "low"


def test_skill_market_search_schema_is_read_only_marketplace_search() -> None:
    schema = export_tool_schemas(["skill_market_search"])[0]

    assert schema["name"] == "skill_market_search"
    assert "marketplace" in schema["description"]
    assert schema["parameters"]["required"] == ["query"]
    assert set(schema["parameters"]["properties"]) == {"query", "providers", "limit"}
    assert schema["parameters"]["properties"]["providers"]["items"]["enum"] == ["skills_sh", "skillsmp"]
    assert TOOL_REGISTRY["skill_market_search"].risk_level == "low"


def test_web_tool_schemas_describe_explicit_read_only_network_access() -> None:
    schemas = export_tool_schemas(["web_search", "web_fetch"])
    search_schema = schemas[0]
    fetch_schema = schemas[1]

    assert search_schema["name"] == "web_search"
    assert search_schema["parameters"]["required"] == ["query"]
    assert set(search_schema["parameters"]["properties"]) == {
        "query",
        "max_results",
        "provider",
        "topic",
        "freshness",
    }
    assert search_schema["parameters"]["properties"]["provider"]["enum"] == ["tavily", "brave"]
    assert TOOL_REGISTRY["web_search"].risk_level == "low"

    assert fetch_schema["name"] == "web_fetch"
    assert fetch_schema["parameters"]["required"] == ["url"]
    assert set(fetch_schema["parameters"]["properties"]) == {"url", "max_chars"}
    assert TOOL_REGISTRY["web_fetch"].risk_level == "medium"


def test_mcp_resource_tools_are_static_read_tools() -> None:
    list_schema, read_schema = export_tool_schemas(["list_mcp_resources", "read_mcp_resource"])

    assert list_schema["parameters"]["required"] == []
    assert list_schema["parameters"]["additionalProperties"] is False
    assert TOOL_REGISTRY["list_mcp_resources"].risk_level == "low"

    assert read_schema["parameters"]["required"] == ["server", "uri"]
    assert set(read_schema["parameters"]["properties"]) == {"server", "uri"}
    assert TOOL_REGISTRY["read_mcp_resource"].risk_level == "medium"


def test_runtime_registry_exports_dynamic_mcp_tool_schema() -> None:
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
    )
    registry = default_tool_runtime_registry({"mcp__fixture__echo": dynamic})

    schemas = export_tool_schemas(["mcp__fixture__echo"], registry=registry)

    assert schemas == [dynamic.to_model_schema()]
    assert registry.has("mcp__fixture__echo")


def test_runtime_registry_does_not_mutate_global_tool_registry() -> None:
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={"type": "object", "properties": {}, "required": []},
        execution_effect="external_effect",
    )

    default_tool_runtime_registry({"mcp__fixture__echo": dynamic})

    assert "mcp__fixture__echo" not in TOOL_REGISTRY


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


def test_apply_patch_set_schema_describes_atomic_replacements() -> None:
    schemas = export_tool_schemas(["apply_patch_set"])
    schema = schemas[0]

    assert "atomically" in schema["description"]
    assert "Prefer this over repeated apply_patch calls" in schema["description"]
    assert schema["parameters"]["required"] == ["replacements"]
    assert schema["parameters"]["properties"]["replacements"]["type"] == "array"
    assert TOOL_REGISTRY["apply_patch_set"].risk_level == "high"


def test_tool_registry_rejects_unknown_tool() -> None:
    with pytest.raises(KeyError, match="unknown tool: mystery_tool"):
        export_tool_schemas(["fake_tool", "mystery_tool"])


def test_mutating_tools_are_high_risk() -> None:
    assert TOOL_REGISTRY["apply_patch"].risk_level == "high"
    assert TOOL_REGISTRY["apply_patch_set"].risk_level == "high"
    assert TOOL_REGISTRY["shell"].risk_level == "high"
    assert TOOL_REGISTRY["file_write"].risk_level == "high"
    assert TOOL_REGISTRY["code_run"].risk_level == "high"


def test_current_tool_registry_self_check_passes() -> None:
    validate_tool_registry()


def test_tool_registry_self_check_rejects_invalid_execution_effect() -> None:
    registry = {
        "bad": ToolDefinition(
            name="bad",
            description="bad",
            risk_level="low",
            parameters={"type": "object", "properties": {}, "required": []},
            execution_effect="side_effect",  # type: ignore[arg-type]
        ),
    }

    with pytest.raises(ValueError, match="bad execution_effect is invalid: side_effect"):
        validate_tool_registry(registry)


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
            execution_effect="read_only",
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
            execution_effect="read_only",
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
            execution_effect="read_only",
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
            execution_effect="read_only",
        ),
    }

    with pytest.raises(ValueError, match="bad risk_level is invalid: extreme"):
        validate_tool_registry(registry)
