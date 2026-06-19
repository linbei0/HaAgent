"""
tests/test_tool_registry.py - Tool Registry v1 测试

验证工具注册表包含本阶段全部工具定义和最小元数据。
"""

from agentfoundry.tools.registry import TOOL_REGISTRY, ToolDefinition


def test_tool_registry_contains_mvp_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "fake_tool",
        "file_search",
        "file_read",
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
