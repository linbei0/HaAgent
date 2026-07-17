"""
haagent/tools/presentation.py - 工具展示摘要

委托静态 ToolCatalog 生成参数/结果摘要；未知与动态 MCP 工具走通用回退。
"""

from __future__ import annotations

from haagent.tools.catalog import default_tool_catalog


def summarize_tool_args(tool_name: str, args: dict[str, object]) -> dict[str, object]:
    return default_tool_catalog().summarize_args(tool_name, args)


def summarize_tool_result(tool_name: str, result: dict[str, object]) -> dict[str, object]:
    return default_tool_catalog().summarize_result(tool_name, result)
