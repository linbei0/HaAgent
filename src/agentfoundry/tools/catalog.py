"""
agentfoundry/tools/catalog.py - 工具目录兼容层

保留旧 TOOL_CATALOG 导入路径，并从 Tool Registry 派生一句话用途。
"""

from __future__ import annotations

from agentfoundry.tools.registry import TOOL_REGISTRY


TOOL_CATALOG = {
    name: definition.description
    for name, definition in TOOL_REGISTRY.items()
}
