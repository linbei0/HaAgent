"""
haagent/context/observations.py - 工具 observation 摘要兼容入口

保留旧导入路径，实际压缩规则位于 observation_compaction。
"""

from __future__ import annotations

from haagent.context.observation_compaction import (
    observation_summary,
    observation_tool_name,
    raw_observation_summary,
)

__all__ = [
    "observation_summary",
    "observation_tool_name",
    "raw_observation_summary",
]
