"""
src/haagent/context/compression/messages.py - 历史工具消息压缩

按类型构建历史 tool message 的模型视图，不按消息年龄改写已有结果。

核心设计:
- build_compressed_model_view: 只识别已经完成投影的 ToolResultView，不再次折叠工具正文。
- checkpoint 负责历史回收；本函数保持消息链和已投影结果不变。
"""

from __future__ import annotations

import json
from typing import Any

from haagent.context.compression.budget import CompressionBudget
from haagent.context.compression.diagnostics import CompressionDiagnostic


def build_compressed_model_view(
    messages: list[dict[str, Any]],
    budget: CompressionBudget,
) -> tuple[list[dict[str, Any]], list[CompressionDiagnostic]]:
    """构建压缩视图，不修改原始消息。

    返回 (view_messages, diagnostics)。当前版本不在这里改写工具消息。

    设计要点:
    - 纯函数，无副作用：相同输入始终产生相同输出
    - 原始 messages 列表不被修改，保留完整历史供 transcript 回放和调试
    - artifact-backed 与 inline ToolResultView 都已经完成模型投影
    - 预算超限时由 checkpoint 只回收更早的连续历史
    """
    del budget
    view = list(messages)
    diagnostics: list[CompressionDiagnostic] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if _tool_result_view_payload(message) is not None:
            # artifact capsule 和 inline ToolResultView 都已在工具边界投影，历史轮次不得改写。
            continue
        # 正常路径的 raw tool message 应在 Router 边界完成投影；这里保持原样以避免第二层有损截断。
    return view, diagnostics


def _tool_result_view_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") == "tool_result_view":
        return payload
    return None
