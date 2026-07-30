"""
src/haagent/context/compression/messages.py - 历史工具消息压缩

按类型构建历史 tool message 的模型视图，不按消息年龄改写已有结果。

核心设计:
- build_compressed_model_view: 纯函数，构建压缩视图副本，不修改原始消息列表。
  同一 context epoch 内 state.messages 保持 append-only，模型每轮接收确定性视图副本。
  artifact capsule 原样透传；非 artifact 长文本从首次可见起使用同一折叠表示。
"""

from __future__ import annotations

import json
from typing import Any

from haagent.context.compression.budget import CompressionBudget, estimate_text_tokens
from haagent.context.compression.diagnostics import CompressionDiagnostic


class HistoricalToolCompressionPolicy:
    def __init__(self, budget: CompressionBudget) -> None:
        self.budget = budget


def build_compressed_model_view(
    messages: list[dict[str, Any]],
    budget: CompressionBudget,
) -> tuple[list[dict[str, Any]], list[CompressionDiagnostic]]:
    """构建压缩视图，不修改原始消息。

    返回 (view_messages, diagnostics)。view_messages 是新列表，
    只有需要压缩的消息使用浅拷贝 + 替换 content，未压缩消息保持原引用。

    设计要点:
    - 纯函数，无副作用：相同输入始终产生相同输出
    - 原始 messages 列表不被修改，保留完整历史供 transcript 回放和调试
    - artifact-backed 消息首次进入模型后保持不可变，保证同一 epoch 内前缀稳定
    - 非 artifact 历史长文本使用与消息年龄无关的确定性折叠
    """
    view = list(messages)
    diagnostics: list[CompressionDiagnostic] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        payload = _tool_result_view_payload(message)
        # 浅拷贝消息 dict，后续 helper 在副本上修改，原始消息不受影响
        msg_copy: dict[str, Any] | None = None
        diagnostic: CompressionDiagnostic | None = None
        if payload is not None:
            # artifact capsule 的预览和回读指针已在工具执行边界确定，历史轮次不得改写。
            continue
        elif field_diagnostic := _collapse_json_tool_result_fields(msg_copy := dict(message), content, budget):
            diagnostic = field_diagnostic
        elif len(content) > _historical_text_limit(budget):
            collapsed = _collapse_text_head_tail(
                content,
                head_chars=budget.historical_collapse_head_chars,
                tail_chars=budget.historical_collapse_tail_chars,
            )
            msg_copy = dict(message)
            msg_copy["content"] = collapsed
            diagnostic = CompressionDiagnostic(
                stage="historical_tool_message",
                subject=str(message.get("name", "unknown_tool")),
                decision="collapsed",
                reason="long_text_result",
                original_chars=len(content),
                final_chars=len(collapsed),
                original_tokens=estimate_text_tokens(content),
                final_tokens=estimate_text_tokens(collapsed),
            )
        if diagnostic is None:
            continue
        if msg_copy is not None:
            view[index] = msg_copy
        diagnostics.append(diagnostic)
    return view, diagnostics


def _collapse_json_tool_result_fields(
    message: dict[str, Any],
    content: str,
    budget: CompressionBudget,
) -> CompressionDiagnostic | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    changed = False
    field_limit = budget.historical_collapse_head_chars + budget.historical_collapse_tail_chars + 200
    for key in ("content", "output", "stdout", "stderr"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) <= field_limit:
            continue
        collapsed = _collapse_text_head_tail(
            value,
            head_chars=budget.historical_collapse_head_chars,
            tail_chars=budget.historical_collapse_tail_chars,
        )
        payload[key] = collapsed
        payload["truncated"] = True
        changed = True
    if not changed:
        return None
    message["content"] = json.dumps(payload, ensure_ascii=False)
    return CompressionDiagnostic(
        stage="historical_tool_message",
        subject=str(message.get("name", "unknown_tool")),
        decision="collapsed",
        reason="long_text_result",
        original_chars=len(content),
        final_chars=len(message["content"]),
        original_tokens=estimate_text_tokens(content),
        final_tokens=estimate_text_tokens(message["content"]),
    )


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
    if payload.get("kind") == "tool_result_view" and isinstance(payload.get("artifact"), dict):
        return payload
    if "artifact_path" in payload:
        path = str(payload.get("artifact_path") or "")
        if not path:
            return None
        return {
            "kind": "tool_result_view",
            "tool_name": str(message.get("name", "unknown_tool")),
            "status": "success",
            "content": str(payload.get("output") or payload.get("content") or ""),
            "content_format": "text",
            "artifact": {
                "path": path,
                "original_chars": _int_value(payload.get("original_chars")),
                "preview_chars": _int_value(payload.get("preview_chars")),
            },
            "truncated": bool(payload.get("truncated", True)),
            "continuation_hint": payload.get("continuation_hint"),
        }
    return None


def _historical_text_limit(budget: CompressionBudget) -> int:
    return budget.tool_output_inline_chars


def _collapse_text_head_tail(text: str, *, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    return f"{text[:head_chars].rstrip()}\n...[collapsed {omitted} chars]...\n{text[-tail_chars:].lstrip()}"


def _int_value(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0
