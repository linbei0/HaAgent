"""
src/haagent/context/compression/budget.py - 压缩预算派生

根据模型上下文窗口派生统一压缩预算，并提供轻量 token 估算。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompressionBudget:
    context_window_tokens: int
    reserved_output_tokens: int
    safety_buffer_tokens: int
    available_input_tokens: int
    context_builder_max_tokens: int
    tool_output_inline_chars: int = 12_000
    tool_output_preview_chars: int = 3_000
    artifact_recent_preview_count: int = 3
    historical_collapse_head_chars: int = 900
    historical_collapse_tail_chars: int = 500
    full_compact_preserve_recent: int = 6


@dataclass(frozen=True)
class CompressionPolicy:
    fallback_context_window: int = 200_000
    reserved_output_token_cap: int = 20_000
    safety_buffer_token_cap: int = 13_000
    context_builder_budget_min_tokens: int = 8_000
    context_builder_budget_max_tokens: int = 50_000


def derive_compression_budget(
    model_metadata: object | None,
    *,
    fallback_context_window: int = 200_000,
) -> CompressionBudget:
    context_window = _metadata_context_window(model_metadata) or fallback_context_window
    context_window = max(1, int(context_window))
    reserved_output = min(20_000, int(context_window * 0.10))
    safety_buffer = min(13_000, int(context_window * 0.08))
    available_input = max(0, context_window - reserved_output - safety_buffer)
    context_builder = _clamp(int(available_input * 0.20), 8_000, 50_000)
    return CompressionBudget(
        context_window_tokens=context_window,
        reserved_output_tokens=reserved_output,
        safety_buffer_tokens=safety_buffer,
        available_input_tokens=available_input,
        context_builder_max_tokens=context_builder,
    )


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    # 统一预算只需要稳定近似值；局部预览仍使用字符限制。
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        try:
            serialized = json.dumps(message, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            serialized = str(message)
        total += estimate_text_tokens(serialized)
    return total


def _metadata_context_window(model_metadata: object | None) -> int | None:
    if model_metadata is None:
        return None
    for field in ("context_window_tokens", "context_window", "max_context_tokens"):
        value = getattr(model_metadata, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    if isinstance(model_metadata, dict):
        for field in ("context_window_tokens", "context_window", "max_context_tokens"):
            value = model_metadata.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))
