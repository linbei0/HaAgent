"""
src/haagent/context/compression/__init__.py - 统一压缩入口

集中导出 HaAgent 上下文压缩流水线使用的预算和诊断核心类型。
"""

from haagent.context.compression.budget import (
    CompressionBudget,
    CompressionPolicy,
    derive_compression_budget,
    estimate_message_tokens,
    estimate_text_tokens,
)
from haagent.context.compression.diagnostics import CompressionDiagnostic
from haagent.context.compression.sections import ContextBudget, context_budget_from_compression_budget

__all__ = [
    "CompressionBudget",
    "CompressionDiagnostic",
    "CompressionPolicy",
    "ContextBudget",
    "context_budget_from_compression_budget",
    "derive_compression_budget",
    "estimate_message_tokens",
    "estimate_text_tokens",
]
