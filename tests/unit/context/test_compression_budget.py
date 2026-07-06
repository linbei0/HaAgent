"""
tests/unit/context/test_compression_budget.py - 统一压缩预算测试

验证压缩预算从模型上下文窗口派生，而不是使用固定字符预算。
"""

from haagent.context.compression.budget import derive_compression_budget


class Metadata:
    def __init__(self, context_window_tokens: int) -> None:
        self.context_window_tokens = context_window_tokens


def test_context_builder_budget_scales_with_context_window() -> None:
    small = derive_compression_budget(Metadata(32_000))
    medium = derive_compression_budget(Metadata(128_000))
    large = derive_compression_budget(Metadata(256_000))

    assert small.context_builder_max_tokens == 8_000
    assert 20_000 <= medium.context_builder_max_tokens <= 25_000
    assert 40_000 <= large.context_builder_max_tokens <= 50_000


def test_context_builder_budget_has_upper_bound() -> None:
    budget = derive_compression_budget(Metadata(1_000_000))

    assert budget.context_builder_max_tokens == 50_000


def test_budget_uses_fallback_when_metadata_has_no_context_window() -> None:
    budget = derive_compression_budget(object(), fallback_context_window=64_000)

    assert budget.context_window_tokens == 64_000
