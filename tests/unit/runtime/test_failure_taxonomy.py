"""
tests/unit/runtime/test_failure_taxonomy.py - Failure Taxonomy 测试

验证当前 run failure category 集中定义且覆盖 MVP 分类。
"""

from haagent.runtime.orchestration.failure import FailureCategory


def test_failure_taxonomy_contains_current_categories() -> None:
    assert {category.value for category in FailureCategory} == {
        "Task Spec Failure",
        "Context Failure",
        "Model Failure",
        "Model Call Failure",
        "Tool Interface Failure",
        "Tool Argument Failure",
        "User Denied Failure",
        "Guardrail Failure",
        "Verification Failure",
        "Loop Limit Failure",
        "Runtime Failure",
    }
