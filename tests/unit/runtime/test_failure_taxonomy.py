"""
tests/unit/runtime/test_failure_taxonomy.py - Failure Taxonomy 测试

验证当前 run failure category 集中定义且覆盖 MVP 分类。
"""

from haagent.runtime.orchestration.failure import FailureCategory
from haagent.runtime.orchestration.orchestrator import _tool_error_is_terminal


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


def test_tool_error_terminality_matches_runtime_boundary() -> None:
    """模型接口误用可自纠；用户拒绝和 policy 拒绝必须终止当前执行。"""
    expected = {
        "tool_not_allowed": False,
        "unknown_tool": False,
        "approval_denied": True,
        "policy_denied": True,
    }

    for error_type, terminal in expected.items():
        result = _tool_error_is_terminal(
            {
                "status": "error",
                "error": {"type": error_type, "message": "test error"},
            }
        )

        assert result is terminal
