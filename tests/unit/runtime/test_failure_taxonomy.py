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


def test_tool_not_allowed_is_recoverable_observation() -> None:
    """模型误用未授权工具名时，应回 observation 让模型自纠，而不是整轮终态失败。"""
    assert (
        _tool_error_is_terminal(
            {
                "status": "error",
                "error": {"type": "tool_not_allowed", "message": "tool is not allowed: read_file"},
            }
        )
        is False
    )


def test_unknown_tool_is_recoverable_observation() -> None:
    assert (
        _tool_error_is_terminal(
            {
                "status": "error",
                "error": {"type": "unknown_tool", "message": "unknown tool: foo"},
            }
        )
        is False
    )


def test_approval_denied_remains_terminal() -> None:
    assert (
        _tool_error_is_terminal(
            {
                "status": "error",
                "error": {"type": "approval_denied", "message": "user denied"},
            }
        )
        is True
    )


def test_policy_denied_remains_terminal() -> None:
    assert (
        _tool_error_is_terminal(
            {
                "status": "error",
                "error": {"type": "policy_denied", "message": "policy denied"},
            }
        )
        is True
    )
