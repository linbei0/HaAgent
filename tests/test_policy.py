"""
tests/test_policy.py - Policy Engine 行为测试

验证工具风险等级会映射为稳定的 allow/deny 决策。
"""

from agentfoundry.runtime.policy import evaluate_tool_call
from agentfoundry.tools.registry import ToolDefinition


def make_tool(risk_level: str) -> ToolDefinition:
    return ToolDefinition(
        name=f"{risk_level}_tool",
        description="test tool",
        risk_level=risk_level,
        parameters={"type": "object", "properties": {}, "required": []},
    )


def test_policy_allows_low_and_medium_risk_tools() -> None:
    low_decision = evaluate_tool_call(make_tool("low"))
    medium_decision = evaluate_tool_call(make_tool("medium"))

    assert low_decision.action == "allow"
    assert low_decision.reason == "policy allows low risk tool low_tool"
    assert low_decision.approval.required is False
    assert low_decision.approval.status == "not_required"
    assert medium_decision.action == "allow"
    assert medium_decision.reason == "policy allows medium risk tool medium_tool"
    assert medium_decision.approval.required is False
    assert medium_decision.approval.status == "not_required"


def test_policy_denies_high_risk_tools() -> None:
    decision = evaluate_tool_call(make_tool("high"))

    assert decision.action == "deny"
    assert decision.reason == "policy denies high risk tool high_tool"
    assert decision.approval.required is True
    assert decision.approval.status == "missing"
    assert decision.approval.reason == "approval not allowed for high risk tool high_tool"


def test_policy_records_approval_allowed_but_missing_for_high_risk_tools() -> None:
    decision = evaluate_tool_call(make_tool("high"), approval_allowed_tools=["high_tool"])

    assert decision.action == "deny"
    assert decision.approval.required is True
    assert decision.approval.status == "missing"
    assert decision.approval.reason == "approval allowed but missing for high risk tool high_tool"
