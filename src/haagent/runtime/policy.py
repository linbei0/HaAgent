"""
haagent/runtime/policy.py - Policy Engine

为工具调用生成可执行的 policy decision，高风险工具会在路由层被拒绝。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from haagent.tools.registry import ToolDefinition


@dataclass(frozen=True)
class ApprovalDecision:
    required: bool
    status: str
    reason: str


@dataclass(frozen=True)
class PolicyDecision:
    tool_name: str
    risk_level: str
    action: str
    reason: str
    approval: ApprovalDecision

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_tool_call(
    tool_definition: ToolDefinition,
    approval_allowed_tools: list[str] | None = None,
    approved_tools: list[str] | None = None,
) -> PolicyDecision:
    """根据 Tool Registry 风险等级返回工具调用决策。"""
    approval_allowed_tools = approval_allowed_tools or []
    approved_tools = approved_tools or []
    action = (
        "allow"
        if tool_definition.risk_level != "high" or tool_definition.name in approved_tools
        else "deny"
    )
    reason_action = "denies" if action == "deny" else "allows"
    reason = f"policy {reason_action} {tool_definition.risk_level} risk tool {tool_definition.name}"
    if tool_definition.risk_level == "high":
        approval_status = "granted" if tool_definition.name in approved_tools else "missing"
        if approval_status == "granted":
            approval_reason = f"approval granted for high risk tool {tool_definition.name}"
        elif tool_definition.name in approval_allowed_tools:
            approval_reason = f"approval allowed but missing for high risk tool {tool_definition.name}"
        else:
            approval_reason = f"approval not allowed for high risk tool {tool_definition.name}"
        approval = ApprovalDecision(
            required=True,
            status=approval_status,
            reason=approval_reason,
        )
    else:
        approval = ApprovalDecision(
            required=False,
            status="not_required",
            reason=f"approval not required for {tool_definition.risk_level} risk tool {tool_definition.name}",
        )
    return PolicyDecision(
        tool_name=tool_definition.name,
        risk_level=tool_definition.risk_level,
        action=action,
        reason=reason,
        approval=approval,
    )


def grant_tool_approval(decision: PolicyDecision) -> PolicyDecision:
    return PolicyDecision(
        tool_name=decision.tool_name,
        risk_level=decision.risk_level,
        action="allow",
        reason=f"policy allows {decision.risk_level} risk tool {decision.tool_name}",
        approval=ApprovalDecision(
            required=True,
            status="granted",
            reason=f"approval granted for high risk tool {decision.tool_name}",
        ),
    )


def deny_tool_approval(decision: PolicyDecision) -> PolicyDecision:
    return PolicyDecision(
        tool_name=decision.tool_name,
        risk_level=decision.risk_level,
        action="deny",
        reason=f"policy denies {decision.risk_level} risk tool {decision.tool_name}",
        approval=ApprovalDecision(
            required=True,
            status="denied",
            reason=f"approval denied for high risk tool {decision.tool_name}",
        ),
    )
