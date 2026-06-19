"""
agentfoundry/runtime/policy.py - Policy Engine v0

为工具调用生成审计用 policy decision；v0 只记录 allow，不阻止调用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentfoundry.tools.registry import ToolDefinition


@dataclass(frozen=True)
class PolicyDecision:
    tool_name: str
    risk_level: str
    action: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def evaluate_tool_call(tool_definition: ToolDefinition) -> PolicyDecision:
    """返回工具调用的审计决策；v0 所有已知工具都 allow。"""
    reason = f"policy v0 allows {tool_definition.risk_level} risk tool {tool_definition.name}"
    if tool_definition.risk_level == "high":
        reason += " for audit-only enforcement"
    return PolicyDecision(
        tool_name=tool_definition.name,
        risk_level=tool_definition.risk_level,
        action="allow",
        reason=reason,
    )
