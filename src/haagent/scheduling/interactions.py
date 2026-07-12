"""
haagent/scheduling/interactions.py - 无人值守人机交互失败边界

计划任务运行不得自动批准工具或回答用户问题；任何 interaction 请求显式失败。
"""

from __future__ import annotations

from haagent.runtime.execution.human_interaction import (
    HumanInteractionRequest,
    HumanInteractionResponse,
)


class UnattendedInteractionRequired(RuntimeError):
    """无人值守运行收到需要人工处理的请求。"""

    def __init__(self, *, kind: str, summary: str) -> None:
        self.kind = kind
        self.summary = summary
        super().__init__(f"unattended interaction required: {kind}: {summary}")


def _safe_summary(text: str, *, limit: int = 240) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


class UnattendedInteractionHandler:
    """始终拒绝；绝不返回虚构批准或空答案。"""

    def request(self, request: HumanInteractionRequest) -> HumanInteractionResponse:
        # 安全边界：无人值守禁止伪造批准/默认选项
        kind = request.interaction_type or "interaction"
        parts = [request.question or "", request.reason or "", request.tool_name or ""]
        summary = _safe_summary(" ".join(p for p in parts if p).strip() or kind)
        raise UnattendedInteractionRequired(kind=kind, summary=summary)

    def __call__(self, request: HumanInteractionRequest) -> HumanInteractionResponse:
        return self.request(request)
