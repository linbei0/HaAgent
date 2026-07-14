"""
tests/unit/scheduling/test_interactions.py - 无人值守 interaction handler

验证 UnattendedInteractionHandler 始终失败，绝不伪造批准或空答案。
"""

from __future__ import annotations

import pytest

from haagent.runtime.execution.human_interaction import HumanInteractionRequest
from haagent.scheduling.interactions import (
    UnattendedInteractionHandler,
    UnattendedInteractionRequired,
)


def test_request_always_raises_unattended_required() -> None:
    handler = UnattendedInteractionHandler()
    request = HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="允许执行 shell 吗？",
        reason="高风险命令",
    )
    with pytest.raises(UnattendedInteractionRequired) as exc:
        handler.request(request)
    assert exc.value.kind == "approval"
    assert "shell" in exc.value.summary


def test_callable_handler_also_raises() -> None:
    handler = UnattendedInteractionHandler()
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        question="请补充目标路径",
    )
    with pytest.raises(UnattendedInteractionRequired) as exc:
        handler(request)
    assert exc.value.kind == "user_input"
    assert exc.value.summary
