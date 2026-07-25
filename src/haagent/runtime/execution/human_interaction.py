"""
src/haagent/runtime/execution/human_interaction.py - 人机交互请求协议

定义 runtime 内部传递澄清问题和高风险审批请求的最小结构。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


UserInputOutcome = Literal["answered", "dismissed", "timed_out"]


@dataclass(frozen=True)
class UserQuestionOption:
    label: str
    description: str


@dataclass(frozen=True)
class UserQuestion:
    id: str
    header: str
    question: str
    options: tuple[UserQuestionOption, ...] = ()
    multiple: bool = False
    custom: bool = True
    placeholder: str = ""


@dataclass(frozen=True)
class HumanInteractionRequest:
    interaction_type: str
    tool_name: str
    question: str = ""
    reason: str = ""
    risk_level: str | None = None
    args_summary: dict[str, object] = field(default_factory=dict)
    questions: tuple[UserQuestion, ...] = ()


@dataclass(frozen=True)
class HumanInteractionResponse:
    approved: bool
    answer: str = ""
    outcome: UserInputOutcome | None = None
    answers: dict[str, tuple[str, ...]] = field(default_factory=dict)


HumanInteractionHandler = Callable[[HumanInteractionRequest], HumanInteractionResponse]


def interaction_question_summary(request: HumanInteractionRequest, *, limit: int = 120) -> str:
    """返回可写入事件和 working state 的短标题，不复制完整问题正文。"""

    value = "、".join(question.header for question in request.questions) or request.question
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


@dataclass(frozen=True)
class ToolPermissionRequest:
    """工具执行期间提交给统一权限入口的结构化请求。"""

    permission: str
    patterns: tuple[str, ...]
    always: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    question: str = ""
    reason: str = ""
    risk_level: str | None = None


def interaction_args_summary(tool_name: str, args: dict[str, Any]) -> dict[str, object]:
    # 静态工具摘要由 ToolCatalog contribution 提供；未知/动态工具走通用回退。
    from haagent.tools.catalog import default_tool_catalog

    return default_tool_catalog().interaction_args_summary(tool_name, args)
