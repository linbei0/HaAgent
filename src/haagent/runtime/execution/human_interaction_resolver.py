"""
src/haagent/runtime/execution/human_interaction_resolver.py - 人机交互解析器

记录同一次 run 内已完成的人机交互，并按稳定签名复用审批和补充信息结果。
edit_diff 的「本会话始终允许」与 permission_mode 自动跳过由结构化 session 状态决定，
不得用完整 path/diff 签名冒充同类改动，也不得伪造用户手动点击事件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from haagent.runtime.execution.command import redact_secret_like_text
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.runtime.execution.path_policy import PermissionMode


ANSWER_EXCERPT_LIMIT = 240
QUESTION_EXCERPT_LIMIT = 180

# 审计用合成 answer：标明跳过原因，避免被当成用户在 modal 上点了 once/always
MODE_AUTO_ANSWER = "mode_auto"
SESSION_ALWAYS_ANSWER = "session_always"
STATUS_MODE_AUTO = "mode_auto_approved"
STATUS_SESSION_ALWAYS = "session_always_allowed"


@dataclass
class SessionInteractionState:
    """跨 turn / resume 的会话级交互状态（仅结构化标志，不含完整 diff）。"""

    edit_diff_session_always: bool = False


@dataclass(frozen=True)
class HumanInteractionResolution:
    signature: str
    interaction_type: str
    tool_name: str
    question: str
    status: str
    approved: bool
    answer: str
    turn: int
    args_summary: dict[str, object]

    def to_response(self) -> HumanInteractionResponse:
        return HumanInteractionResponse(approved=self.approved, answer=self.answer)

    def state_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "type": self.interaction_type,
            "tool": self.tool_name,
            "status": self.status,
            "question": _safe_excerpt(self.question, QUESTION_EXCERPT_LIMIT),
            "turn": self.turn,
        }
        if self.interaction_type == "user_input":
            record["answer_excerpt"] = _safe_excerpt(self.answer, ANSWER_EXCERPT_LIMIT)
            record["answer_chars"] = len(self.answer)
        else:
            record["approved"] = self.approved
        return record


class HumanInteractionResolver:
    def __init__(
        self,
        *,
        permission_mode: PermissionMode = "request_approval",
        edit_diff_session_always: bool = False,
        session_interaction_state: SessionInteractionState | None = None,
    ) -> None:
        self._permission_mode: PermissionMode = permission_mode
        # 允许外部传入同一 session 的可变状态，使 always 能跨 turn 并随 metadata 持久化
        if session_interaction_state is not None:
            self._session_state = session_interaction_state
        else:
            self._session_state = SessionInteractionState(
                edit_diff_session_always=edit_diff_session_always,
            )
        self._resolutions: dict[str, HumanInteractionResolution] = {}
        self._ordered_signatures: list[str] = []

    def resolve(self, request: HumanInteractionRequest) -> HumanInteractionResolution | None:
        # edit_diff：先看 permission_mode / session always，再看完整签名（once 复用）
        if request.interaction_type == "edit_diff":
            if self._permission_mode in {"auto_approve", "full_access"}:
                return self._synthetic_edit_diff(
                    request,
                    status=STATUS_MODE_AUTO,
                    answer=MODE_AUTO_ANSWER,
                )
            if self._session_state.edit_diff_session_always:
                return self._synthetic_edit_diff(
                    request,
                    status=STATUS_SESSION_ALWAYS,
                    answer=SESSION_ALWAYS_ANSWER,
                )
        return self._resolutions.get(interaction_signature(request))

    def record(
        self,
        request: HumanInteractionRequest,
        response: HumanInteractionResponse,
        *,
        turn: int,
    ) -> HumanInteractionResolution:
        signature = interaction_signature(request)
        # always 只提升 edit_diff 会话标志，绝不扩展到 shell/code_run 等 approval 类别
        if (
            request.interaction_type == "edit_diff"
            and response.approved
            and response.answer == "always"
        ):
            self._session_state.edit_diff_session_always = True
        resolution = HumanInteractionResolution(
            signature=signature,
            interaction_type=request.interaction_type,
            tool_name=request.tool_name,
            question=request.question,
            status=_status_for(request.interaction_type, response.approved),
            approved=response.approved,
            answer=response.answer,
            turn=turn,
            args_summary=dict(request.args_summary),
        )
        if signature not in self._resolutions:
            self._ordered_signatures.append(signature)
        self._resolutions[signature] = resolution
        return resolution

    def state_records(self) -> list[dict[str, object]]:
        return [
            self._resolutions[signature].state_record()
            for signature in self._ordered_signatures
            if signature in self._resolutions
        ]

    def _synthetic_edit_diff(
        self,
        request: HumanInteractionRequest,
        *,
        status: str,
        answer: str,
    ) -> HumanInteractionResolution:
        # 合成解析仅用于自动跳过；signature 仍按请求计算便于 trace，但不写入 once 复用表
        return HumanInteractionResolution(
            signature=interaction_signature(request),
            interaction_type=request.interaction_type,
            tool_name=request.tool_name,
            question=request.question,
            status=status,
            approved=True,
            answer=answer,
            turn=0,
            args_summary=dict(request.args_summary),
        )


def interaction_signature(request: HumanInteractionRequest) -> str:
    payload = {
        "interaction_type": _normalize_string(request.interaction_type),
        "tool_name": _normalize_string(request.tool_name),
        "question": _normalize_string(request.question),
        "args_summary": _normalize_value(request.args_summary),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _status_for(interaction_type: str, approved: bool) -> str:
    if interaction_type == "user_input":
        return "answered" if approved else "declined"
    return "approved" if approved else "denied"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return _normalize_string(str(value))


def _normalize_string(value: str) -> str:
    return " ".join(value.split()).casefold()


def _safe_excerpt(value: str, limit: int) -> str:
    redacted, _ = redact_secret_like_text(value)
    normalized = " ".join(redacted.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"
