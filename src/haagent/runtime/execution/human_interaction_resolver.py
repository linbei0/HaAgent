"""
src/haagent/runtime/execution/human_interaction_resolver.py - 人机交互解析器

记录同一次 run 内已完成的人机交互，并按稳定签名复用审批和补充信息结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from haagent.runtime.execution.command import redact_secret_like_text
from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse


ANSWER_EXCERPT_LIMIT = 240
QUESTION_EXCERPT_LIMIT = 180


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
    def __init__(self) -> None:
        self._resolutions: dict[str, HumanInteractionResolution] = {}
        self._ordered_signatures: list[str] = []

    def resolve(self, request: HumanInteractionRequest) -> HumanInteractionResolution | None:
        return self._resolutions.get(interaction_signature(request))

    def record(
        self,
        request: HumanInteractionRequest,
        response: HumanInteractionResponse,
        *,
        turn: int,
    ) -> HumanInteractionResolution:
        signature = interaction_signature(request)
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
