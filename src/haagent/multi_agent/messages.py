"""
haagent/multi_agent/messages.py - worker 通信数据结构

定义主 Agent 与 worker 之间的结构化通知、消息和权限请求。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkerNotification:
    event_type: str
    team_id: str
    agent_id: str
    task_id: str
    status: str
    summary: str
    result_excerpt: str
    episode_path: str
    error: str
    needs_attention: bool
    request_id: str = ""
    parent_step_id: str = ""
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerMessage:
    sender: str
    recipient: str
    content: str
    message_id: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["message_id"] = self.message_id or f"msg-{uuid.uuid4().hex[:12]}"
        payload["created_at"] = self.created_at or time.time()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WorkerMessage":
        return cls(
            sender=str(payload["sender"]),
            recipient=str(payload["recipient"]),
            content=str(payload["content"]),
            message_id=str(payload["message_id"]),
            created_at=float(payload["created_at"]),
        )


@dataclass(frozen=True)
class WorkerPermissionRequest:
    request_id: str
    team_id: str
    agent_id: str
    task_id: str
    tool_name: str
    tool_args_summary: str
    reason: str
    status: str
    response_message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "WorkerPermissionRequest":
        return cls(
            request_id=str(payload["request_id"]),
            team_id=str(payload["team_id"]),
            agent_id=str(payload["agent_id"]),
            task_id=str(payload["task_id"]),
            tool_name=str(payload["tool_name"]),
            tool_args_summary=str(payload["tool_args_summary"]),
            reason=str(payload["reason"]),
            status=str(payload["status"]),
            response_message=str(payload.get("response_message", "")),
        )
