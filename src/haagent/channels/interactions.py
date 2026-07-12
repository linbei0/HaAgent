"""
haagent/channels/interactions.py - 远程审批与补充输入 broker

将同步 HumanInteractionHandler 桥接为聊天命令 /approve /deny /answer。
"""

from __future__ import annotations

import secrets
import string
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

from haagent.channels.types import ChannelAddress
from haagent.runtime.execution.human_interaction import (
    HumanInteractionRequest,
    HumanInteractionResponse,
)


class InteractionError(RuntimeError):
    """nonce、sender 或 binding 校验失败。"""


@dataclass(repr=False)
class PendingInteraction:
    nonce: str
    kind: Literal["approval", "user_input"]
    address: ChannelAddress
    owner_sender_id: str
    binding_key: str
    tool_name: str
    question: str
    created_at: float = field(default_factory=time.time)
    # 敏感参数摘要只在内存中用于生成提示，repr 不得泄露。
    _args_summary: dict[str, object] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return (
            f"PendingInteraction(nonce={self.nonce!r}, kind={self.kind!r}, "
            f"tool_name={self.tool_name!r}, binding_key={self.binding_key!r})"
        )


class InteractionBroker:
    def __init__(self, *, timeout_seconds: float = 600.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._pending: dict[str, PendingInteraction] = {}
        self._events: dict[str, threading.Event] = {}
        self._results: dict[str, HumanInteractionResponse] = {}
        self._notify = threading.Event()

    def request_approval(
        self,
        request: HumanInteractionRequest,
        *,
        address: ChannelAddress,
        owner_sender_id: str,
        binding_key: str,
    ) -> HumanInteractionResponse:
        return self._request(
            kind="approval",
            request=request,
            address=address,
            owner_sender_id=owner_sender_id,
            binding_key=binding_key,
        )

    def request_user_input(
        self,
        request: HumanInteractionRequest,
        *,
        address: ChannelAddress,
        owner_sender_id: str,
        binding_key: str,
    ) -> HumanInteractionResponse:
        return self._request(
            kind="user_input",
            request=request,
            address=address,
            owner_sender_id=owner_sender_id,
            binding_key=binding_key,
        )

    def _request(
        self,
        *,
        kind: Literal["approval", "user_input"],
        request: HumanInteractionRequest,
        address: ChannelAddress,
        owner_sender_id: str,
        binding_key: str,
    ) -> HumanInteractionResponse:
        nonce = _generate_nonce()
        pending = PendingInteraction(
            nonce=nonce,
            kind=kind,
            address=address,
            owner_sender_id=owner_sender_id,
            binding_key=binding_key,
            tool_name=request.tool_name,
            question=request.question,
            _args_summary=dict(request.args_summary),
        )
        event = threading.Event()
        with self._lock:
            self._pending[nonce] = pending
            self._events[nonce] = event
        self._notify.set()
        finished = event.wait(timeout=self._timeout_seconds)
        with self._lock:
            self._pending.pop(nonce, None)
            self._events.pop(nonce, None)
            result = self._results.pop(nonce, None)
        if not finished or result is None:
            # 超时视为明确拒绝，不静默继续。
            return HumanInteractionResponse(approved=False, answer="")
        return result

    def wait_for_pending(
        self,
        *,
        timeout: float = 1.0,
        binding_key: str | None = None,
        exclude_nonces: set[str] | frozenset[str] | None = None,
    ) -> PendingInteraction | None:
        """
        等待 pending interaction。

        binding_key：只返回该会话的项，避免多会话审批饥饿。
        exclude_nonces：跳过已提示过的 nonce，避免高频空转。
        """
        excluded = set(exclude_nonces or ())
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                for pending in self._pending.values():
                    if binding_key is not None and pending.binding_key != binding_key:
                        continue
                    if pending.nonce in excluded:
                        continue
                    return pending
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            # 无匹配时真正阻塞等待通知，而不是立刻返回首项。
            self._notify.wait(timeout=min(0.05, remaining))
            self._notify.clear()
        return None

    def resolve(
        self,
        nonce: str,
        *,
        approved: bool,
        sender_id: str,
        binding_key: str,
        answer: str = "",
    ) -> bool:
        with self._lock:
            pending = self._pending.get(nonce)
            if pending is None:
                raise InteractionError("unknown or reused nonce")
            if pending.kind != "approval":
                raise InteractionError("nonce is not an approval")
            self._validate_actor(pending, sender_id=sender_id, binding_key=binding_key)
            self._results[nonce] = HumanInteractionResponse(approved=approved, answer=answer)
            event = self._events.get(nonce)
        if event is not None:
            event.set()
        return True

    def resolve_answer(
        self,
        nonce: str,
        *,
        answer: str,
        sender_id: str,
        binding_key: str,
    ) -> bool:
        with self._lock:
            pending = self._pending.get(nonce)
            if pending is None:
                raise InteractionError("unknown or reused nonce")
            if pending.kind != "user_input":
                raise InteractionError("nonce is not a user_input")
            self._validate_actor(pending, sender_id=sender_id, binding_key=binding_key)
            self._results[nonce] = HumanInteractionResponse(approved=True, answer=answer)
            event = self._events.get(nonce)
        if event is not None:
            event.set()
        return True

    def _validate_actor(
        self,
        pending: PendingInteraction,
        *,
        sender_id: str,
        binding_key: str,
    ) -> None:
        if sender_id != pending.owner_sender_id:
            raise InteractionError("sender is not owner")
        if binding_key != pending.binding_key:
            raise InteractionError("binding mismatch")

    def build_handler(
        self,
        *,
        address: ChannelAddress,
        owner_sender_id: str,
        binding_key: str,
    ):
        def handler(request: HumanInteractionRequest) -> HumanInteractionResponse:
            if request.interaction_type == "user_input":
                return self.request_user_input(
                    request,
                    address=address,
                    owner_sender_id=owner_sender_id,
                    binding_key=binding_key,
                )
            return self.request_approval(
                request,
                address=address,
                owner_sender_id=owner_sender_id,
                binding_key=binding_key,
            )

        return handler


def _generate_nonce(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
