"""
tests/unit/channels/test_interactions.py - InteractionBroker 审批与补充输入测试
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import pytest

from haagent.channels.interactions import InteractionBroker, InteractionError
from haagent.channels.types import ChannelAddress, ChannelReplyHandle, InboundChannelMessage
from haagent.runtime.execution.human_interaction import HumanInteractionRequest


def _address() -> ChannelAddress:
    return ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="owner-1",
    )


def _message(text: str = "hello") -> InboundChannelMessage:
    return InboundChannelMessage(
        address=_address(),
        message_id="m-1",
        sender_id="owner-1",
        text=text,
        received_at=datetime.now(timezone.utc),
        reply_handle=ChannelReplyHandle(platform="weixin", payload={"context_token": "hidden"}),
    )


def test_approval_blocks_worker_and_owner_nonce_approves() -> None:
    broker = InteractionBroker(timeout_seconds=5.0)
    request = HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="run tests?",
        reason="high risk",
        args_summary={"command": "pytest"},
    )
    result_holder: list[object] = []

    def worker() -> None:
        result_holder.append(
            broker.request_approval(
                request,
                address=_address(),
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    assert pending.kind == "approval"
    assert "pytest" not in repr(pending) or "command" in repr(pending)
    assert broker.resolve(pending.nonce, approved=True, sender_id="owner-1", binding_key=_address().binding_key())
    thread.join(timeout=3)
    assert len(result_holder) == 1
    assert result_holder[0].approved is True


def test_non_owner_wrong_expired_duplicate_rejected() -> None:
    broker = InteractionBroker(timeout_seconds=0.2)
    request = HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="run?",
    )
    holder: list[object] = []

    def worker() -> None:
        holder.append(
            broker.request_approval(
                request,
                address=_address(),
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    with pytest.raises(InteractionError):
        broker.resolve(pending.nonce, approved=True, sender_id="other", binding_key=_address().binding_key())
    with pytest.raises(InteractionError):
        broker.resolve("WRONG01", approved=True, sender_id="owner-1", binding_key=_address().binding_key())
    # timeout path
    thread.join(timeout=2)
    assert holder[0].approved is False

    # duplicate after resolve
    holder2: list[object] = []

    def worker2() -> None:
        holder2.append(
            broker.request_approval(
                request,
                address=_address(),
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    t2 = threading.Thread(target=worker2)
    t2.start()
    pending2 = broker.wait_for_pending(timeout=2.0)
    assert pending2 is not None
    broker.resolve(pending2.nonce, approved=False, sender_id="owner-1", binding_key=_address().binding_key())
    t2.join(timeout=2)
    with pytest.raises(InteractionError):
        broker.resolve(pending2.nonce, approved=True, sender_id="owner-1", binding_key=_address().binding_key())


def test_answer_separated_from_approval() -> None:
    broker = InteractionBroker(timeout_seconds=5.0)
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        question="which file?",
    )
    holder: list[object] = []

    def worker() -> None:
        holder.append(
            broker.request_user_input(
                request,
                address=_address(),
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    assert pending.kind == "user_input"
    broker.resolve_answer(
        pending.nonce,
        answer="readme.md",
        sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    thread.join(timeout=3)
    assert holder[0].approved is True
    assert holder[0].answer == "readme.md"
    # full answer must not appear in pending/state repr after settle if we keep history carefully
    assert "readme.md" not in repr(pending)


def test_pending_repr_hides_secret_like_params() -> None:
    broker = InteractionBroker(timeout_seconds=1.0)
    request = HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="run",
        args_summary={"command": "echo secret-token-xyz", "api_key": "sk-test"},
    )
    holder: list[object] = []

    def worker() -> None:
        holder.append(
            broker.request_approval(
                request,
                address=_address(),
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    text = repr(pending)
    assert "secret-token-xyz" not in text
    assert "sk-test" not in text
    broker.resolve(pending.nonce, approved=False, sender_id="owner-1", binding_key=_address().binding_key())
    thread.join(timeout=2)
