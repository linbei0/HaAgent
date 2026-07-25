"""
tests/unit/channels/test_interactions.py - InteractionBroker 审批与补充输入测试
"""

from __future__ import annotations

import threading

import pytest

from haagent.channels.interactions import InteractionBroker, InteractionError, PendingInteraction
from haagent.channels.types import ChannelAddress
from haagent.runtime.execution.human_interaction import (
    HumanInteractionRequest,
    UserQuestion,
    UserQuestionOption,
)


def _address() -> ChannelAddress:
    return ChannelAddress(
        instance_id="wx-1",
        platform="weixin",
        conversation_kind="dm",
        conversation_id="owner-1",
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
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    assert pending.kind == "approval"
    assert "pytest" not in repr(pending)
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


def test_user_input_questions_use_separate_nonces_and_collect_answers() -> None:
    broker = InteractionBroker(timeout_seconds=5.0)
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        questions=(
            UserQuestion(id="file", header="文件", question="选择文件。"),
            UserQuestion(
                id="mode",
                header="模式",
                question="选择模式。",
                options=(
                    UserQuestionOption(label="快速（推荐）", description="更少检查。"),
                    UserQuestionOption(label="完整", description="运行完整检查。"),
                ),
                custom=False,
            ),
        ),
    )
    holder: list[object] = []

    def worker() -> None:
        holder.append(
            broker.request_user_input(
                request,
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    assert pending.kind == "user_input"
    assert pending.question_id == "file"
    broker.resolve_answer(
        pending.nonce,
        answer="readme.md",
        sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    second = broker.wait_for_pending(timeout=2.0, exclude_nonces={pending.nonce})
    assert second is not None
    assert second.nonce != pending.nonce
    assert second.question_id == "mode"
    broker.resolve_answer(
        second.nonce,
        answer="1",
        sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    thread.join(timeout=3)
    assert holder[0].approved is True
    assert holder[0].outcome == "answered"
    assert holder[0].answers == {"file": ("readme.md",), "mode": ("快速（推荐）",)}
    assert "readme.md" not in repr(pending)


def test_user_input_multiple_choice_and_custom_validation_keep_nonce_pending() -> None:
    broker = InteractionBroker(timeout_seconds=5.0)
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        questions=(
            UserQuestion(
                id="checks",
                header="检查",
                question="选择检查。",
                options=(
                    UserQuestionOption(label="单测", description="运行单元测试。"),
                    UserQuestionOption(label="类型", description="运行类型检查。"),
                    UserQuestionOption(label="格式", description="运行格式检查。"),
                ),
                multiple=True,
                custom=False,
            ),
        ),
    )
    holder: list[object] = []

    thread = threading.Thread(
        target=lambda: holder.append(
            broker.request_user_input(
                request,
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )
    )
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    with pytest.raises(InteractionError, match="编号"):
        broker.resolve_answer(
            pending.nonce,
            answer="1,4",
            sender_id="owner-1",
            binding_key=_address().binding_key(),
        )
    with pytest.raises(InteractionError, match="自定义"):
        broker.resolve_answer(
            pending.nonce,
            answer="只运行关键检查",
            sender_id="owner-1",
            binding_key=_address().binding_key(),
        )
    assert broker.wait_for_pending(timeout=0.1) == pending
    broker.resolve_answer(
        pending.nonce,
        answer="1,3",
        sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    thread.join(timeout=3)
    assert holder[0].answers == {"checks": ("单测", "格式")}


def test_user_input_timeout_and_dismiss_are_distinct() -> None:
    request = HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        questions=(UserQuestion(id="note", header="说明", question="补充说明。"),),
    )
    timed_out = InteractionBroker(timeout_seconds=0.05).request_user_input(
        request,
        owner_sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    assert timed_out.outcome == "timed_out"

    broker = InteractionBroker(timeout_seconds=5.0)
    holder: list[object] = []
    thread = threading.Thread(
        target=lambda: holder.append(
            broker.request_user_input(
                request,
                owner_sender_id="owner-1",
                binding_key=_address().binding_key(),
            )
        )
    )
    thread.start()
    pending = broker.wait_for_pending(timeout=2.0)
    assert pending is not None
    broker.dismiss(
        pending.nonce,
        sender_id="owner-1",
        binding_key=_address().binding_key(),
    )
    thread.join(timeout=3)
    assert holder[0].outcome == "dismissed"


def test_channel_prompt_lists_option_numbers_and_commands() -> None:
    from haagent.channels.session_actor import _format_pending_interaction

    question = UserQuestion(
        id="checks",
        header="检查范围",
        question="请选择检查。",
        options=(
            UserQuestionOption(label="单测（推荐）", description="速度快。"),
            UserQuestionOption(label="完整", description="覆盖更全面。"),
        ),
        multiple=True,
        custom=False,
    )
    pending = PendingInteraction(
        nonce="ABC123",
        kind="user_input",
        owner_sender_id="owner-1",
        binding_key=_address().binding_key(),
        tool_name="request_user_input",
        question=question.question,
        user_question=question,
    )

    text = _format_pending_interaction(pending)

    assert "检查范围" in text
    assert "1. 单测（推荐） — 速度快。" in text
    assert "2. 完整 — 覆盖更全面。" in text
    assert "/answer ABC123 1,2" in text
    assert "/dismiss ABC123" in text


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
