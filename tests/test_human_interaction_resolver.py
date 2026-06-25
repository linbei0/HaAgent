"""
tests/test_human_interaction_resolver.py - 人机交互解析器测试

验证审批和用户补充信息能按通用签名复用，并生成中性状态摘要。
"""

from haagent.runtime.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.runtime.human_interaction_resolver import HumanInteractionResolver


def _user_input_request(question: str = "Which file?", path: str = "README.md") -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        question=question,
        reason="Need target",
        args_summary={"question": question, "path": path},
    )


def _approval_request(command: str = "echo approved") -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="approval",
        tool_name="shell",
        question="Approve high risk tool shell?",
        reason="policy requires approval",
        risk_level="high",
        args_summary={"command": command, "cwd": "."},
    )


def test_resolver_reuses_answered_user_input_by_signature() -> None:
    resolver = HumanInteractionResolver()
    request = _user_input_request()

    resolution = resolver.record(request, HumanInteractionResponse(approved=True, answer="README.md"), turn=1)
    reused = resolver.resolve(_user_input_request())

    assert reused == resolution
    assert reused is not None
    assert reused.to_response() == HumanInteractionResponse(approved=True, answer="README.md")
    assert len(resolver.state_records()) == 1
    assert resolver.state_records()[0]["status"] == "answered"


def test_resolver_reuses_declined_user_input() -> None:
    resolver = HumanInteractionResolver()
    request = _user_input_request()

    resolver.record(request, HumanInteractionResponse(approved=False, answer=""), turn=1)
    reused = resolver.resolve(_user_input_request())

    assert reused is not None
    assert reused.status == "declined"
    assert reused.to_response() == HumanInteractionResponse(approved=False, answer="")


def test_resolver_reuses_approval_decisions() -> None:
    resolver = HumanInteractionResolver()

    approved = resolver.record(_approval_request(), HumanInteractionResponse(approved=True), turn=1)
    denied = resolver.record(
        _approval_request(command="rm -rf scratch"),
        HumanInteractionResponse(approved=False),
        turn=2,
    )

    assert resolver.resolve(_approval_request()) == approved
    assert resolver.resolve(_approval_request(command="rm -rf scratch")) == denied
    assert resolver.resolve(_approval_request()).to_response().approved is True
    assert resolver.resolve(_approval_request(command="rm -rf scratch")).to_response().approved is False


def test_resolver_does_not_reuse_different_args_summary() -> None:
    resolver = HumanInteractionResolver()
    resolver.record(_user_input_request(path="README.md"), HumanInteractionResponse(approved=True, answer="README.md"), turn=1)

    assert resolver.resolve(_user_input_request(path="docs/harness-requirements.md")) is None


def test_state_records_are_neutral_bounded_and_redacted() -> None:
    resolver = HumanInteractionResolver()
    secret = "sk-test1234567890abcdef1234567890abcdef"
    resolver.record(
        _user_input_request(question="Token?", path=secret),
        HumanInteractionResponse(approved=True, answer=secret + ("A" * 400)),
        turn=1,
    )

    record = resolver.state_records()[0]

    assert record["type"] == "user_input"
    assert record["tool"] == "request_user_input"
    assert record["status"] == "answered"
    assert secret not in str(record)
    assert "[REDACTED_TOKEN]" in str(record)
    assert len(str(record["answer_excerpt"])) < 280
