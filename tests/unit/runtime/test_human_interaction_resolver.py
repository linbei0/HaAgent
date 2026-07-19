"""
tests/unit/runtime/test_human_interaction_resolver.py - 人机交互解析器测试

验证审批和用户补充信息能按签名复用；edit_diff 的 once/always 语义与
permission_mode 自动跳过必须可区分，且 always 不得跨类别批准 shell。
"""

from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver


def _user_input_request(question: str = "Which file?", path: str = "README.md") -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="user_input",
        tool_name="request_user_input",
        question=question,
        reason="Need target",
        args_summary={"question": question, "path": path},
    )


def _approval_request(command: str = "echo approved", tool_name: str = "shell") -> HumanInteractionRequest:
    command_prefix = command.split(maxsplit=1)[0]
    return HumanInteractionRequest(
        interaction_type="approval",
        tool_name=tool_name,
        question=f"Approve high risk tool {tool_name}?",
        reason="policy requires approval",
        risk_level="high",
        args_summary={
            "command": command,
            "cwd": ".",
            "permission_patterns": [command],
            "permission_always": [f"{command_prefix} *"],
        },
    )


def _edit_diff_request(
    path: str = "notes.txt",
    *,
    tool_name: str = "file_write",
    diff_preview: str = "+hello\n",
) -> HumanInteractionRequest:
    return HumanInteractionRequest(
        interaction_type="edit_diff",
        tool_name=tool_name,
        question=f"Approve file edit for {path}?",
        reason=f"{tool_name} will modify {path}",
        risk_level="high",
        args_summary={
            "path": path,
            "change_type": "added",
            "additions": 1,
            "deletions": 0,
            "diff_preview": diff_preview,
        },
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


def test_resolver_does_not_reuse_allow_once_or_denial() -> None:
    resolver = HumanInteractionResolver()

    resolver.record(
        _approval_request(),
        HumanInteractionResponse(approved=True, answer="once"),
        turn=1,
    )
    resolver.record(
        _approval_request(command="rm -rf scratch"),
        HumanInteractionResponse(approved=False, answer="deny"),
        turn=2,
    )

    assert resolver.resolve(_approval_request()) is None
    assert resolver.resolve(_approval_request(command="rm -rf scratch")) is None


def test_resolver_reuses_always_permission_rule_by_pattern() -> None:
    resolver = HumanInteractionResolver()
    resolver.record(
        _approval_request(command="pytest tests/unit"),
        HumanInteractionResponse(approved=True, answer="always"),
        turn=1,
    )

    reused = resolver.resolve(_approval_request(command="pytest tests/integration"))

    assert reused is not None
    assert reused.status == "session_always_allowed"
    assert reused.to_response() == HumanInteractionResponse(
        approved=True,
        answer="session_always",
    )


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


def test_edit_diff_once_does_not_reuse_for_different_file() -> None:
    """once 只批准当前这一次改动；不同路径必须再次请求。"""
    resolver = HumanInteractionResolver(permission_mode="request_approval")
    first = _edit_diff_request("a.txt", diff_preview="+a\n")
    second = _edit_diff_request("b.txt", diff_preview="+b\n")

    resolver.record(first, HumanInteractionResponse(approved=True, answer="once"), turn=1)

    assert resolver.resolve(first) is not None
    assert resolver.resolve(first).to_response().approved is True
    assert resolver.resolve(second) is None


def test_edit_diff_always_reuses_for_different_files_same_session() -> None:
    """always 免除当前 session 后续全部 edit_diff，不依赖具体 path/diff 签名。"""
    resolver = HumanInteractionResolver(permission_mode="request_approval")
    first = _edit_diff_request("a.txt", diff_preview="+a\n")
    second = _edit_diff_request("b.txt", diff_preview="+b\n")
    third = _edit_diff_request("c.py", tool_name="apply_patch", diff_preview="+c\n")

    resolver.record(first, HumanInteractionResponse(approved=True, answer="always"), turn=1)

    reused_b = resolver.resolve(second)
    reused_c = resolver.resolve(third)
    assert reused_b is not None
    assert reused_c is not None
    assert reused_b.status == "session_always_allowed"
    assert reused_c.status == "session_always_allowed"
    assert reused_b.to_response().approved is True
    assert reused_c.to_response().approved is True
    # 审计标记不得伪装成用户刚点击了批准
    assert reused_b.to_response().answer == "session_always"
    assert "always" != reused_b.signature  # 类别复用，不是完整签名复用


def test_edit_diff_always_does_not_approve_shell_or_code_run() -> None:
    resolver = HumanInteractionResolver(permission_mode="request_approval")
    resolver.record(
        _edit_diff_request("a.txt"),
        HumanInteractionResponse(approved=True, answer="always"),
        turn=1,
    )

    assert resolver.resolve(_approval_request(tool_name="shell")) is None
    assert resolver.resolve(_approval_request(tool_name="code_run", command="print(1)")) is None


def test_new_resolver_does_not_inherit_edit_diff_always() -> None:
    """新 session / 新 resolver 默认不继承 always。"""
    previous = HumanInteractionResolver(permission_mode="request_approval")
    previous.record(
        _edit_diff_request("a.txt"),
        HumanInteractionResponse(approved=True, answer="always"),
        turn=1,
    )
    fresh = HumanInteractionResolver(permission_mode="request_approval")
    assert fresh.resolve(_edit_diff_request("b.txt")) is None


def test_resolver_restores_edit_diff_session_always_from_session_state() -> None:
    """resume 同一 session 时应恢复 edit_diff always 状态。"""
    restored = HumanInteractionResolver(
        permission_mode="request_approval",
        edit_diff_session_always=True,
    )
    resolution = restored.resolve(_edit_diff_request("other.txt", diff_preview="+other\n"))
    assert resolution is not None
    assert resolution.status == "session_always_allowed"
    assert resolution.to_response().approved is True


def test_auto_approve_mode_skips_edit_diff_without_forging_user_click() -> None:
    resolver = HumanInteractionResolver(permission_mode="auto_approve")
    resolution = resolver.resolve(_edit_diff_request("a.txt"))

    assert resolution is not None
    assert resolution.status == "mode_auto_approved"
    assert resolution.to_response().approved is True
    assert resolution.to_response().answer == "mode_auto"
    # 未经过用户 handler 的 record，state 仍可审计
    assert resolution.approved is True


def test_full_access_mode_skips_edit_diff_without_forging_user_click() -> None:
    resolver = HumanInteractionResolver(permission_mode="full_access")
    resolution = resolver.resolve(_edit_diff_request("a.txt"))

    assert resolution is not None
    assert resolution.status == "mode_auto_approved"
    assert resolution.to_response().approved is True
    assert resolution.to_response().answer == "mode_auto"


def test_request_approval_default_still_requires_edit_diff() -> None:
    resolver = HumanInteractionResolver(permission_mode="request_approval")
    assert resolver.resolve(_edit_diff_request("a.txt")) is None


def test_auto_approve_mode_does_not_auto_approve_tool_policy_via_edit_diff_path() -> None:
    """auto_approve 对 edit_diff 跳过，但 approval 类型仍走原签名复用逻辑（无预存则不复用）。"""
    resolver = HumanInteractionResolver(permission_mode="auto_approve")
    # tool policy 由 approved_tools 预填处理；resolver 不对任意 approval 无条件放行
    assert resolver.resolve(_approval_request()) is None
