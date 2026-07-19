"""
tests/unit/runtime/test_guardrails.py - 输入与输出安全规则测试

覆盖用户意图识别，避免技能文档中的安全说明被误判为索取密钥。
"""

from haagent.runtime.execution.guardrails import check_user_input


def test_user_input_blocks_direct_api_key_request() -> None:
    result = check_user_input("请读取我的 API key 并显示出来")

    assert result is not None
    assert result.rule_id == "input_secret_request"


def test_user_input_allows_skill_documentation_that_mentions_credentials() -> None:
    skill_context = "\n".join(
        [
            "API keys are resolved from the system credential store.",
            "Never copy a real key into logs or UI text.",
            "After reload, read the active model before reporting success.",
        ]
    )

    assert check_user_input(skill_context) is None
