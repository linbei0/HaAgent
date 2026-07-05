"""
tests/unit/prompts/test_prompt_commands.py - 显式提示词命令测试

验证内置提示词包查找和 /review、/debug、/verify 命令解析。
"""

from __future__ import annotations

from haagent.prompts.commands import parse_prompt_command
from haagent.prompts.packs import get_prompt_pack


def test_review_command_selects_code_review_pack() -> None:
    result = parse_prompt_command("/review 看看这个改动")

    assert result.command == "review"
    assert result.prompt_pack_ids == ["code-review"]
    assert result.normalized_prompt == "看看这个改动"


def test_plain_prompt_selects_no_pack() -> None:
    result = parse_prompt_command("看看这个改动")

    assert result.command is None
    assert result.prompt_pack_ids == []
    assert result.normalized_prompt == "看看这个改动"


def test_prompt_pack_lookup_returns_atomic_builtin_pack() -> None:
    pack = get_prompt_pack("code-review")

    assert pack.id == "code-review"
    assert pack.hard_required is True
    assert 0 < len(pack.content) <= pack.max_chars


def test_builtin_prompt_packs_define_workflow_and_output_contracts() -> None:
    for pack_id in ["code-review", "debugging", "verification"]:
        content = get_prompt_pack(pack_id).content

        assert "Workflow:" in content
        assert "Output:" in content
        assert "Evidence" in content
        assert len(content) <= get_prompt_pack(pack_id).max_chars


def test_debug_command_uses_default_goal_when_body_is_empty() -> None:
    result = parse_prompt_command("/debug")

    assert result.command == "debug"
    assert result.prompt_pack_ids == ["debugging"]
    assert result.normalized_prompt == "Debug the described failure."


def test_verify_command_uses_default_goal_when_body_is_empty() -> None:
    result = parse_prompt_command("/verify")

    assert result.command == "verify"
    assert result.prompt_pack_ids == ["verification"]
    assert result.normalized_prompt == "Verify the current result with concrete evidence."


def test_unknown_slash_command_is_plain_prompt() -> None:
    result = parse_prompt_command("/unknown hi")

    assert result.command is None
    assert result.prompt_pack_ids == []
    assert result.normalized_prompt == "/unknown hi"
