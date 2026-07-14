"""
tests/unit/prompts/test_prompt_commands.py - 显式提示词命令测试

验证内置提示词包查找和 /review、/debug、/verify 命令解析。
"""

from __future__ import annotations

from haagent.prompts.commands import parse_prompt_command
from haagent.prompts.packs import get_prompt_pack, iter_prompt_modes


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
    for mode in iter_prompt_modes():
        content = mode.pack.content

        assert "Workflow:" in content
        assert "Output:" in content
        assert "Evidence" in content
        assert mode.pack.hard_required is True
        assert len(content) <= mode.pack.max_chars


def test_prompt_modes_are_the_single_source_for_command_parsing() -> None:
    modes = {mode.command: mode for mode in iter_prompt_modes()}

    assert set(modes) == {"review", "debug", "verify"}
    for command, mode in modes.items():
        with_body = parse_prompt_command(f"/{command} custom request")
        empty_body = parse_prompt_command(f"/{command}")

        assert with_body.command == command
        assert with_body.prompt_pack_ids == [mode.pack.id]
        assert with_body.normalized_prompt == "custom request"
        assert empty_body.command == command
        assert empty_body.prompt_pack_ids == [mode.pack.id]
        assert empty_body.normalized_prompt == mode.default_goal


def test_unknown_slash_command_is_plain_prompt() -> None:
    result = parse_prompt_command("/unknown hi")

    assert result.command is None
    assert result.prompt_pack_ids == []
    assert result.normalized_prompt == "/unknown hi"
