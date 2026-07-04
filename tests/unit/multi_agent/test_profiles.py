"""
tests/unit/multi_agent/test_profiles.py - 多智能体 profile 加载测试

验证内置 agent profile 的稳定字段和显式失败行为。
"""

import pytest

import haagent.multi_agent.profiles as profiles_module
from haagent.multi_agent.profiles import (
    AgentProfile,
    get_agent_profile,
    load_builtin_agent_profiles,
    load_user_agent_profiles,
)


def test_load_builtin_agent_profiles_has_core_roles() -> None:
    profiles = load_builtin_agent_profiles()

    assert set(profiles) == {"explorer", "worker", "verification"}
    assert profiles["explorer"].subagent_type == "explorer"
    assert profiles["worker"].subagent_type == "worker"
    assert profiles["verification"].subagent_type == "verification"


def test_get_agent_profile_returns_builtin_profile() -> None:
    profile = get_agent_profile("explorer")

    assert isinstance(profile, AgentProfile)
    assert profile.name == "explorer"
    assert profile.allowed_tools == ["file_list", "file_search", "file_read", "skill_list", "skill_read"]


def test_get_agent_profile_unknown_name_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="unknown agent profile: missing"):
        get_agent_profile("missing")


def test_load_user_agent_profiles_reads_json_file(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "doc-editor.json").write_text(
        """
        {
          "name": "doc-editor",
          "description": "润色文档草稿。",
          "subagent_type": "worker",
          "system_prompt": "你是文档编辑助手。",
          "allowed_tools": ["file_read", "file_write"],
          "model_profile": "long-context",
          "max_turns": 8,
          "enable_web": false
        }
        """,
        encoding="utf-8",
    )

    profiles = load_user_agent_profiles(tmp_path)

    assert profiles["doc-editor"].model_profile == "long-context"
    assert profiles["doc-editor"].allowed_tools == ["file_read", "file_write"]
    assert profiles["doc-editor"].max_turns == 8
    assert profiles["doc-editor"].enable_web is False


def test_user_agent_profile_overrides_builtin_by_name(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "explorer.json").write_text(
        """
        {
          "name": "explorer",
          "description": "项目索引助手。",
          "subagent_type": "explorer",
          "system_prompt": "只读取项目结构。",
          "allowed_tools": ["file_list"]
        }
        """,
        encoding="utf-8",
    )

    profile = get_agent_profile("explorer", config_dir=tmp_path)

    assert profile.description == "项目索引助手。"
    assert profile.allowed_tools == ["file_list"]


def test_user_agent_profile_invalid_subagent_type_fails(tmp_path) -> None:
    profiles_dir = tmp_path / "agents"
    profiles_dir.mkdir()
    (profiles_dir / "bad.json").write_text(
        """
        {
          "name": "bad",
          "description": "bad",
          "subagent_type": "admin",
          "system_prompt": "bad"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid subagent_type"):
        load_user_agent_profiles(tmp_path)


def test_resolve_worker_profile_uses_user_config_dir_when_config_dir_omitted(tmp_path, monkeypatch) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "doc-editor.json").write_text(
        """
        {
          "name": "doc-editor",
          "description": "润色文档。",
          "subagent_type": "worker",
          "system_prompt": "你是文档编辑助手。",
          "model_profile": "writer-model",
          "allowed_tools": ["file_read", "file_write"],
          "max_turns": 6,
          "enable_web": false,
          "backend": "in_process",
          "worktree": false
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(profiles_module, "user_config_dir", lambda: tmp_path, raising=False)

    profile = profiles_module.resolve_worker_profile(
        "doc-editor",
        fallback_subagent_type="worker",
    )

    assert profile.name == "doc-editor"
    assert profile.model_profile == "writer-model"
    assert profile.allowed_tools == ["file_read", "file_write"]
    assert profile.max_turns == 6
    assert profile.enable_web is False


def test_resolve_worker_profile_rejects_declared_field_without_runtime_support(tmp_path) -> None:
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "bad.json").write_text(
        """
        {
          "name": "bad",
          "description": "bad",
          "subagent_type": "worker",
          "system_prompt": "bad",
          "backend": "remote"
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported backend"):
        profiles_module.resolve_worker_profile("bad", fallback_subagent_type="worker", config_dir=tmp_path)
