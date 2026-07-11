"""
tests/unit/test_pytest_environment.py - pytest 用户环境隔离回归

确保自动化测试不会读取开发者真实的 HaAgent 用户配置。
"""

from pathlib import Path

from haagent.mcp.settings import load_mcp_settings, user_mcp_settings_path


def test_pytest_uses_empty_isolated_user_home() -> None:
    assert user_mcp_settings_path().parent == Path.home() / ".haagent"
    assert load_mcp_settings().servers == {}
