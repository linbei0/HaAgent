"""
tests/conftest.py - pytest 测试分层入口

默认 pytest 只收集日常高信号快测；完整 Textual 接线、真实长流程和
inspect/eval/export 高级 harness 回归保留显式入口，避免每次本地回归都
支付低价值长尾成本。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


DEFAULT_EXCLUDED_DIRS = {
    "e2e",
    "extended",
    "tui",
}

# 单个慢用例(其所在文件整体不慢，只标到具体用例)。
SLOW_NODE_KEYWORDS = {
    "test_run_command_records_timeout",
    "test_verification_engine_records_timeout",
}


@pytest.fixture(scope="session", autouse=True)
def isolate_user_home(tmp_path_factory: pytest.TempPathFactory):
    """每个 xdist worker 使用独立 HOME，避免读取真实凭据和 MCP 配置。"""
    isolated_home = tmp_path_factory.mktemp("user-home")
    original = {name: os.environ.get(name) for name in ("HOME", "USERPROFILE")}
    os.environ["HOME"] = str(isolated_home)
    os.environ["USERPROFILE"] = str(isolated_home)
    try:
        yield isolated_home
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run tests/e2e long-flow tests when no explicit e2e path is selected",
    )
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="run tests/extended inspect/eval/export regressions by default",
    )
    parser.addoption(
        "--run-tui",
        action="store_true",
        default=False,
        help="run tests/tui Textual wiring tests when no explicit tui path is selected",
    )
    parser.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="run manual real-model dogfood tests; skipped by default",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """默认跳过重型分层目录；显式路径或开关可恢复收集。"""
    path = Path(collection_path)
    if not path.is_dir() or path.parent.name != "tests":
        return False

    dirname = path.name
    if dirname not in DEFAULT_EXCLUDED_DIRS:
        return False

    if dirname == "e2e" and config.getoption("--run-e2e"):
        return False
    if dirname == "extended" and config.getoption("--run-extended"):
        return False
    if dirname == "tui" and config.getoption("--run-tui"):
        return False

    root_path = Path(str(config.rootpath))
    explicit_targets = [
        (target if target.is_absolute() else root_path / target).resolve()
        for target in (Path(arg) for arg in config.args)
    ]
    if any(target == path or path in target.parents for target in explicit_targets):
        return False
    return True


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """为少量留在默认树里的慢用例打 slow 标，便于串行诊断时过滤。"""
    for item in items:
        if any(keyword in item.nodeid for keyword in SLOW_NODE_KEYWORDS):
            item.add_marker(pytest.mark.slow)
