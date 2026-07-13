"""
tests/unit/runtime/test_safety_guard.py - 安全防护层测试

验证 SafetyGuard 只检测真正的异常，不干预正常的探索行为。
"""

from __future__ import annotations

from haagent.runtime.execution.safety_guard import SafetyGuard


def _success(tool_name: str, args: dict) -> dict:
    return {"status": "success", **args}


def _error(tool_name: str, args: dict, error_type: str = "tool_error") -> dict:
    return {"status": "error", "error": {"type": error_type, "message": "failed"}}


# --- 死循环检测 ---

def test_identical_calls_no_longer_abort() -> None:
    """相同参数循环终止已迁至 ProgressGuard，SafetyGuard 不再 abort。"""
    guard = SafetyGuard()
    for _ in range(5):
        violation = guard.check("file_read", {"path": "README.md"}, _success("file_read", {}))
        assert violation is None


def test_different_paths_not_flagged_as_loop() -> None:
    guard = SafetyGuard()
    paths = ["README.md", "setup.py", "docs/api.md"]
    for path in paths:
        violation = guard.check("file_read", {"path": path}, _success("file_read", {}))
        assert violation is None


def test_same_tool_different_args_not_flagged() -> None:
    guard = SafetyGuard()
    for i in range(5):
        violation = guard.check("file_read", {"path": f"file_{i}.py"}, _success("file_read", {}))
        assert violation is None


def test_repeated_reads_of_same_file_nonconsecutively_not_flagged() -> None:
    """非连续的重复读取不应该被判定为死循环。"""
    guard = SafetyGuard()
    guard.check("file_read", {"path": "README.md"}, _success("file_read", {}))
    guard.check("file_read", {"path": "other.py"}, _success("file_read", {}))  # 中断了连续性
    violation = guard.check("file_read", {"path": "README.md"}, _success("file_read", {}))
    assert violation is None


def test_ten_varied_file_reads_never_abort() -> None:
    """关键回归：读取多个不同文件不应触发任何保护。"""
    guard = SafetyGuard()
    files = [
        "README.md", "setup.py", "docs/api.md", "src/main.py",
        "tests/test_main.py", "config.yaml", "README.md",  # 重复但非连续
        "pyproject.toml", "src/utils.py", "docs/guide.md",
    ]
    for path in files:
        violation = guard.check("file_read", {"path": path}, _success("file_read", {}))
        assert violation is None


# --- 连续失败检测 ---

def test_three_consecutive_failures_warns() -> None:
    guard = SafetyGuard()
    for i in range(2):
        violation = guard.check("file_read", {"path": "x.py"}, _error("file_read", {}))
        assert violation is None or not violation.should_abort
    violation = guard.check("file_read", {"path": "x.py"}, _error("file_read", {}))
    assert violation is not None
    assert violation.type == "repeated_failure"
    assert violation.should_abort is False  # 警告，不强制中止


def test_success_resets_failure_count() -> None:
    guard = SafetyGuard()
    guard.check("file_read", {"path": "x.py"}, _error("file_read", {}))
    guard.check("file_read", {"path": "x.py"}, _error("file_read", {}))
    guard.check("file_read", {"path": "x.py"}, _success("file_read", {}))  # 重置
    violation = guard.check("file_read", {"path": "x.py"}, _error("file_read", {}))
    assert violation is None  # 计数已重置，不应该警告


# --- 原始 bug 场景回归测试 ---

def test_original_bug_scenario_four_reads_then_file_list() -> None:
    """
    原始 bug：'生成介绍文档' 任务读取 4 次文件后被强制终止。
    新 SafetyGuard 不应该干预这个行为。
    """
    guard = SafetyGuard()
    reads = [
        ("file_read", {"path": "docs/harness-requirements.md"}),
        ("file_read", {"path": "docs/unresolved-risks-and-roadmap.md"}),
        ("file_read", {"path": "docs/code-governance.md"}),
        ("file_list", {"max_depth": 1}),
    ]
    for tool_name, args in reads:
        violation = guard.check(tool_name, args, _success(tool_name, {}))
        assert violation is None, f"不应该在 {tool_name} 时触发保护"
