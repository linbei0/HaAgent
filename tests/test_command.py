"""
tests/test_command.py - CommandRunner 测试

验证统一命令执行器能表达成功、非零退出和超时结果。
"""

from pathlib import Path

from agentfoundry.runtime.command import run_command


def test_run_command_records_success(tmp_path: Path) -> None:
    result = run_command("python -c \"print('ok')\"", cwd=tmp_path, timeout_seconds=5)

    assert result.command == "python -c \"print('ok')\""
    assert result.status == "success"
    assert result.exit_code == 0
    assert "ok" in result.stdout
    assert result.stderr == ""
    assert result.duration_seconds >= 0
    assert result.timeout_seconds == 5


def test_run_command_records_non_zero_exit(tmp_path: Path) -> None:
    result = run_command("python -c \"import sys; sys.exit(9)\"", cwd=tmp_path, timeout_seconds=5)

    assert result.status == "failed"
    assert result.exit_code == 9
    assert result.timeout_seconds == 5


def test_run_command_records_timeout(tmp_path: Path) -> None:
    result = run_command(
        "python -c \"import time; time.sleep(1)\"",
        cwd=tmp_path,
        timeout_seconds=0.01,
    )

    assert result.status == "timeout"
    assert result.exit_code is None
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.duration_seconds >= 0
    assert result.timeout_seconds == 0.01
