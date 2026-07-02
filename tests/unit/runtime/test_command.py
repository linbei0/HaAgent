"""
tests/unit/runtime/test_command.py - CommandRunner 测试

验证统一命令执行器能表达成功、非零退出、超时和取消结果。
"""

import threading
import time
from pathlib import Path

from haagent.runtime.cancellation import CancellationToken
from haagent.runtime.command import run_command


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


def test_run_command_replaces_invalid_utf8_output(tmp_path: Path) -> None:
    result = run_command(
        "python -c \"import sys; sys.stdout.buffer.write(bytes([0xd5])); sys.stderr.buffer.write(bytes([0xff]))\"",
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert result.status == "success"
    assert result.exit_code == 0
    assert "\ufffd" in result.stdout
    assert "\ufffd" in result.stderr


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


def test_run_command_terminates_process_when_cancelled(tmp_path: Path) -> None:
    token = CancellationToken()

    def cancel_soon() -> None:
        time.sleep(0.05)
        token.cancel()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    result = run_command(
        "python -c \"import time; time.sleep(5)\"",
        cwd=tmp_path,
        timeout_seconds=10,
        cancellation_token=token,
    )
    thread.join(timeout=1)

    assert result.status == "cancelled"
    assert result.exit_code is None
    assert result.timeout is False
    assert result.duration_seconds < 5
