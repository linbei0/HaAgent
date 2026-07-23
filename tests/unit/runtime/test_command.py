"""
tests/unit/runtime/test_command.py - CommandRunner 测试

验证统一命令执行器能表达成功、非零退出、超时和取消结果。
"""

import os
import shutil
import threading
import time
from pathlib import Path

import pytest

import haagent.runtime.execution.command as command_module
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.command import (
    ShellContract,
    _powershell_command,
    build_shell_command_argv,
    build_python_utf8_environment,
    describe_shell_contract,
    redact_secret_like_text,
    resolve_shell_command,
    resolve_shell_contract,
    run_command,
    run_process,
)


def test_secret_environment_value_does_not_redact_substrings_of_regular_output(monkeypatch) -> None:
    monkeypatch.setenv("HAAGENT_TEST_SECRET_TOKEN", "root")

    redacted, changed = redact_secret_like_text("C:/tmp/workspace_root0")

    assert redacted == "C:/tmp/workspace_root0"
    assert changed is False


def test_shell_contract_selects_powershell_on_windows_and_drives_argv() -> None:
    def fake_which(name: str) -> str | None:
        return {"pwsh": "C:/Program Files/PowerShell/7/pwsh.exe"}.get(name)

    contract = resolve_shell_contract(os_name="nt", which=fake_which)
    argv, use_shell = build_shell_command_argv("Get-ChildItem", contract)

    assert contract == ShellContract("powershell", "C:/Program Files/PowerShell/7/pwsh.exe", "windows")
    assert argv[:4] == [contract.executable, "-NoLogo", "-NoProfile", "-Command"]
    assert use_shell is False


def test_shell_contract_description_is_language_level_not_command_examples() -> None:
    contract = ShellContract("powershell", "C:/Program Files/PowerShell/7/pwsh.exe", "windows")

    description = describe_shell_contract(contract)

    assert "interpreter=PowerShell" in description
    assert "native syntax" in description
    assert "another command language" in description
    assert "USERPROFILE" not in description


def test_resolve_shell_command_prefers_powershell_on_windows_even_when_bash_exists() -> None:
    def fake_which(name: str) -> str | None:
        return {
            "bash": "C:/Program Files/Git/bin/bash.exe",
            "pwsh": "C:/Program Files/PowerShell/7/pwsh.exe",
        }.get(name)

    argv, use_shell = resolve_shell_command(
        "pwd && ls -la",
        os_name="nt",
        which=fake_which,
    )

    assert argv[:4] == [
        "C:/Program Files/PowerShell/7/pwsh.exe",
        "-NoLogo",
        "-NoProfile",
        "-Command",
    ]
    assert use_shell is False


def test_python_utf8_environment_overrides_windows_locale_settings() -> None:
    environment = build_python_utf8_environment(
        {"PYTHONUTF8": "0", "PYTHONIOENCODING": "gbk", "CUSTOM_ENV": "1"},
        inherit=False,
    )

    assert environment == {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUSTOM_ENV": "1",
    }


def test_resolve_shell_command_falls_back_to_powershell_on_windows() -> None:
    def fake_which(name: str) -> str | None:
        return {"pwsh": "C:/Program Files/PowerShell/7/pwsh.exe"}.get(name)

    argv, use_shell = resolve_shell_command(
        "Get-ChildItem",
        os_name="nt",
        which=fake_which,
    )

    assert argv[:4] == [
        "C:/Program Files/PowerShell/7/pwsh.exe",
        "-NoLogo",
        "-NoProfile",
        "-Command",
    ]
    assert "Get-ChildItem" in argv[4]
    assert "$global:LASTEXITCODE" in argv[4]
    assert use_shell is False


def test_resolve_shell_command_configures_legacy_powershell_utf8_reads() -> None:
    def fake_which(name: str) -> str | None:
        return {"powershell": "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"}.get(name)

    argv, use_shell = resolve_shell_command(
        "type README.md",
        os_name="nt",
        which=fake_which,
    )

    assert argv[0].endswith("powershell.exe")
    assert "$PSDefaultParameterValues['Get-Content:Encoding'] = 'utf8'" in argv[4]
    assert use_shell is False


def test_resolve_shell_command_skips_wsl_bash_on_windows() -> None:
    def fake_which(name: str) -> str | None:
        return {
            "bash": "C:/Windows/System32/bash.exe",
            "pwsh": "C:/Program Files/PowerShell/7/pwsh.exe",
        }.get(name)

    argv, use_shell = resolve_shell_command("python -c \"print('ok')\"", os_name="nt", which=fake_which)

    assert argv[0] == "C:/Program Files/PowerShell/7/pwsh.exe"
    assert use_shell is False


def test_powershell_command_converts_cmdlet_errors_to_failed_exit() -> None:
    wrapped = _powershell_command("ls -la")

    assert "[Console]::InputEncoding" in wrapped
    assert "[Console]::OutputEncoding" in wrapped
    assert "$OutputEncoding" in wrapped
    assert "$ErrorActionPreference = 'Stop'" in wrapped
    assert "catch" in wrapped
    assert "exit 1" in wrapped
    assert "}; exit 0" in wrapped


@pytest.mark.skipif(os.name != "nt" or shutil.which("powershell.exe") is None, reason="需要 Windows PowerShell 5.1")
def test_legacy_powershell_reads_bomless_utf8_file(tmp_path: Path) -> None:
    source = tmp_path / "utf8-source.txt"
    source.write_text("压缩预算 € →\n", encoding="utf-8")
    powershell = shutil.which("powershell.exe")
    assert powershell is not None
    escaped_path = str(source).replace("'", "''")
    command = f"type '{escaped_path}'"

    result = run_process(
        command=command,
        popen_args=[
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-Command",
            _powershell_command(command, legacy=True),
        ],
        shell=False,
        cwd=tmp_path,
        # Windows PowerShell 5.1 在并行 CI 的进程启动与代码页初始化可能超过 5 秒。
        timeout_seconds=15,
    )

    assert result.status == "success"
    assert result.stdout.splitlines() == ["压缩预算 € →"]


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
        "python -c \"import sys; sys.stdout.buffer.write(bytes([0x81])); sys.stderr.buffer.write(bytes([0x8d]))\"",
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert result.status == "success"
    assert result.exit_code == 0
    assert "\ufffd" in result.stdout
    assert "\ufffd" in result.stderr


def test_run_command_decodes_windows_locale_output_after_utf8_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(command_module.locale, "getpreferredencoding", lambda _setlocale=False: "gbk")
    result = run_command(
        "python -c \"import sys; sys.stdout.buffer.write(bytes([0xd6, 0xd0, 0xce, 0xc4]))\"",
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert result.status == "success"
    assert result.stdout == "中文"


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
