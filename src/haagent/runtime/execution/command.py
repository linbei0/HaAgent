"""
src/haagent/runtime/execution/command.py - 统一命令执行器

封装本地进程执行边界、输出摘要和 subprocess 结果。
"""

from __future__ import annotations

import locale
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from haagent.runtime.execution.cancellation import CancellationToken


CWD_GUIDANCE = 'cwd is relative to workspace_root; use "." or omit cwd for workspace root'
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 120.0
OUTPUT_EXCERPT_CHAR_LIMIT = 2400
REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_TOKEN = "[REDACTED_TOKEN]"
SECRET_TOKEN_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
KEY_VALUE_PATTERN = re.compile(
    r"\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|credential)\b\s*[:=]\s*\S{4,}",
    re.IGNORECASE,
)
SECRET_ENV_NAME_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|credential)",
    re.IGNORECASE,
)
PYTHON_UTF8_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


@dataclass(frozen=True)
class CommandResult:
    command: str
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_excerpt: str
    stderr_excerpt: str
    stdout_truncated: bool
    stderr_truncated: bool
    truncated: bool
    timeout: bool
    redacted: bool
    duration_seconds: float
    timeout_seconds: float

def run_command(
    command: str,
    cwd: Path,
    timeout_seconds: float,
    cancellation_token: CancellationToken | None = None,
) -> CommandResult:
    """运行 shell 命令，并用统一结构表达执行结果。"""
    popen_args, use_shell = resolve_shell_command(command)
    return run_process(
        command=command,
        popen_args=popen_args,
        shell=use_shell,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        cancellation_token=cancellation_token,
    )


def resolve_shell_command(
    command: str,
    *,
    os_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    bash_is_usable: Callable[[str], bool] | None = None,
) -> tuple[str | list[str], bool]:
    """选择平台合适的 shell argv，避免 Windows 默认 cmd 误跑 Unix 风格命令。"""
    resolved_os = os_name or os.name
    usable_bash = bash_is_usable or _bash_is_usable
    if resolved_os == "nt":
        bash = which("bash")
        if bash and not _is_wsl_bash_path(bash) and usable_bash(bash):
            return [bash, "-lc", command], False
        pwsh = which("pwsh")
        if pwsh:
            return [pwsh, "-NoLogo", "-NoProfile", "-Command", _powershell_command(command)], False
        powershell = which("powershell")
        if powershell:
            return [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-Command",
                _powershell_command(command, legacy=True),
            ], False
        return [which("cmd.exe") or "cmd.exe", "/d", "/s", "/c", command], False

    shell = which("bash") or which("sh") or os.environ.get("SHELL") or "/bin/sh"
    return [shell, "-lc", command], False


def _bash_is_usable(bash_path: str) -> bool:
    try:
        result = subprocess.run(
            [bash_path, "-lc", "exit 0"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _powershell_command(command: str, *, legacy: bool = False) -> str:
    utf8_setup = (
        "try { [Console]::InputEncoding = [System.Text.Encoding]::UTF8 } catch {}; "
        "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}; "
        "try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}; "
    )
    legacy_file_read = (
        "$PSDefaultParameterValues['Get-Content:Encoding'] = 'utf8'; "
        if legacy
        else ""
    )
    return (
        "& { "
        f"{utf8_setup}"
        f"{legacy_file_read}"
        "$ErrorActionPreference = 'Stop'; "
        "try { "
        f"{command}; "
        "if ($null -ne $global:LASTEXITCODE) { exit $global:LASTEXITCODE } "
        "} catch { Write-Error $_; exit 1 } "
        "}"
    )


def _is_wsl_bash_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return normalized.endswith("/windows/system32/bash.exe") or "/microsoft/windowsapps/bash.exe" in normalized


def run_process(
    *,
    command: str,
    popen_args: str | list[str],
    shell: bool,
    cwd: Path,
    timeout_seconds: float,
    cancellation_token: CancellationToken | None = None,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    """运行本地进程，支持超时和外部取消。"""
    started = time.perf_counter()
    process = subprocess.Popen(
        popen_args,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
    )
    while True:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            stdout_bytes, stderr_bytes = _stop_process(process)
            stdout, stderr = _decode_process_streams(stdout_bytes, stderr_bytes)
            output = build_output_summary(stdout, stderr)
            return CommandResult(
                command=command,
                status="cancelled",
                exit_code=None,
                stdout=output["stdout"],
                stderr=output["stderr"],
                stdout_excerpt=output["stdout_excerpt"],
                stderr_excerpt=output["stderr_excerpt"],
                stdout_truncated=output["stdout_truncated"],
                stderr_truncated=output["stderr_truncated"],
                truncated=output["truncated"],
                timeout=False,
                redacted=output["redacted"],
                duration_seconds=time.perf_counter() - started,
                timeout_seconds=timeout_seconds,
            )
        elapsed = time.perf_counter() - started
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            stdout_bytes, stderr_bytes = _stop_process(process)
            stdout, stderr = _decode_process_streams(stdout_bytes, stderr_bytes)
            output = build_output_summary(stdout, stderr)
            return CommandResult(
                command=command,
                status="timeout",
                exit_code=None,
                stdout=output["stdout"],
                stderr=output["stderr"],
                stdout_excerpt=output["stdout_excerpt"],
                stderr_excerpt=output["stderr_excerpt"],
                stdout_truncated=output["stdout_truncated"],
                stderr_truncated=output["stderr_truncated"],
                truncated=output["truncated"],
                timeout=True,
                redacted=output["redacted"],
                duration_seconds=time.perf_counter() - started,
                timeout_seconds=timeout_seconds,
            )
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=min(0.05, remaining))
            break
        except subprocess.TimeoutExpired:
            continue

    stdout, stderr = _decode_process_streams(stdout_bytes, stderr_bytes)
    output = build_output_summary(stdout, stderr)
    return CommandResult(
        command=command,
        status="success" if process.returncode == 0 else "failed",
        exit_code=process.returncode,
        stdout=output["stdout"],
        stderr=output["stderr"],
        stdout_excerpt=output["stdout_excerpt"],
        stderr_excerpt=output["stderr_excerpt"],
        stdout_truncated=output["stdout_truncated"],
        stderr_truncated=output["stderr_truncated"],
        truncated=output["truncated"],
        timeout=False,
        redacted=output["redacted"],
        duration_seconds=time.perf_counter() - started,
        timeout_seconds=timeout_seconds,
    )


def _stop_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()
    try:
        return process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        stdout, stderr = process.communicate()
        return stdout, stderr


def build_python_utf8_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    inherit: bool = True,
) -> dict[str, str]:
    """构造 Python 子进程环境，强制默认文件 IO 与标准流使用 UTF-8。"""
    environment = dict(os.environ) if inherit else {}
    if overrides:
        environment.update(overrides)
    # code_run 是确定性的 Python 执行边界，不能继承 Windows 本地代码页。
    environment.update(PYTHON_UTF8_ENV)
    return environment


def _decode_process_streams(stdout: bytes, stderr: bytes) -> tuple[str, str]:
    return _decode_process_output(stdout), _decode_process_output(stderr)


def _decode_process_output(output: bytes) -> str:
    if not output:
        return ""
    try:
        decoded = output.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return _normalize_process_newlines(decoded)

    fallback = locale.getpreferredencoding(False)
    if fallback.lower().replace("_", "-") not in {"utf-8", "utf8"}:
        try:
            decoded = output.decode(fallback)
        except (LookupError, UnicodeDecodeError):
            pass
        else:
            return _normalize_process_newlines(decoded)
    # 未知原生命令可能输出混合代码页；保留可见错误而不是让 reader thread 崩溃。
    return _normalize_process_newlines(output.decode("utf-8", errors="replace"))


def _normalize_process_newlines(output: str) -> str:
    """保持原 text=True 合同：跨平台统一换行为 LF。"""
    return output.replace("\r\n", "\n").replace("\r", "\n")


def normalize_timeout(value: Any) -> float | str:
    """校验执行 timeout，省略时使用默认值，超过上限直接拒绝。"""
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "timeout_seconds must be a number"
    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        return "timeout_seconds must be positive"
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        return f"timeout_seconds must be <= {int(MAX_TIMEOUT_SECONDS)}"
    return timeout_seconds


def build_output_summary(stdout: str, stderr: str) -> dict[str, Any]:
    """生成脱敏后的 stdout/stderr 摘要，避免工具结果暴露完整长输出。"""
    safe_stdout, stdout_redacted = redact_secret_like_text(stdout)
    safe_stderr, stderr_redacted = redact_secret_like_text(stderr)
    stdout_excerpt, stdout_truncated = _excerpt(safe_stdout)
    stderr_excerpt, stderr_truncated = _excerpt(safe_stderr)
    return {
        "stdout": safe_stdout,
        "stderr": safe_stderr,
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "truncated": stdout_truncated or stderr_truncated,
        "redacted": stdout_redacted or stderr_redacted,
    }


def redact_secret_like_text(text: str) -> tuple[str, bool]:
    """按 secret-like 模式和当前环境中的敏感变量值脱敏。"""
    redacted = KEY_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED_SECRET}", text)
    redacted = SECRET_TOKEN_PATTERN.sub(REDACTED_TOKEN, redacted)
    for value in _secret_environment_values():
        redacted = redacted.replace(value, REDACTED_SECRET)
    return redacted, redacted != text


def _excerpt(value: str) -> tuple[str, bool]:
    truncated = len(value) > OUTPUT_EXCERPT_CHAR_LIMIT
    return value[:OUTPUT_EXCERPT_CHAR_LIMIT], truncated


def _secret_environment_values() -> list[str]:
    values = [
        value
        for name, value in os.environ.items()
        if SECRET_ENV_NAME_PATTERN.search(name) and len(value) >= 4
    ]
    return sorted(set(values), key=len, reverse=True)
