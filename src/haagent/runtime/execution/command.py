"""
src/haagent/runtime/execution/command.py - 统一命令执行器

封装本地进程执行边界、输出摘要和 subprocess 结果。
"""

from __future__ import annotations

import codecs
import locale
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from haagent.runtime.execution.cancellation import CancellationToken


CWD_GUIDANCE = 'cwd is relative to workspace_root; use "." or omit cwd for workspace root'
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 120.0
# 仅用于 UI/episode 的短摘要；模型输入由 ToolResultView 统一预算。
OUTPUT_UI_EXCERPT_CHAR_LIMIT = 2400
# stdout/stderr 共享此进程层内存预算；超出部分流式写入临时文件。
PROCESS_CAPTURE_MEMORY_LIMIT = 1 * 1024 * 1024
PROCESS_CAPTURE_CHUNK_BYTES = 64 * 1024
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
    stdout_original_chars: int = 0
    stderr_original_chars: int = 0
    stdout_original_bytes: int = 0
    stderr_original_bytes: int = 0
    stdout_artifact_path: str | None = None
    stderr_artifact_path: str | None = None


@dataclass(frozen=True)
class _CapturedStream:
    memory: bytes
    spill_path: Path | None
    total_bytes: int


@dataclass
class _CaptureBudget:
    limit_bytes: int = PROCESS_CAPTURE_MEMORY_LIMIT
    used_bytes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True)
class _MaterializedStream:
    preview: str
    artifact_path: str | None
    original_chars: int
    original_bytes: int
    redacted: bool


@dataclass(frozen=True)
class ShellContract:
    """一次 shell 执行实际采用的解释器及其模型可见契约。"""

    kind: str
    executable: str
    platform: str


def run_command(
    command: str,
    cwd: Path,
    timeout_seconds: float,
    cancellation_token: CancellationToken | None = None,
    shell_contract: ShellContract | None = None,
    output_artifact_root: Path | None = None,
    artifact_name: str = "shell",
) -> CommandResult:
    """运行 shell 命令，并用统一结构表达执行结果。"""
    contract = shell_contract or resolve_shell_contract()
    popen_args, use_shell = build_shell_command_argv(command, contract)
    return run_process(
        command=command,
        popen_args=popen_args,
        shell=use_shell,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        cancellation_token=cancellation_token,
        output_artifact_root=output_artifact_root,
        artifact_name=artifact_name,
    )


def resolve_shell_command(
    command: str,
    *,
    os_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[str | list[str], bool]:
    """兼容入口：按实际 Shell 契约构造执行 argv。"""
    return build_shell_command_argv(
        command,
        resolve_shell_contract(os_name=os_name, which=which),
    )


def resolve_shell_contract(
    *,
    os_name: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> ShellContract:
    """确定本机执行 shell；模型 schema 与执行器必须使用同一契约。"""
    resolved_os = os_name or os.name
    if resolved_os == "nt":
        pwsh = which("pwsh")
        if pwsh:
            return ShellContract("powershell", pwsh, "windows")
        powershell = which("powershell")
        if powershell:
            return ShellContract("powershell_legacy", powershell, "windows")
        return ShellContract("cmd", which("cmd.exe") or "cmd.exe", "windows")

    shell = which("bash") or which("sh") or os.environ.get("SHELL") or "/bin/sh"
    return ShellContract("posix", shell, "posix")


def build_shell_command_argv(
    command: str,
    contract: ShellContract,
) -> tuple[list[str], bool]:
    """按契约构造 argv；不猜测也不改写模型提供的命令文本。"""
    if contract.kind == "powershell":
        return [contract.executable, "-NoLogo", "-NoProfile", "-Command", _powershell_command(command)], False
    if contract.kind == "powershell_legacy":
        return [
            contract.executable,
            "-NoLogo",
            "-NoProfile",
            "-Command",
            _powershell_command(command, legacy=True),
        ], False
    if contract.kind == "cmd":
        return [contract.executable, "/d", "/s", "/c", command], False
    if contract.kind == "posix":
        return [contract.executable, "-lc", command], False
    raise ValueError(f"unsupported shell contract: {contract.kind}")


def describe_shell_contract(contract: ShellContract) -> str:
    """生成解释器无关的事实说明，不列举脆弱的单条命令写法。"""
    language = {
        "powershell": "PowerShell",
        "powershell_legacy": "Windows PowerShell",
        "cmd": "cmd.exe command language",
        "posix": "POSIX shell",
    }.get(contract.kind, contract.kind)
    return (
        f"Runtime shell contract: platform={contract.platform}; interpreter={language}; "
        f"executable={contract.executable}. The command is interpreted as {language}; use that language's "
        "native syntax. The tool supplies cwd separately. To use another command language, invoke its "
        "interpreter explicitly in command."
    )


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
        "}; exit 0"
    )


def run_process(
    *,
    command: str,
    popen_args: str | list[str],
    shell: bool,
    cwd: Path,
    timeout_seconds: float,
    cancellation_token: CancellationToken | None = None,
    env: Mapping[str, str] | None = None,
    output_artifact_root: Path | None = None,
    artifact_name: str = "command",
) -> CommandResult:
    """运行本地进程，支持超时、取消和有界 stdout/stderr 采集。"""
    started = time.perf_counter()
    process = subprocess.Popen(
        popen_args,
        shell=shell,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env) if env is not None else None,
    )
    captures: dict[str, _CapturedStream] = {}
    capture_budget = _CaptureBudget()
    threads = [
        threading.Thread(
            target=_capture_pipe,
            args=("stdout", process.stdout, captures, capture_budget),
            daemon=True,
        ),
        threading.Thread(
            target=_capture_pipe,
            args=("stderr", process.stderr, captures, capture_budget),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    status = "success"
    timeout = False
    while process.poll() is None:
        if cancellation_token is not None and cancellation_token.is_cancelled:
            status = "cancelled"
            _stop_process(process)
            break
        if time.perf_counter() - started >= timeout_seconds:
            status = "timeout"
            timeout = True
            _stop_process(process)
            break
        time.sleep(0.01)
    if process.poll() is None:
        _stop_process(process)
    for thread in threads:
        thread.join(timeout=3)

    stdout_materialized = _materialize_stream(
        captures.get("stdout", _CapturedStream(b"", None, 0)),
        stream_name="stdout",
        output_artifact_root=output_artifact_root,
        artifact_name=artifact_name,
    )
    stderr_materialized = _materialize_stream(
        captures.get("stderr", _CapturedStream(b"", None, 0)),
        stream_name="stderr",
        output_artifact_root=output_artifact_root,
        artifact_name=artifact_name,
    )
    output = build_output_summary(stdout_materialized.preview, stderr_materialized.preview)
    if status == "success" and process.returncode not in (0, None):
        status = "failed"
    return CommandResult(
        command=command,
        status=status,
        exit_code=process.returncode if status == "failed" else (None if status != "success" else process.returncode),
        stdout=output["stdout"],
        stderr=output["stderr"],
        stdout_excerpt=output["stdout_excerpt"],
        stderr_excerpt=output["stderr_excerpt"],
        stdout_truncated=output["stdout_truncated"],
        stderr_truncated=output["stderr_truncated"],
        truncated=bool(stdout_materialized.artifact_path or stderr_materialized.artifact_path),
        timeout=timeout,
        redacted=output["redacted"] or stdout_materialized.redacted or stderr_materialized.redacted,
        duration_seconds=time.perf_counter() - started,
        timeout_seconds=timeout_seconds,
        stdout_original_chars=stdout_materialized.original_chars,
        stderr_original_chars=stderr_materialized.original_chars,
        stdout_original_bytes=stdout_materialized.original_bytes,
        stderr_original_bytes=stderr_materialized.original_bytes,
        stdout_artifact_path=stdout_materialized.artifact_path,
        stderr_artifact_path=stderr_materialized.artifact_path,
    )


def _stop_process(process: subprocess.Popen[bytes]) -> None:
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
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)


def _capture_pipe(
    stream_name: str,
    stream: Any,
    captures: dict[str, _CapturedStream],
    capture_budget: _CaptureBudget,
) -> None:
    """在共享进程预算耗尽后把原始字节转存临时文件，避免管道阻塞和内存爆炸。"""
    buffer = bytearray()
    spill_path: Path | None = None
    spill_handle: Any | None = None
    total_bytes = 0
    try:
        while True:
            chunk = stream.read(PROCESS_CAPTURE_CHUNK_BYTES)
            if not chunk:
                break
            total_bytes += len(chunk)
            if spill_handle is None:
                with capture_budget.lock:
                    available = capture_budget.limit_bytes - capture_budget.used_bytes
                    if len(buffer) + len(chunk) <= available:
                        buffer.extend(chunk)
                        capture_budget.used_bytes += len(chunk)
                        continue
            if spill_handle is None:
                handle = tempfile.NamedTemporaryFile(prefix="haagent-process-output-", suffix=".bin", delete=False)
                spill_path = Path(handle.name)
                spill_handle = handle
                if buffer:
                    spill_handle.write(buffer)
                    buffer.clear()
            spill_handle.write(chunk)
    finally:
        if spill_handle is not None:
            spill_handle.flush()
            spill_handle.close()
        captures[stream_name] = _CapturedStream(bytes(buffer), spill_path, total_bytes)


def _materialize_stream(
    capture: _CapturedStream,
    *,
    stream_name: str,
    output_artifact_root: Path | None,
    artifact_name: str,
) -> _MaterializedStream:
    if capture.spill_path is None:
        text = _decode_process_output(capture.memory)
        safe_text, redacted = redact_secret_like_text(text)
        return _MaterializedStream(
            preview=safe_text,
            artifact_path=None,
            original_chars=len(safe_text),
            original_bytes=len(safe_text.encode("utf-8")),
            redacted=redacted,
        )

    root = output_artifact_root or Path(tempfile.gettempdir()) / "haagent-process-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", artifact_name).strip("._") or "command"
    target = root / f"{safe_name}-{stream_name}-{uuid.uuid4().hex[:8]}.txt"
    head_limit = 32 * 1024
    tail_limit = 32 * 1024
    head = ""
    tail = ""
    total_chars = 0
    total_bytes = 0
    redacted = False
    overlap = 256
    pending = ""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with capture.spill_path.open("rb") as source, target.open("wb") as destination:
        while chunk := source.read(PROCESS_CAPTURE_CHUNK_BYTES):
            decoded = _normalize_process_newlines(pending + decoder.decode(chunk, final=False))
            if len(decoded) <= overlap:
                pending = decoded
                continue
            body, pending = decoded[:-overlap], decoded[-overlap:]
            safe, changed = redact_secret_like_text(body)
            redacted = redacted or changed
            destination.write(safe.encode("utf-8"))
            total_chars += len(safe)
            total_bytes += len(safe.encode("utf-8"))
            if len(head) < head_limit:
                head += safe[: head_limit - len(head)]
            tail = (tail + safe)[-tail_limit:]
        final_text = _normalize_process_newlines(pending + decoder.decode(b"", final=True))
        safe, changed = redact_secret_like_text(final_text)
        redacted = redacted or changed
        destination.write(safe.encode("utf-8"))
        total_chars += len(safe)
        total_bytes += len(safe.encode("utf-8"))
        if len(head) < head_limit:
            head += safe[: head_limit - len(head)]
        tail = (tail + safe)[-tail_limit:]
    capture.spill_path.unlink(missing_ok=True)
    preview = head + tail if total_chars > len(head) + len(tail) else head
    return _MaterializedStream(
        preview=preview,
        artifact_path=str(target),
        original_chars=total_chars,
        original_bytes=total_bytes,
        redacted=redacted,
    )


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
    """生成脱敏后的完整 stdout/stderr，并单独提供 UI 短摘要。"""
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
        # 这是 UI excerpt 是否缩短，不代表模型视图是否截断。
        "truncated": False,
        "redacted": stdout_redacted or stderr_redacted,
        "stdout_original_chars": len(safe_stdout),
        "stderr_original_chars": len(safe_stderr),
        "stdout_original_bytes": len(safe_stdout.encode("utf-8")),
        "stderr_original_bytes": len(safe_stderr.encode("utf-8")),
    }


def redact_secret_like_text(text: str) -> tuple[str, bool]:
    """按 secret-like 模式和当前环境中的敏感变量值脱敏。"""
    redacted = KEY_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED_SECRET}", text)
    redacted = SECRET_TOKEN_PATTERN.sub(REDACTED_TOKEN, redacted)
    for value in _secret_environment_values():
        redacted = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
            REDACTED_SECRET,
            redacted,
        )
    return redacted, redacted != text


def _excerpt(value: str) -> tuple[str, bool]:
    truncated = len(value) > OUTPUT_UI_EXCERPT_CHAR_LIMIT
    return value[:OUTPUT_UI_EXCERPT_CHAR_LIMIT], truncated


def _secret_environment_values() -> list[str]:
    values = [
        value
        for name, value in os.environ.items()
        if SECRET_ENV_NAME_PATTERN.search(name) and len(value) >= 4
    ]
    return sorted(set(values), key=len, reverse=True)
