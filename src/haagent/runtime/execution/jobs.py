"""
src/haagent/runtime/execution/jobs.py - 后台命令 Job 运行时

管理长命令的启动、状态、日志与取消；日志与元数据落在用户级 jobs 目录，
不污染 workspace。单次 shell 仍保持短超时，长任务走本模块。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from haagent.runtime.execution.command import (
    build_output_summary,
    build_shell_command_argv,
    resolve_shell_contract,
    redact_secret_like_text,
)

DEFAULT_JOB_TIMEOUT_SECONDS = 3600.0
MAX_JOB_TIMEOUT_SECONDS = 7200.0
DEFAULT_JOB_WAIT_SECONDS = 30.0
MAX_JOB_WAIT_SECONDS = 60.0
JOB_STATUS = Literal["running", "finished", "failed", "timeout", "cancelled"]
_TERMINAL = frozenset({"finished", "failed", "timeout", "cancelled"})


@dataclass
class JobRecord:
    job_id: str
    command: str
    cwd: str
    workspace_root: str
    status: str
    created_at: float
    updated_at: float
    timeout_seconds: float
    pid: int | None = None
    exit_code: int | None = None
    timeout: bool = False
    redacted: bool = False
    finished_at: float | None = None
    error_message: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "timeout_seconds": self.timeout_seconds,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "timeout": self.timeout,
            "redacted": self.redacted,
            "error_message": self.error_message,
            "duration_seconds": self.duration_seconds,
        }

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.created_at)


def normalize_job_timeout(value: Any) -> float | str:
    """校验后台 job timeout；默认 1 小时，上限 2 小时。"""
    if value is None:
        return DEFAULT_JOB_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "timeout_seconds must be a number"
    timeout_seconds = float(value)
    if timeout_seconds <= 0:
        return "timeout_seconds must be positive"
    if timeout_seconds > MAX_JOB_TIMEOUT_SECONDS:
        return f"timeout_seconds must be <= {int(MAX_JOB_TIMEOUT_SECONDS)}"
    return timeout_seconds


def normalize_job_wait(value: Any) -> float | str:
    """校验状态查询等待时间；默认等待，显式 0 保留即时快照。"""
    if value is None:
        return DEFAULT_JOB_WAIT_SECONDS
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "wait_seconds must be a number"
    wait_seconds = float(value)
    if wait_seconds < 0:
        return "wait_seconds must be non-negative"
    if wait_seconds > MAX_JOB_WAIT_SECONDS:
        return f"wait_seconds must be <= {int(MAX_JOB_WAIT_SECONDS)}"
    return wait_seconds


def default_jobs_root() -> Path:
    from haagent.models.config.connections import user_config_dir

    return user_config_dir() / "jobs"


class JobManager:
    """进程内后台 job 表 + 磁盘元数据/日志。"""

    def __init__(self, jobs_root: Path | None = None) -> None:
        self._jobs_root = (jobs_root or default_jobs_root()).resolve()
        self._jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._records: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._watchers: dict[str, threading.Thread] = {}

    @property
    def jobs_root(self) -> Path:
        return self._jobs_root

    def start(
        self,
        *,
        command: str,
        cwd: Path,
        workspace_root: Path,
        timeout_seconds: float,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._jobs_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        now = time.time()
        record = JobRecord(
            job_id=job_id,
            command=command,
            cwd=str(cwd.resolve()),
            workspace_root=str(workspace_root.resolve()),
            status="running",
            created_at=now,
            updated_at=now,
            timeout_seconds=float(timeout_seconds),
        )
        contract = resolve_shell_contract()
        popen_args, use_shell = build_shell_command_argv(command, contract)
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        try:
            process = subprocess.Popen(
                popen_args,
                shell=use_shell,
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
            )
        except Exception as error:
            stdout_file.close()
            stderr_file.close()
            record.status = "failed"
            record.error_message = str(error)
            record.finished_at = time.time()
            record.updated_at = record.finished_at
            self._persist(record)
            with self._condition:
                self._records[job_id] = record
                self._condition.notify_all()
            return record

        record.pid = process.pid
        self._persist(record)
        with self._lock:
            self._records[job_id] = record
            self._processes[job_id] = process

        # 文件句柄交给 watcher 关闭，避免父进程提前 close 打断子进程写日志。
        watcher = threading.Thread(
            target=self._watch_process,
            args=(job_id, process, stdout_file, stderr_file),
            name=f"haagent-job-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._watchers[job_id] = watcher
        watcher.start()
        return record

    def status(self, job_id: str, *, wait_seconds: float = 0.0) -> JobRecord | None:
        """返回状态；运行中任务可等待终态通知，避免调用方高频轮询。"""
        wait_seconds = max(0.0, float(wait_seconds))
        deadline = time.monotonic() + wait_seconds
        with self._condition:
            record = self._records.get(job_id)
            while record is not None and record.status not in _TERMINAL and wait_seconds > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                # watcher 在完成、超时或取消时通知；等待发生在工具层，不消耗模型轮次。
                self._condition.wait(timeout=remaining)
                record = self._records.get(job_id)
            if record is not None:
                return self._copy_record(record)
        return self._load_record(job_id)

    def logs(
        self,
        job_id: str,
        *,
        stream: str = "both",
        max_chars: int = 4000,
        offset: int = 0,
    ) -> dict[str, Any] | None:
        job_dir = self._jobs_root / job_id
        if not job_dir.is_dir():
            return None
        if stream not in {"stdout", "stderr", "both"}:
            stream = "both"
        max_chars = max(200, min(int(max_chars), 20000))
        offset = max(0, int(offset))
        stdout_text = ""
        stderr_text = ""
        if stream in {"stdout", "both"}:
            stdout_text = self._read_log_tail(job_dir / "stdout.log", max_chars=max_chars, offset=offset)
        if stream in {"stderr", "both"}:
            stderr_text = self._read_log_tail(job_dir / "stderr.log", max_chars=max_chars, offset=offset)
        safe_stdout, stdout_redacted = redact_secret_like_text(stdout_text)
        safe_stderr, stderr_redacted = redact_secret_like_text(stderr_text)
        summary = build_output_summary(safe_stdout, safe_stderr)
        record = self.status(job_id)
        return {
            "job_id": job_id,
            "stream": stream,
            "stdout": safe_stdout,
            "stderr": safe_stderr,
            "stdout_excerpt": summary["stdout_excerpt"],
            "stderr_excerpt": summary["stderr_excerpt"],
            "truncated": summary["truncated"],
            "redacted": stdout_redacted or stderr_redacted or summary["redacted"],
            "job_status": record.status if record is not None else "unknown",
            "exit_code": record.exit_code if record is not None else None,
        }

    def cancel(self, job_id: str) -> JobRecord | None:
        with self._lock:
            process = self._processes.get(job_id)
            record = self._records.get(job_id)
        if record is None:
            record = self._load_record(job_id)
        if record is None:
            return None
        if record.status in _TERMINAL:
            return self._copy_record(record)
        now = time.time()
        with self._condition:
            live = self._records.get(job_id)
            if live is None:
                live = record
            # 先提交取消终态再杀进程，避免 watcher 抢先把人为终止记录成 failed。
            if live.status not in _TERMINAL:
                live.status = "cancelled"
                live.updated_at = now
                live.finished_at = now
                self._persist(live)
                self._records[job_id] = live
                self._condition.notify_all()
            result = self._copy_record(live)
        if process is not None and process.poll() is None:
            _stop_process_tree(process)
        return result

    def _watch_process(
        self,
        job_id: str,
        process: subprocess.Popen[bytes],
        stdout_file: Any,
        stderr_file: Any,
    ) -> None:
        try:
            while True:
                with self._lock:
                    record = self._records.get(job_id)
                if record is None:
                    break
                if record.status == "cancelled":
                    if process.poll() is None:
                        _stop_process_tree(process)
                    break
                elapsed = time.time() - record.created_at
                if elapsed >= record.timeout_seconds:
                    if process.poll() is None:
                        _stop_process_tree(process)
                    self._finalize(job_id, status="timeout", exit_code=None, timeout=True)
                    break
                code = process.poll()
                if code is not None:
                    status = "finished" if code == 0 else "failed"
                    self._finalize(job_id, status=status, exit_code=code, timeout=False)
                    break
                time.sleep(0.05)
        finally:
            try:
                stdout_file.close()
            except Exception:
                pass
            try:
                stderr_file.close()
            except Exception:
                pass
            with self._lock:
                self._processes.pop(job_id, None)
                self._watchers.pop(job_id, None)

    def _finalize(
        self,
        job_id: str,
        *,
        status: str,
        exit_code: int | None,
        timeout: bool,
    ) -> None:
        now = time.time()
        with self._condition:
            record = self._records.get(job_id)
            if record is None:
                return
            # cancel 优先于自然结束/超时覆盖。
            if record.status == "cancelled":
                return
            if record.status in _TERMINAL:
                return
            record.status = status
            record.exit_code = exit_code
            record.timeout = timeout
            record.updated_at = now
            record.finished_at = now
            self._persist(record)
            self._condition.notify_all()

    def _persist(self, record: JobRecord) -> None:
        job_dir = self._jobs_root / record.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / "meta.json"
        payload = asdict(record)
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load_record(self, job_id: str) -> JobRecord | None:
        meta_path = self._jobs_root / job_id / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            record = JobRecord(
                job_id=str(payload["job_id"]),
                command=str(payload["command"]),
                cwd=str(payload["cwd"]),
                workspace_root=str(payload.get("workspace_root", "")),
                status=str(payload["status"]),
                created_at=float(payload["created_at"]),
                updated_at=float(payload["updated_at"]),
                timeout_seconds=float(payload["timeout_seconds"]),
                pid=payload.get("pid"),
                exit_code=payload.get("exit_code"),
                timeout=bool(payload.get("timeout", False)),
                redacted=bool(payload.get("redacted", False)),
                finished_at=payload.get("finished_at"),
                error_message=payload.get("error_message"),
            )
        except (KeyError, TypeError, ValueError):
            return None
        with self._lock:
            self._records.setdefault(job_id, record)
        return self._copy_record(record)

    @staticmethod
    def _copy_record(record: JobRecord) -> JobRecord:
        return JobRecord(
            job_id=record.job_id,
            command=record.command,
            cwd=record.cwd,
            workspace_root=record.workspace_root,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            timeout_seconds=record.timeout_seconds,
            pid=record.pid,
            exit_code=record.exit_code,
            timeout=record.timeout,
            redacted=record.redacted,
            finished_at=record.finished_at,
            error_message=record.error_message,
        )

    @staticmethod
    def _read_log_tail(path: Path, *, max_chars: int, offset: int) -> str:
        if not path.is_file():
            return ""
        try:
            raw = path.read_bytes()
        except OSError:
            return ""
        text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
        if offset > 0:
            text = text[offset:]
        if len(text) > max_chars:
            return text[-max_chars:]
        return text


def _stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
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
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


_DEFAULT_JOB_MANAGER: JobManager | None = None
_DEFAULT_JOB_MANAGER_LOCK = threading.Lock()


def default_job_manager() -> JobManager:
    """进程级默认 JobManager，供工具 handler 共享。"""
    global _DEFAULT_JOB_MANAGER
    with _DEFAULT_JOB_MANAGER_LOCK:
        if _DEFAULT_JOB_MANAGER is None:
            _DEFAULT_JOB_MANAGER = JobManager()
        return _DEFAULT_JOB_MANAGER


def reset_default_job_manager_for_tests() -> None:
    """测试专用：清空默认 JobManager。"""
    global _DEFAULT_JOB_MANAGER
    with _DEFAULT_JOB_MANAGER_LOCK:
        _DEFAULT_JOB_MANAGER = None
