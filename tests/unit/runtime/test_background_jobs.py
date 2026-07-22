"""
tests/unit/runtime/test_background_jobs.py - 后台 job 运行时单元测试

覆盖启动、状态轮询、日志读取、取消与超时，以及不污染 workspace。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

import haagent.runtime.execution.jobs as jobs_runtime
from haagent.runtime.execution.jobs import (
    DEFAULT_JOB_TIMEOUT_SECONDS,
    DEFAULT_JOB_WAIT_SECONDS,
    MAX_JOB_TIMEOUT_SECONDS,
    MAX_JOB_WAIT_SECONDS,
    JobManager,
    JobRecord,
    normalize_job_timeout,
    normalize_job_wait,
)


def test_normalize_job_timeout_defaults_and_bounds() -> None:
    assert normalize_job_timeout(None) == DEFAULT_JOB_TIMEOUT_SECONDS
    assert normalize_job_timeout(120) == 120.0
    assert isinstance(normalize_job_timeout(0), str)
    assert isinstance(normalize_job_timeout(MAX_JOB_TIMEOUT_SECONDS + 1), str)


def test_normalize_job_wait_defaults_and_bounds() -> None:
    assert normalize_job_wait(None) == DEFAULT_JOB_WAIT_SECONDS
    assert normalize_job_wait(0) == 0.0
    assert normalize_job_wait(5) == 5.0
    assert isinstance(normalize_job_wait(-1), str)
    assert isinstance(normalize_job_wait(MAX_JOB_WAIT_SECONDS + 1), str)


def test_job_status_waits_for_terminal_state_without_polling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(0.3)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(0.3)'"

    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=30,
    )
    started = time.perf_counter()
    final = manager.status(record.job_id, wait_seconds=5)
    elapsed = time.perf_counter() - started

    assert final is not None
    assert final.status == "finished"
    assert final.exit_code == 0
    assert elapsed >= 0.15
    assert elapsed < 5


def test_job_start_returns_immediately_and_does_not_pollute_workspace(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=jobs_root)

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(2); print(\'done-bg\')"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(2); print(\"done-bg\")'"

    started = time.perf_counter()
    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=30,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert record.job_id
    assert record.status == "running"
    assert not any(workspace.iterdir())
    assert (jobs_root / record.job_id / "meta.json").is_file()

    deadline = time.time() + 10
    while time.time() < deadline:
        status = manager.status(record.job_id)
        if status is not None and status.status in {"finished", "failed", "timeout", "cancelled"}:
            break
        time.sleep(0.05)
    final = manager.status(record.job_id)
    assert final is not None
    assert final.status == "finished"
    assert final.exit_code == 0
    logs = manager.logs(record.job_id, stream="stdout", max_chars=2000)
    assert "done-bg" in logs["stdout"]


def test_job_cancel_stops_running_process(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=jobs_root)

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=60,
    )
    cancelled = manager.cancel(record.job_id)
    assert cancelled is not None
    assert cancelled.status == "cancelled"

    deadline = time.time() + 5
    while time.time() < deadline:
        status = manager.status(record.job_id)
        if status is not None and status.status == "cancelled":
            break
        time.sleep(0.05)
    assert manager.status(record.job_id).status == "cancelled"


def test_job_cancel_commits_cancelled_before_stopping_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=60,
    )
    observed_statuses: list[str] = []
    original_stop = jobs_runtime._stop_process_tree

    def observe_stop(process) -> None:
        current = manager.status(record.job_id)
        assert current is not None
        observed_statuses.append(current.status)
        original_stop(process)

    monkeypatch.setattr(jobs_runtime, "_stop_process_tree", observe_stop)
    cancelled = manager.cancel(record.job_id)

    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert observed_statuses
    assert set(observed_statuses) == {"cancelled"}


def test_job_cancel_wakes_status_waiter(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=60,
    )
    result: list[JobRecord | None] = []
    waiter = threading.Thread(
        target=lambda: result.append(manager.status(record.job_id, wait_seconds=5)),
    )
    waiter.start()
    time.sleep(0.1)
    manager.cancel(record.job_id)
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result
    assert result[0] is not None
    assert result[0].status == "cancelled"


def test_job_timeout_marks_timeout(tmp_path: Path) -> None:
    jobs_root = tmp_path / "jobs"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = JobManager(jobs_root=jobs_root)

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    record = manager.start(
        command=command,
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=0.3,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        status = manager.status(record.job_id)
        if status is not None and status.status == "timeout":
            break
        time.sleep(0.05)
    final = manager.status(record.job_id)
    assert final is not None
    assert final.status == "timeout"
    assert final.timeout is True


def test_unknown_job_status_and_logs(tmp_path: Path) -> None:
    manager = JobManager(jobs_root=tmp_path / "jobs")
    assert manager.status("missing") is None
    assert manager.logs("missing") is None
    assert manager.cancel("missing") is None
