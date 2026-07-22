"""
tests/unit/tools/test_job_tools.py - job_start/status/logs/cancel 工具测试

验证工具参数、路径边界、短返回与结构化结果。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from haagent.runtime.execution.jobs import JobManager
from haagent.tools.jobs import job_cancel, job_logs, job_start, job_status


def test_job_start_rejects_invalid_command(tmp_path: Path) -> None:
    manager = JobManager(jobs_root=tmp_path / "jobs")
    result = job_start(
        {"command": ""},
        workspace_root=tmp_path,
        job_manager=manager,
    )
    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"


def test_job_tools_roundtrip(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; print(\'hello-job\'); time.sleep(0.4)"'
    else:
        command = f"{sys.executable} -c 'import time; print(\"hello-job\"); time.sleep(0.4)'"

    started = job_start(
        {"command": command, "timeout_seconds": 30},
        workspace_root=workspace,
        job_manager=manager,
    )
    assert started["status"] == "success"
    assert started["job_status"] == "running"
    job_id = started["job_id"]
    assert isinstance(job_id, str) and job_id

    final_status = job_status({"job_id": job_id}, job_manager=manager)
    assert final_status["status"] == "success"
    assert final_status["job_status"] == "finished"
    assert final_status["exit_code"] == 0
    assert final_status["waited_seconds"] > 0
    assert "hello-job" in final_status["stdout_excerpt"]

    logs = job_logs({"job_id": job_id, "stream": "stdout"}, job_manager=manager)
    assert logs["status"] == "success"
    assert "hello-job" in logs["stdout"]


def test_job_status_allows_explicit_immediate_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    started = job_start(
        {"command": command, "timeout_seconds": 60},
        workspace_root=workspace,
        job_manager=manager,
    )
    result = job_status(
        {"job_id": started["job_id"], "wait_seconds": 0},
        job_manager=manager,
    )

    assert result["status"] == "success"
    assert result["job_status"] == "running"
    assert result["waited_seconds"] < 0.5
    job_cancel({"job_id": started["job_id"]}, job_manager=manager)


def test_job_cancel_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = JobManager(jobs_root=tmp_path / "jobs")

    if os.name == "nt":
        command = f'{sys.executable} -c "import time; time.sleep(30)"'
    else:
        command = f"{sys.executable} -c 'import time; time.sleep(30)'"

    started = job_start(
        {"command": command, "timeout_seconds": 60},
        workspace_root=workspace,
        job_manager=manager,
    )
    job_id = started["job_id"]
    cancelled = job_cancel({"job_id": job_id}, job_manager=manager)
    assert cancelled["status"] == "success"
    assert cancelled["job_status"] == "cancelled"


def test_job_status_unknown(tmp_path: Path) -> None:
    manager = JobManager(jobs_root=tmp_path / "jobs")
    result = job_status({"job_id": "nope"}, job_manager=manager)
    assert result["status"] == "error"
    assert result["error"]["type"] == "job_not_found"
