"""
tests/test_verification_engine.py - VerificationEngine 验证命令测试

验证 verification_commands 会执行、落盘，并把失败显式暴露给 orchestrator。
"""

import json
from pathlib import Path

from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.verification.engine import VerificationEngine


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Verify commands
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    return EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)


def test_verification_engine_records_each_command(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)

    result = engine.run(["python -c \"print('verified')\""])

    assert result.status == "success"
    record = json.loads((writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    assert record["command"] == "python -c \"print('verified')\""
    assert record["status"] == "success"
    assert record["exit_code"] == 0
    assert record["timeout"] is False
    assert "verified" in record["stdout"]
    assert record["stdout_excerpt"] == "verified\n"
    assert record["stderr"] == ""
    assert record["stderr_excerpt"] == ""
    assert record["duration_seconds"] >= 0
    assert record["timeout_seconds"] == 60


def test_verification_engine_reports_failed_command(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)

    result = engine.run(["python -c \"import sys; print('bad-out'); print('bad-err', file=sys.stderr); sys.exit(7)\""])

    assert result.status == "failed"
    assert result.failed_command == "python -c \"import sys; print('bad-out'); print('bad-err', file=sys.stderr); sys.exit(7)\""
    assert result.exit_code == 7
    assert result.stdout_excerpt == "bad-out\n"
    assert result.stderr_excerpt == "bad-err\n"
    record = json.loads((writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["exit_code"] == 7
    assert record["timeout"] is False
    assert record["stdout_excerpt"] == "bad-out\n"
    assert record["stderr_excerpt"] == "bad-err\n"
    assert record["timeout_seconds"] == 60


def test_verification_engine_records_timeout(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(
        episode_writer=writer,
        workspace_root=tmp_path,
        timeout_seconds=0.01,
    )

    result = engine.run(["python -c \"import time; print('start'); time.sleep(1)\""])

    assert result.status == "failed"
    assert result.failed_command == "python -c \"import time; print('start'); time.sleep(1)\""
    assert result.exit_code is None
    assert result.failure_reason == "timeout"
    assert result.timeout is True
    record = json.loads((writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    assert record["command"] == "python -c \"import time; print('start'); time.sleep(1)\""
    assert record["status"] == "timeout"
    assert record["exit_code"] is None
    assert record["timeout"] is True
    assert "stdout" in record
    assert "stderr" in record
    assert "stdout_excerpt" in record
    assert "stderr_excerpt" in record
    assert record["duration_seconds"] >= 0
    assert record["timeout_seconds"] == 0.01
