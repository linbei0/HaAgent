"""
tests/integration/runtime/test_verification_engine.py - VerificationEngine 验证命令测试

验证 verification_commands 会执行、落盘，并把失败显式暴露给 orchestrator。
"""

import json
from hashlib import sha256
from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.verification.engine import EXCERPT_LIMIT, VerificationEngine


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
    assert record["stdout_truncated"] is False
    assert record["stdout_original_length"] == len("verified\n")
    assert record["stderr"] == ""
    assert record["stderr_excerpt"] == ""
    assert record["stderr_truncated"] is False
    assert record["stderr_original_length"] == 0
    assert record["redacted"] is False
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
    assert record["stdout_truncated"] is False
    assert record["stderr_truncated"] is False
    assert record["stdout_original_length"] == len("bad-out\n")
    assert record["stderr_original_length"] == len("bad-err\n")
    assert record["redacted"] is False
    assert record["timeout_seconds"] == 60


def test_verification_engine_records_truncation_metadata(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)

    stdout = "x" * (EXCERPT_LIMIT + 7)
    stderr = "y" * (EXCERPT_LIMIT + 9)
    engine.run(
        [
            (
                "python -c "
                "\"import sys; "
                f"print('{stdout}', end=''); "
                f"print('{stderr}', end='', file=sys.stderr)\""
            ),
        ],
    )

    record = json.loads((writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    assert record["stdout_excerpt"] == "x" * EXCERPT_LIMIT
    assert record["stderr_excerpt"] == "y" * EXCERPT_LIMIT
    assert record["stdout_truncated"] is True
    assert record["stderr_truncated"] is True
    assert record["stdout_original_length"] == EXCERPT_LIMIT + 7
    assert record["stderr_original_length"] == EXCERPT_LIMIT + 9
    assert record["redacted"] is False


def test_verification_engine_redacts_sensitive_output(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)
    raw_key = "OPENAI_API_KEY=super-secret-value"
    raw_token = "sk-" + ("a" * 48)

    result = engine.run(
        [
            (
                "python -c "
                "\"import sys; "
                f"print('{raw_key}'); "
                f"print('{raw_token}', file=sys.stderr); "
                "sys.exit(3)\""
            ),
        ],
    )

    record = json.loads((writer.path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    assert result.status == "failed"
    assert raw_key not in record["stdout"]
    assert raw_key not in record["stdout_excerpt"]
    assert raw_token not in record["stderr"]
    assert raw_token not in record["stderr_excerpt"]
    assert record["stdout_excerpt"] == "OPENAI_API_KEY=[REDACTED]\n"
    assert record["stderr_excerpt"] == "[REDACTED_TOKEN]\n"
    assert record["redacted"] is True


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


def test_verification_engine_records_final_workspace_file_evidence(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("final", encoding="utf-8")
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)

    result = engine.run([], changed_files=[{"path": str(target), "change_type": "modified"}])

    assert result.status == "success"
    record = json.loads((writer.path / "verification" / "files.jsonl").read_text(encoding="utf-8"))
    assert record == {
        "path": "notes.txt",
        "change_type": "modified",
        "status": "success",
        "size_bytes": 5,
        "sha256": sha256(b"final").hexdigest(),
    }


def test_verification_engine_fails_when_changed_workspace_file_is_missing(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    engine = VerificationEngine(episode_writer=writer, workspace_root=tmp_path)

    result = engine.run([], changed_files=[{"path": str(tmp_path / "missing.txt"), "change_type": "added"}])

    assert result.status == "failed"
    assert result.failure_reason == "changed workspace file is missing"
    record = json.loads((writer.path / "verification" / "files.jsonl").read_text(encoding="utf-8"))
    assert record == {
        "path": "missing.txt",
        "change_type": "added",
        "status": "failed",
        "reason": "changed workspace file is missing",
    }
