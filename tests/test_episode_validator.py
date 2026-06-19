"""
tests/test_episode_validator.py - Episode Validator 测试

验证 episode.json 和 failure.json 的 inspect 侧 schema 校验行为。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.runtime.episode_validator import (
    EpisodeValidationError,
    read_episode_metadata,
    read_failure_record,
    validate_episode_package,
)
from agentfoundry.runtime.orchestrator import RunOrchestrator
from agentfoundry.runtime.state import RunStatus


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_task(path: Path) -> None:
    path.write_text(
        """
goal: Validate package
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Run reaches completed state
verification_commands: []
""".strip(),
        encoding="utf-8",
    )


def test_validator_accepts_valid_episode_metadata(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", valid_episode_json(tmp_path))

    metadata, warnings = read_episode_metadata(episode_path)

    assert metadata is not None
    assert metadata["status"] == "completed"
    assert warnings == []


def test_validator_rejects_unknown_episode_version(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", {"episode_version": "9.9"})

    with pytest.raises(EpisodeValidationError, match="unsupported episode_version: 9.9"):
        read_episode_metadata(episode_path)


def test_validator_rejects_invalid_episode_status(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(episode_path / "episode.json", valid_episode_json(tmp_path, status="done-ish"))

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: episode.json status is invalid: done-ish",
    ):
        read_episode_metadata(episode_path)


@pytest.mark.parametrize("created_at", [123, "not-a-date"])
def test_validator_rejects_invalid_created_at(tmp_path: Path, created_at: object) -> None:
    episode_path = tmp_path / "episode-1"
    payload = valid_episode_json(tmp_path)
    payload["created_at"] = created_at
    write_json(episode_path / "episode.json", payload)

    with pytest.raises(EpisodeValidationError, match="corrupt episode: episode.json created_at"):
        read_episode_metadata(episode_path)


def test_validator_rejects_non_string_workspace_root(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    payload = valid_episode_json(tmp_path)
    payload["workspace_root"] = 123
    write_json(episode_path / "episode.json", payload)

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: episode.json workspace_root must be a string",
    ):
        read_episode_metadata(episode_path)


def test_validator_accepts_legacy_missing_episode_and_failure_json(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    metadata, warnings = read_episode_metadata(episode_path)
    failure_record = read_failure_record(episode_path)

    assert metadata is None
    assert warnings == ["warning: episode.json missing; inspecting legacy episode", ""]
    assert failure_record is None


def test_validator_rejects_failure_unknown_category(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(
        episode_path / "failure.json",
        {
            "status": "failed",
            "failure": {
                "category": "Surprise Failure",
                "stage": "verifying",
                "evidence": "bad",
            },
        },
    )

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: failure.json category is invalid: Surprise Failure",
    ):
        read_failure_record(episode_path)


def test_validator_rejects_failure_invalid_stage(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(
        episode_path / "failure.json",
        {
            "status": "failed",
            "failure": {
                "category": "Verification Failure",
                "stage": "cleanup",
                "evidence": "bad",
            },
        },
    )

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: failure.json stage is invalid: cleanup",
    ):
        read_failure_record(episode_path)


def test_validator_rejects_failure_non_string_evidence(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    write_json(
        episode_path / "failure.json",
        {
            "status": "failed",
            "failure": {
                "category": "Verification Failure",
                "stage": "verifying",
                "evidence": ["bad"],
            },
        },
    )

    with pytest.raises(
        EpisodeValidationError,
        match="corrupt episode: failure.json evidence must be a string",
    ):
        read_failure_record(episode_path)


def test_package_validator_accepts_new_run_episode(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)

    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    assert result.status is RunStatus.COMPLETED
    validate_episode_package(result.episode_path)


def test_package_validator_rejects_missing_required_file(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "environment.json").unlink()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: environment.json",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_invalid_transcript_jsonl(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "transcript.jsonl").write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(EpisodeValidationError, match="transcript.jsonl line 1 is not valid JSON"):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_call_missing_status(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "tool-calls.jsonl").write_text(
        json.dumps({"tool_name": "fake_tool"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 missing required field: status",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_command_missing_command(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"status": "success"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 missing required field: command",
    ):
        validate_episode_package(result.episode_path)
