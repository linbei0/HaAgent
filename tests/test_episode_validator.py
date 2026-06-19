"""
tests/test_episode_validator.py - Episode Validator 测试

验证 episode.json 和 failure.json 的 inspect 侧 schema 校验行为。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.runtime.episode_validator import (
    EpisodeValidationError,
    EpisodePackageView,
    load_validated_episode_package,
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_context_manifest(episode_path: Path, **updates: object) -> None:
    context_manifest = read_json(episode_path / "context-manifest.json")
    context_manifest.update(updates)
    write_json(episode_path / "context-manifest.json", context_manifest)


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


def test_package_validator_rejects_tool_name_non_string(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "tool-calls.jsonl").write_text(
        json.dumps({"tool_name": 123, "status": "success"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 tool_name must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_status_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "tool-calls.jsonl").write_text(
        json.dumps({"tool_name": "fake_tool", "status": "timeout"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 status is invalid: timeout",
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


def test_package_validator_rejects_verification_command_non_string(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": 123, "status": "success"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 command must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_status_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": "uv run pytest", "status": "skipped"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 status is invalid: skipped",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_exit_code_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": "uv run pytest", "status": "success", "exit_code": "0"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 exit_code must be an integer or null",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_timeout_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": "uv run pytest", "status": "success", "timeout": "false"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 timeout must be a bool",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_episode_status_mismatching_transcript(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    episode_json = read_json(result.episode_path / "episode.json")
    episode_json["status"] = "failed"
    write_json(result.episode_path / "episode.json", episode_json)

    with pytest.raises(
        EpisodeValidationError,
        match="episode status failed does not match transcript final status completed",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_failure_success_with_failed_episode(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    episode_json = read_json(result.episode_path / "episode.json")
    episode_json["status"] = "failed"
    write_json(result.episode_path / "episode.json", episode_json)
    (result.episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "failed"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        EpisodeValidationError,
        match="failure.json status success requires episode status completed",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_failure_failed_with_completed_episode(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_json(
        result.episode_path / "failure.json",
        {
            "status": "failed",
            "failure": {
                "category": "Verification Failure",
                "stage": "verifying",
                "evidence": "bad",
            },
        },
    )

    with pytest.raises(
        EpisodeValidationError,
        match="failure.json status failed requires episode status failed",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_context_count_mismatch(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    context_manifest = read_json(result.episode_path / "context-manifest.json")
    context_manifest["context_count"] = 999
    write_json(result.episode_path / "context-manifest.json", context_manifest)

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json context_count 999 does not match contexts length",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_non_integer_context_count(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    update_context_manifest(result.episode_path, context_count="2")

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json context_count must be an integer",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_non_object_context_item(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    update_context_manifest(result.episode_path, context_count=1, contexts=["bad"])

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json contexts\\[0\\] must be an object",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_context_item_missing_field(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    update_context_manifest(
        result.episode_path,
        context_count=1,
        contexts=[
            {
                "context_id": "0001",
                "model_input_path": "contexts/0001.txt",
            },
        ],
    )

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json contexts\\[0\\] missing required field: manifest_path",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_context_item_non_string_field(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    update_context_manifest(
        result.episode_path,
        context_count=1,
        contexts=[
            {
                "context_id": "0001",
                "model_input_path": 123,
                "manifest_path": "contexts/0001.json",
            },
        ],
    )

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json contexts\\[0\\].model_input_path must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_missing_context_file(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    context_manifest = read_json(result.episode_path / "context-manifest.json")
    first_context = context_manifest["contexts"][0]
    (result.episode_path / first_context["model_input_path"]).unlink()

    with pytest.raises(
        EpisodeValidationError,
        match="context-manifest.json contexts\\[0\\].model_input_path file missing",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_missing_state_transition(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "model_call", "provider": "fake"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EpisodeValidationError, match="transcript.jsonl missing state_transition"):
        validate_episode_package(result.episode_path)


def test_load_validated_episode_package_returns_view_for_valid_run(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    package_view = load_validated_episode_package(result.episode_path)

    assert isinstance(package_view, EpisodePackageView)
    assert package_view.episode_metadata["status"] == "completed"
    assert package_view.failure_record == {"status": "success", "failure": None}
    assert package_view.context_manifest["context_count"] == 2


def test_load_validated_episode_package_raises_for_invalid_episode(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: episode.json",
    ):
        load_validated_episode_package(episode_path)


def test_load_validated_episode_package_returns_parsed_jsonl_lists(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    package_view = load_validated_episode_package(result.episode_path)

    assert isinstance(package_view.transcript, list)
    assert all(isinstance(record, dict) for record in package_view.transcript)
    assert isinstance(package_view.tool_calls, list)
    assert all(isinstance(record, dict) for record in package_view.tool_calls)
    assert isinstance(package_view.verification_commands, list)
    assert all(isinstance(record, dict) for record in package_view.verification_commands)
