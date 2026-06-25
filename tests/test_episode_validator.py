"""
tests/test_episode_validator.py - Episode Validator 测试

验证 episode.json 和 failure.json 的 inspect 侧 schema 校验行为。
"""

import json
from pathlib import Path

import pytest

from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.episode_validator import (
    EpisodeValidationError,
    EpisodePackageView,
    load_validated_episode_package,
    read_episode_metadata,
    read_failure_record,
    validate_episode_package,
)
from haagent.runtime.orchestrator import RunOrchestrator
from haagent.runtime.state import RunStatus
from tests.support.episode_packages import (
    read_json,
    valid_episode_json,
    valid_policy,
    valid_verification_command,
    write_json,
    write_task,
    write_tool_call,
    write_verification_command,
)


def update_context_manifest(episode_path: Path, **updates: object) -> None:
    context_manifest = read_json(episode_path / "context-manifest.json")
    context_manifest.update(updates)
    write_json(episode_path / "context-manifest.json", context_manifest)


def update_first_context_index_budget(episode_path: Path, budget: object | None) -> None:
    context_manifest = read_json(episode_path / "context-manifest.json")
    if budget is None:
        context_manifest["contexts"][0].pop("budget", None)
    else:
        context_manifest["contexts"][0]["budget"] = budget
    write_json(episode_path / "context-manifest.json", context_manifest)


def first_context_json_path(episode_path: Path) -> Path:
    context_manifest = read_json(episode_path / "context-manifest.json")
    return episode_path / context_manifest["contexts"][0]["manifest_path"]


def update_first_context_json(episode_path: Path, **updates: object) -> None:
    path = first_context_json_path(episode_path)
    context_json = read_json(path)
    context_json.update(updates)
    write_json(path, context_json)


def update_environment(episode_path: Path, **updates: object) -> None:
    environment = read_json(episode_path / "environment.json")
    environment.update(updates)
    write_json(episode_path / "environment.json", environment)


def update_sandbox(episode_path: Path, **updates: object) -> None:
    sandbox = read_json(episode_path / "sandbox.json")
    sandbox.update(updates)
    write_json(episode_path / "sandbox.json", sandbox)


class OneShotGateway:
    provider_name = "one-shot"

    def __init__(self, response: ModelResponse) -> None:
        self._response = response

    def generate(self, messages, tool_schemas):
        return self._response


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


def test_validator_rejects_missing_episode_json(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: episode.json",
    ):
        read_episode_metadata(episode_path)


def test_validator_rejects_missing_failure_json(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: failure.json",
    ):
        read_failure_record(episode_path)


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


def test_package_validator_rejects_missing_plan_json(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "plan.json").unlink()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: plan.json",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_missing_sandbox_json(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "sandbox.json").unlink()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: sandbox.json",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_sandbox_field_type_error(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    update_sandbox(result.episode_path, network_policy=123)

    with pytest.raises(
        EpisodeValidationError,
        match="sandbox.json network_policy must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_sandbox_command_timeout_type_error(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    sandbox = read_json(result.episode_path / "sandbox.json")
    sandbox["resource_limits"]["command_timeout_seconds"] = "60"
    write_json(result.episode_path / "sandbox.json", sandbox)

    with pytest.raises(
        EpisodeValidationError,
        match="sandbox.json resource_limits.command_timeout_seconds must be a number",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_plan_field_type_error(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    plan = read_json(result.episode_path / "plan.json")
    plan["planned_steps"] = "not-a-list"
    write_json(result.episode_path / "plan.json", plan)

    with pytest.raises(
        EpisodeValidationError,
        match="plan.json planned_steps must be a list of strings",
    ):
        validate_episode_package(result.episode_path)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("python", 123, "environment.json python must be a string"),
        ("platform", 123, "environment.json platform must be a string"),
        ("created_at", "not-a-date", "environment.json created_at is invalid: not-a-date"),
        ("workspace_root", 123, "environment.json workspace_root must be a string"),
        (
            "workspace_root",
            "__other_root__",
            "environment.json workspace_root does not match episode.json workspace_root",
        ),
    ],
)
def test_package_validator_rejects_environment_field_errors(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    value = str(tmp_path / "other") if bad_value == "__other_root__" else bad_value
    update_environment(result.episode_path, **{field: value})

    with pytest.raises(EpisodeValidationError, match=message):
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
    write_tool_call(result.episode_path, include_policy=True)
    record = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    record.pop("status")
    (result.episode_path / "tool-calls.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 missing required field: status",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_name_non_string(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, tool_name=123, policy=valid_policy("fake_tool"))

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 tool_name must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_status_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, status="timeout")

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 status is invalid: timeout",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_accepts_tool_status_error(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, status="error")

    validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_call_missing_policy(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, include_policy=False)

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 missing required field: policy",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_call_missing_approval(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    policy = valid_policy()
    policy.pop("approval")
    write_tool_call(result.episode_path, policy=policy)

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 policy.approval must be an object",
    ):
        validate_episode_package(result.episode_path)


@pytest.mark.parametrize("error_type", ["tool_not_allowed", "unknown_tool"])
def test_package_validator_allows_policy_none_for_policy_not_evaluated_errors(
    tmp_path: Path,
    error_type: str,
) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(
        result.episode_path,
        status="error",
        policy=None,
        error={"type": error_type, "message": "policy not evaluated"},
    )

    validate_episode_package(result.episode_path)


def test_package_validator_rejects_tool_status_failed(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, status="failed")

    with pytest.raises(
        EpisodeValidationError,
        match="tool-calls.jsonl line 1 status is invalid: failed",
    ):
        validate_episode_package(result.episode_path)


@pytest.mark.parametrize("status", ["timeout", "skipped"])
def test_package_validator_rejects_unknown_tool_status(tmp_path: Path, status: str) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_tool_call(result.episode_path, status=status)

    with pytest.raises(
        EpisodeValidationError,
        match=f"tool-calls.jsonl line 1 status is invalid: {status}",
    ):
        validate_episode_package(result.episode_path)


def test_failed_run_with_tool_argument_error_validates_package(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    gateway = OneShotGateway(
        ModelResponse(
            content="",
            tool_calls=[ToolCall(name="file_read", args={"path": 123})],
        ),
    )

    result = RunOrchestrator(runs_root=tmp_path / ".runs", model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert tool_call["status"] == "error"
    (result.episode_path / "verification").mkdir(exist_ok=True)
    (result.episode_path / "verification" / "commands.jsonl").write_text("", encoding="utf-8")
    validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_command_missing_command(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path)
    record = json.loads((result.episode_path / "verification" / "commands.jsonl").read_text(encoding="utf-8"))
    record.pop("command")
    (result.episode_path / "verification" / "commands.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 missing required field: command",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_command_non_string(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path, command=123)

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 command must be a string",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_status_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path, status="skipped")

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 status is invalid: skipped",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_exit_code_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path, exit_code="0")

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 exit_code must be an integer or null",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_verification_timeout_invalid(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path, timeout="false")

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 timeout must be a bool",
    ):
        validate_episode_package(result.episode_path)


def test_package_validator_rejects_missing_verification_metadata(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    write_verification_command(result.episode_path)
    commands_path = result.episode_path / "verification" / "commands.jsonl"
    record = json.loads(commands_path.read_text(encoding="utf-8"))
    record.pop("stdout_excerpt")
    commands_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match="verification/commands.jsonl line 1 missing required field: stdout_excerpt",
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
                "model_input_path": "contexts/0001.json",
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
    assert package_view.context_manifest["context_count"] == 1


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
