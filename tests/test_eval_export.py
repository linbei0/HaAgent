"""
tests/test_eval_export.py - Eval Case Export 测试

验证 episode package 可以导出为后续 eval 数据管线可消费的最小字典。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.models.gateway import ModelResponse, ToolCall
from agentfoundry.runtime import eval_export
from agentfoundry.runtime.episode_validator import EpisodePackageView, EpisodeValidationError
from agentfoundry.runtime.eval_export import EVAL_CASE_VERSION, export_eval_case
from agentfoundry.runtime.orchestrator import RunOrchestrator
from agentfoundry.runtime.state import RunStatus


class BadArgsGateway:
    provider_name = "bad-args"

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        if observations:
            return ModelResponse("done", [])
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class ShellOnceGateway:
    provider_name = "shell-once"

    def __init__(self) -> None:
        self._called = False

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        if self._called or observations:
            return ModelResponse("done", [])
        self._called = True
        return ModelResponse("shell", [ToolCall("shell", {"command": "echo approval"})])


def write_task(path: Path, verification_commands: list[str] | None = None) -> None:
    verification_commands = verification_commands or []
    verification_yaml = "\n".join(f"  - {command}" for command in verification_commands)
    verification_block = f"\n{verification_yaml}" if verification_yaml else " []"
    path.write_text(
        f"""
goal: Export eval case
constraints:
  - Keep export deterministic
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Eval case contains task facts
verification_commands:{verification_block}
""".strip(),
        encoding="utf-8",
    )


def test_completed_episode_can_export_eval_case(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    eval_case = export_eval_case(result.episode_path)

    assert result.status is RunStatus.COMPLETED
    assert eval_case["eval_case_version"] == EVAL_CASE_VERSION
    assert eval_case["episode_version"] == "1.0"
    assert eval_case["task"]["goal"] == "Export eval case"
    assert eval_case["task"]["acceptance_criteria"] == ["Eval case contains task facts"]
    assert eval_case["task"]["verification_commands"] == []
    assert eval_case["workspace_root"] == str(tmp_path.resolve())
    assert eval_case["final_status"] == "completed"
    assert eval_case["failure"] is None
    assert eval_case["verification"] == []
    assert eval_case["tool_names_used"] == ["fake_tool"]
    assert eval_case["tool_argument_errors"] == []
    sandbox = json.loads((result.episode_path / "sandbox.json").read_text(encoding="utf-8"))
    assert eval_case["sandbox_summary"] == {
        "workspace_root": sandbox["workspace_root"],
        "filesystem_boundary": sandbox["filesystem_boundary"],
        "network_policy": sandbox["network_policy"],
        "process_policy": sandbox["process_policy"],
        "credential_policy": sandbox["credential_policy"],
        "command_timeout_seconds": sandbox["resource_limits"]["command_timeout_seconds"],
    }
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "fake_tool",
            "action": "allow",
            "approval_required": False,
            "approval_status": "not_required",
            "approval_reason": "approval not required for low risk tool fake_tool",
        },
    ]
    assert eval_case["next_actions"] == [
        {
            "context_id": "0001",
            "status": "none",
            "reason": "none",
            "based_on_observation_index": None,
            "based_on_tool_name": None,
        },
        {
            "context_id": "0002",
            "status": "continue",
            "reason": "Continue from the latest successful tool observation and judge whether the acceptance criteria are satisfied.",
            "based_on_observation_index": 0,
            "based_on_tool_name": "fake_tool",
        },
    ]


def test_failed_episode_exports_failure_information(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path, verification_commands=["python -c \"import sys; sys.exit(7)\""])
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    eval_case = export_eval_case(result.episode_path)

    assert result.status is RunStatus.FAILED
    assert eval_case["eval_case_version"] == EVAL_CASE_VERSION
    assert eval_case["final_status"] == "failed"
    assert eval_case["failure"]["category"] == "Verification Failure"
    assert eval_case["failure"]["stage"] == "verifying"
    assert "exit_code=7" in eval_case["failure"]["evidence"]
    assert eval_case["verification"] == [
        {
            "command": "python -c \"import sys; sys.exit(7)\"",
            "status": "failed",
            "exit_code": 7,
            "timeout": False,
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "stdout_original_length": 0,
            "stderr_original_length": 0,
            "redacted": False,
        },
    ]


def test_eval_export_includes_tool_argument_errors(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export tool argument errors
constraints: []
allowed_tools:
  - file_read
acceptance_criteria:
  - Argument error is exported
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=BadArgsGateway(),
    ).run(task_path)
    verification_dir = result.episode_path / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "commands.jsonl").write_text("", encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert result.status is RunStatus.FAILED
    assert eval_case["tool_argument_errors"] == [
        {
            "tool_name": "file_read",
            "message": "missing required argument: path",
        },
    ]


def test_eval_export_approval_summary_marks_missing_for_denied_high_risk_tool(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export missing approval
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=ShellOnceGateway(),
    ).run(task_path)
    verification_dir = result.episode_path / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "commands.jsonl").write_text("", encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert result.status is RunStatus.FAILED
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "shell",
            "action": "deny",
            "approval_required": True,
            "approval_status": "missing",
            "approval_reason": "approval allowed but missing for high risk tool shell",
        },
    ]


def test_eval_export_approval_summary_marks_granted_for_approved_high_risk_tool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export granted approval
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
  approved_tools:
    - shell
""".strip(),
        encoding="utf-8",
    )

    def approved_shell(args, workspace_root):
        return {"status": "success", "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr("agentfoundry.tools.router.shell", approved_shell)
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=ShellOnceGateway(),
    ).run(task_path)

    eval_case = export_eval_case(result.episode_path)

    assert result.status is RunStatus.COMPLETED
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "shell",
            "action": "allow",
            "approval_required": True,
            "approval_status": "granted",
            "approval_reason": "approval granted for high risk tool shell",
        },
    ]


def test_eval_export_defaults_missing_approval_summary_for_legacy_tool_call(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    tool_calls_path = result.episode_path / "tool-calls.jsonl"
    record = json.loads(tool_calls_path.read_text(encoding="utf-8"))
    record.pop("policy", None)
    tool_calls_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert eval_case["approval_summary"] == [
        {
            "tool_name": "fake_tool",
            "action": "missing",
            "approval_required": False,
            "approval_status": "missing",
            "approval_reason": "legacy/missing",
        },
    ]


def test_eval_export_includes_verification_evidence_metadata(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    stdout = "x" * 2007
    raw_key = "OPENAI_API_KEY=super-secret-value"
    write_task(
        task_path,
        verification_commands=[
            (
                "python -c "
                "\"import sys; "
                f"print('{stdout}', end=''); "
                f"print('{raw_key}', file=sys.stderr); "
                "sys.exit(5)\""
            ),
        ],
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    eval_case = export_eval_case(result.episode_path)
    verification = eval_case["verification"][0]

    assert result.status is RunStatus.FAILED
    assert verification["stdout_excerpt"] == "x" * 2000
    assert verification["stdout_truncated"] is True
    assert verification["stdout_original_length"] == 2007
    assert verification["stderr_excerpt"] == "OPENAI_API_KEY=[REDACTED]\n"
    assert verification["stderr_truncated"] is False
    assert verification["stderr_original_length"] == len(raw_key + "\n")
    assert verification["redacted"] is True
    assert raw_key not in verification["stderr_excerpt"]


def test_eval_export_reads_sandbox_summary_from_sandbox_json(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    sandbox_path = result.episode_path / "sandbox.json"
    sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    sandbox["network_policy"] = "test-network-policy"
    sandbox["resource_limits"]["command_timeout_seconds"] = 12.5
    sandbox_path.write_text(json.dumps(sandbox), encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert eval_case["sandbox_summary"] == {
        "workspace_root": sandbox["workspace_root"],
        "filesystem_boundary": sandbox["filesystem_boundary"],
        "network_policy": "test-network-policy",
        "process_policy": sandbox["process_policy"],
        "credential_policy": sandbox["credential_policy"],
        "command_timeout_seconds": 12.5,
    }


def test_eval_export_rejects_missing_sandbox_through_validator(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "sandbox.json").unlink()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: sandbox.json",
    ):
        export_eval_case(result.episode_path)


def test_eval_export_rejects_damaged_sandbox_through_validator(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    sandbox_path = result.episode_path / "sandbox.json"
    sandbox = json.loads(sandbox_path.read_text(encoding="utf-8"))
    sandbox["resource_limits"]["command_timeout_seconds"] = "sixty"
    sandbox_path.write_text(json.dumps(sandbox), encoding="utf-8")

    with pytest.raises(
        EpisodeValidationError,
        match="sandbox.json resource_limits.command_timeout_seconds must be a number",
    ):
        export_eval_case(result.episode_path)


def test_eval_export_defaults_missing_verification_metadata(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path, verification_commands=["python -c \"import sys; sys.exit(7)\""])
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    commands_path = result.episode_path / "verification" / "commands.jsonl"
    record = json.loads(commands_path.read_text(encoding="utf-8"))
    for field_name in [
        "stdout_excerpt",
        "stderr_excerpt",
        "stdout_truncated",
        "stderr_truncated",
        "stdout_original_length",
        "stderr_original_length",
        "redacted",
    ]:
        record.pop(field_name, None)
    commands_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert eval_case["verification"][0]["stdout_excerpt"] == ""
    assert eval_case["verification"][0]["stderr_excerpt"] == ""
    assert eval_case["verification"][0]["stdout_truncated"] is False
    assert eval_case["verification"][0]["stderr_truncated"] is False
    assert eval_case["verification"][0]["stdout_original_length"] == 0
    assert eval_case["verification"][0]["stderr_original_length"] == 0
    assert eval_case["verification"][0]["redacted"] is False


def test_invalid_episode_fails_through_validator(tmp_path: Path) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    with pytest.raises(
        EpisodeValidationError,
        match="episode package missing required file: episode.json",
    ):
        export_eval_case(episode_path)


def test_exporting_same_episode_is_deterministic(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    first_export = export_eval_case(result.episode_path)
    second_export = export_eval_case(result.episode_path)

    assert first_export == second_export
    assert first_export["sandbox_summary"] == second_export["sandbox_summary"]


def test_export_eval_case_marks_missing_next_action(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    first_context_path = result.episode_path / "contexts" / "0001.json"
    first_context = json.loads(first_context_path.read_text(encoding="utf-8"))
    first_context.pop("next_action")
    first_context_path.write_text(json.dumps(first_context), encoding="utf-8")

    eval_case = export_eval_case(result.episode_path)

    assert eval_case["next_actions"][0] == {
        "context_id": "0001",
        "status": "missing",
        "reason": "legacy/missing",
        "based_on_observation_index": None,
        "based_on_tool_name": None,
    }


def test_export_eval_case_uses_package_view(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()
    (episode_path / "task.yaml").write_text(task_path.read_text(encoding="utf-8"), encoding="utf-8")
    package_view = EpisodePackageView(
        episode_metadata={
            "episode_version": "1.0",
            "workspace_root": str(tmp_path),
            "status": "completed",
        },
        failure_record={"status": "success", "failure": None},
        context_manifest={"context_count": 0, "contexts": []},
        transcript=[],
        tool_calls=[{"tool_name": "fake_tool", "status": "success"}],
        verification_commands=[],
        sandbox={
            "workspace_root": str(tmp_path),
            "filesystem_boundary": "workspace_root",
            "network_policy": "unrestricted",
            "process_policy": "local_subprocess",
            "credential_policy": "inherit_environment",
            "resource_limits": {"command_timeout_seconds": 60},
        },
    )

    monkeypatch.setattr(eval_export, "load_validated_episode_package", lambda path: package_view)

    eval_case = export_eval_case(episode_path)

    assert eval_case["episode_version"] == "1.0"
    assert eval_case["task"]["goal"] == "Export eval case"
    assert eval_case["tool_names_used"] == ["fake_tool"]
    assert eval_case["tool_argument_errors"] == []
    assert eval_case["sandbox_summary"] == {
        "workspace_root": str(tmp_path),
        "filesystem_boundary": "workspace_root",
        "network_policy": "unrestricted",
        "process_policy": "local_subprocess",
        "credential_policy": "inherit_environment",
        "command_timeout_seconds": 60,
    }
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "fake_tool",
            "action": "missing",
            "approval_required": False,
            "approval_status": "missing",
            "approval_reason": "legacy/missing",
        },
    ]
    assert eval_case["next_actions"] == []
