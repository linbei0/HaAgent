"""
tests/test_orchestrator.py - RunOrchestrator 状态流转测试

验证成功路径、工具失败和模型失败会写入正确 run 状态。
"""

import json
from pathlib import Path

from agentfoundry.models.gateway import ModelCallError
from agentfoundry.runtime.orchestrator import RunOrchestrator
from agentfoundry.runtime.state import RunStatus
from agentfoundry.verification.engine import VerificationResult


class FailingGateway:
    provider_name = "failing"

    def generate(self, task):
        raise ModelCallError("model exploded")


def write_task(
    path: Path,
    allowed_tools: list[str],
    verification_commands: list[str] | None = None,
) -> None:
    allowed_tools_yaml = "\n".join(f"  - {tool}" for tool in allowed_tools)
    verification_commands = verification_commands or []
    verification_yaml = "\n".join(f"  - {command}" for command in verification_commands)
    verification_block = f"\n{verification_yaml}" if verification_yaml else " []"
    path.write_text(
        f"""
goal: Exercise orchestrator
constraints: []
allowed_tools:
{allowed_tools_yaml}
acceptance_criteria:
  - Run reaches terminal state
verification_commands:{verification_block}
""".strip(),
        encoding="utf-8",
    )


def test_orchestrator_records_successful_state_flow(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.EXECUTING,
        RunStatus.VERIFYING,
        RunStatus.COMPLETED,
    ]


def test_orchestrator_fails_when_fake_tool_is_not_allowed(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["other_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history[-1] is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Task Spec Failure" in failure_text
    assert "other_tool" in failure_text
    assert (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8") == ""


def test_orchestrator_fails_when_model_gateway_fails(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=FailingGateway()).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Model Failure" in failure_text
    assert "model exploded" in failure_text
    assert (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8") == ""


def test_orchestrator_fails_when_verification_command_fails(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(
        task_path,
        ["fake_tool"],
        verification_commands=["python -c \"import sys; sys.exit(5)\""],
    )

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.EXECUTING,
        RunStatus.VERIFYING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Verification Failure" in failure_text
    commands_log = result.episode_path / "verification" / "commands.jsonl"
    assert json.loads(commands_log.read_text(encoding="utf-8"))["exit_code"] == 5


def test_orchestrator_fails_unknown_tool_as_task_spec_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["mystery_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Task Spec Failure" in failure_text
    assert "mystery_tool" in failure_text


def test_orchestrator_failure_attribution_includes_verification_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"], verification_commands=["slow command"])

    class TimeoutVerificationEngine:
        def __init__(self, episode_writer, workspace_root):
            pass

        def run(self, commands):
            return VerificationResult(
                status="failed",
                failed_command=commands[0],
                exit_code=None,
                failure_reason="timeout",
            )

    monkeypatch.setattr(
        "agentfoundry.runtime.orchestrator.VerificationEngine",
        TimeoutVerificationEngine,
    )

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Verification Failure" in failure_text
    assert "slow command" in failure_text
    assert "timeout" in failure_text
