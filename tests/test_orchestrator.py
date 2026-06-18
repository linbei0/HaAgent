import json
from pathlib import Path

from agent_foundry.model_gateway import ModelCallError
from agent_foundry.orchestrator import RunOrchestrator
from agent_foundry.states import RunStatus


class FailingGateway:
    provider_name = "failing"

    def generate(self, task):
        raise ModelCallError("model exploded")


def write_task(path: Path, allowed_tools: list[str]) -> None:
    allowed_tools_yaml = "\n".join(f"  - {tool}" for tool in allowed_tools)
    path.write_text(
        f"""
goal: Exercise orchestrator
constraints: []
allowed_tools:
{allowed_tools_yaml}
acceptance_criteria:
  - Run reaches terminal state
verification_commands: []
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
    assert "executing" in failure_text
    tool_trace = (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8")
    assert json.loads(tool_trace)["status"] == "error"


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
