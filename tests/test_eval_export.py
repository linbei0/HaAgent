"""
tests/test_eval_export.py - Eval Case Export 测试

验证 episode package 可以导出为后续 eval 数据管线可消费的最小字典。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.runtime import eval_export
from agentfoundry.runtime.episode_validator import EpisodePackageView, EpisodeValidationError
from agentfoundry.runtime.eval_export import EVAL_CASE_VERSION, export_eval_case
from agentfoundry.runtime.orchestrator import RunOrchestrator
from agentfoundry.runtime.state import RunStatus


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
        },
    ]


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
    )

    monkeypatch.setattr(eval_export, "load_validated_episode_package", lambda path: package_view)

    eval_case = export_eval_case(episode_path)

    assert eval_case["episode_version"] == "1.0"
    assert eval_case["task"]["goal"] == "Export eval case"
    assert eval_case["tool_names_used"] == ["fake_tool"]
    assert eval_case["next_actions"] == []
