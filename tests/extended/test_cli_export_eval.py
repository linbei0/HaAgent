"""
tests/extended/test_cli_export_eval.py - HaAgent export-eval 子命令测试

验证 eval case 单个和批量导出的 CLI 行为。
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from haagent import cli
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.episodes.validator import EpisodePackageView
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.orchestration.state import RunStatus


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


class OneShotGateway:
    provider_name = "one-shot"

    def generate(self, messages, tool_schemas):
        if any(m.get("role") == "tool" for m in messages):
            return ModelResponse("done", [])
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class ShellOnceGateway:
    provider_name = "shell-once"

    def __init__(self) -> None:
        self._called = False

    def generate(self, messages, tool_schemas):
        if self._called or any(m.get("role") == "tool" for m in messages):
            return ModelResponse("done", [])
        self._called = True
        return ModelResponse("shell", [ToolCall("shell", {"command": "echo approval"})])


def write_minimal_episode(
    episode_path: Path,
    episode_json: dict[str, object] | None = None,
    failure_json: dict[str, object] | None = None,
) -> None:
    (episode_path / "verification").mkdir(parents=True)
    if episode_json is not None:
        (episode_path / "episode.json").write_text(json.dumps(episode_json), encoding="utf-8")
        (episode_path / "task.yaml").write_text(
            """
goal: Inspect me
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
            encoding="utf-8",
        )
        (episode_path / "environment.json").write_text(
            json.dumps(
                {
                    "python": "3.13",
                    "platform": "test",
                    "created_at": "2026-06-19T00:00:00+00:00",
                    "workspace_root": episode_json.get("workspace_root"),
                },
            ),
            encoding="utf-8",
        )
        (episode_path / "sandbox.json").write_text(
            json.dumps(
                {
                    "workspace_root": episode_json.get("workspace_root"),
                    "filesystem_boundary": "workspace_root",
                    "network_policy": "unrestricted",
                    "process_policy": "local_subprocess",
                    "credential_policy": "inherit_environment",
                    "resource_limits": {"command_timeout_seconds": 60},
                },
            ),
            encoding="utf-8",
        )
        (episode_path / "plan.json").write_text(
            json.dumps(
                {
                    "goal": "Inspect me",
                    "allowed_tools": ["fake_tool"],
                    "acceptance_criteria": [],
                    "verification_commands": [],
                    "planned_steps": ["Use allowed tools: fake_tool."],
                },
            ),
            encoding="utf-8",
        )
    if failure_json is not None:
        (episode_path / "failure.json").write_text(json.dumps(failure_json), encoding="utf-8")
    (episode_path / "context-manifest.json").write_text(
        json.dumps({"summary": {"provider": "fake"}, "context_count": 0, "contexts": []}),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    (episode_path / "verification" / "commands.jsonl").write_text("", encoding="utf-8")
    (episode_path / "failure-attribution.md").write_text("current attribution", encoding="utf-8")


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def valid_policy(tool_name: str = "fake_tool") -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "risk_level": "low",
        "action": "allow",
        "reason": f"policy allows low risk tool {tool_name}",
        "approval": {
            "required": False,
            "status": "not_required",
            "reason": f"approval not required for low risk tool {tool_name}",
        },
    }


def valid_verification_command(**updates: object) -> dict[str, object]:
    record = {
        "command": "uv run pytest",
        "status": "success",
        "exit_code": 0,
        "timeout": False,
        "stdout_excerpt": "",
        "stderr_excerpt": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_length": 0,
        "stderr_original_length": 0,
        "redacted": False,
    }
    record.update(updates)
    return record

def test_cli_export_eval_outputs_valid_json_for_episode(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export CLI eval case
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Eval JSON is emitted
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    assert exit_code == 0
    eval_case = json.loads(capsys.readouterr().out)
    assert eval_case["eval_case_version"] == "1.0"
    assert eval_case["task"]["goal"] == "Export CLI eval case"
    assert "sandbox_summary" in eval_case
    assert eval_case["sandbox_summary"]["command_timeout_seconds"] == 60
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "fake_tool",
            "action": "allow",
            "approval_required": False,
            "approval_status": "not_required",
            "approval_reason": "approval not required for low risk tool fake_tool",
        },
    ]


def test_cli_export_eval_uses_pretty_utf8_json(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: 导出 eval case
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - 输出 UTF-8 JSON
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "\n  \"eval_case_version\"" in output
    assert "导出 eval case" in output
    assert "\\u5bfc\\u51fa" not in output


def test_cli_export_eval_writes_output_file(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case to file
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - JSON file is emitted
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    stdout = capsys.readouterr().out
    eval_case = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout.strip() == f"exported_eval_case={output_path}"
    assert "\"eval_case_version\"" not in stdout
    assert eval_case["eval_case_version"] == "1.0"
    assert "sandbox_summary" in eval_case
    assert "approval_summary" in eval_case


def test_cli_export_eval_output_file_parent_must_exist(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case to missing directory
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "missing-parent" / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "output parent directory does not exist" in output
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_cli_export_eval_invalid_episode_with_output_does_not_write_file(
    tmp_path: Path,
    capsys,
) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "sandbox.json").unlink()
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(episode_path), "--output", str(output_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "episode package missing required file: sandbox.json" in output
    assert not output_path.exists()


def test_cli_export_eval_batch_writes_one_file_per_episode(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_dir = tmp_path / "eval-batch"
    output_dir.mkdir()

    exit_code = cli.main(
        [
            "export-eval",
            str(first.episode_path),
            str(second.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    stdout = capsys.readouterr().out
    first_output = output_dir / f"{first.episode_path.name}.json"
    second_output = output_dir / f"{second.episode_path.name}.json"
    first_case = json.loads(first_output.read_text(encoding="utf-8"))
    second_case = json.loads(second_output.read_text(encoding="utf-8"))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert f"exported_eval_case={first_output}" in stdout
    assert f"exported_eval_case={second_output}" in stdout
    assert first_case["eval_case_version"] == "1.0"
    assert second_case["eval_case_version"] == "1.0"
    assert "sandbox_summary" in first_case
    assert "approval_summary" in second_case
    assert manifest["manifest_version"] == "1.0"
    assert datetime.fromisoformat(manifest["generated_at"])
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["total_count"] == 2
    assert manifest["success_count"] == 2
    assert manifest["failure_count"] == 0
    assert manifest["records"] == [
        {
            "episode_path": str(first.episode_path),
            "status": "success",
            "output_file": str(first_output),
            "error": None,
        },
        {
            "episode_path": str(second.episode_path),
            "status": "success",
            "output_file": str(second_output),
            "error": None,
        },
    ]


def test_cli_export_eval_batch_continues_after_invalid_episode(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export valid episode in mixed batch
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    valid = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    invalid = tmp_path / "bad-episode"
    write_minimal_episode(
        invalid,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (invalid / "sandbox.json").unlink()
    output_dir = tmp_path / "eval-batch"
    output_dir.mkdir()

    exit_code = cli.main(
        [
            "export-eval",
            str(invalid),
            str(valid.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    stdout = capsys.readouterr().out
    valid_output = output_dir / f"{valid.episode_path.name}.json"
    invalid_output = output_dir / "bad-episode.json"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert f"error={invalid}: episode package missing required file: sandbox.json" in stdout
    assert f"exported_eval_case={valid_output}" in stdout
    assert valid_output.exists()
    assert not invalid_output.exists()
    assert manifest["total_count"] == 2
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 1
    assert manifest["records"] == [
        {
            "episode_path": str(invalid),
            "status": "error",
            "output_file": None,
            "error": "episode package missing required file: sandbox.json",
        },
        {
            "episode_path": str(valid.episode_path),
            "status": "success",
            "output_file": str(valid_output),
            "error": None,
        },
    ]


def test_cli_export_eval_single_stdout_does_not_write_manifest(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case without manifest
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert not (result.episode_path / "manifest.json").exists()


def test_cli_export_eval_single_output_does_not_write_manifest(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case file without manifest
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    assert exit_code == 0
    assert "exported_eval_case=" in capsys.readouterr().out
    assert output_path.exists()
    assert not (tmp_path / "manifest.json").exists()


def test_cli_export_eval_batch_output_dir_must_exist(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch to missing directory
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_dir = tmp_path / "missing-output-dir"

    exit_code = cli.main(
        [
            "export-eval",
            str(first.episode_path),
            str(second.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "output directory does not exist" in output
    assert not output_dir.exists()


def test_cli_export_eval_multiple_episodes_requires_output_dir(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch without output dir
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(first.episode_path), str(second.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "multiple episode paths require --output-dir" in output


def test_cli_export_eval_invalid_episode_returns_error(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "sandbox.json").unlink()

    exit_code = cli.main(["export-eval", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "error:" in output
    assert "episode package missing required file: sandbox.json" in output


