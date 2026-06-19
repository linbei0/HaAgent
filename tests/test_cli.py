"""
tests/test_cli.py - AgentFoundry CLI 测试

验证 run/inspect 子命令参数解析、runs-root 传递和 stdout 输出。
"""

import json
from pathlib import Path

from agentfoundry import cli
from agentfoundry.runtime.state import RunStatus


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


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
            json.dumps({"workspace_root": episode_json.get("workspace_root")}),
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
    (episode_path / "failure-attribution.md").write_text("legacy", encoding="utf-8")


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def test_cli_run_uses_default_runs_root_and_prints_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path)])

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}
    output = capsys.readouterr().out
    assert "status=completed" in output
    assert f"episode_path={tmp_path / '.runs' / 'episode-1'}" in output


def test_cli_run_accepts_custom_runs_root(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    custom_runs = tmp_path / "custom-runs"
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(custom_runs / "episode-1")

    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path), "--runs-root", str(custom_runs)])

    assert exit_code == 0
    assert calls == {"runs_root": custom_runs, "task_path": task_path}


def test_cli_inspect_completed_episode_outputs_summary(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
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
    (episode_path / "episode.json").write_text(
        json.dumps(
            {
                "episode_version": "1.0",
                "created_at": "2026-06-19T00:00:00+00:00",
                "task_path": "task.yaml",
                "status": "completed",
                "provider": "fake",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "context-manifest.json").write_text(
        json.dumps(
            {
                "version": "1.1",
                "context_count": 1,
                "summary": {"provider": "fake", "goal": "Inspect me"},
                "contexts": [
                    {
                        "context_id": "0001",
                        "model_input_path": "contexts/0001.txt",
                        "manifest_path": "contexts/0001.json",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "state_transition", "status": "created"}),
                json.dumps({"event": "state_transition", "status": "completed"}),
                json.dumps({"event": "model_call", "provider": "fake", "context_id": "0001"}),
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text(
        json.dumps({"tool_name": "fake_tool", "status": "success"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": "uv run pytest", "status": "success", "exit_code": 0}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "failure.json").write_text(
        json.dumps({"status": "success", "failure": None}),
        encoding="utf-8",
    )
    (episode_path / "failure-attribution.md").write_text("# Failure Attribution\n\n未失败。\n", encoding="utf-8")
    (episode_path / "environment.json").write_text(
        json.dumps({"workspace_root": str(tmp_path)}),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Run Summary" in output
    assert "episode_version: 1.0" in output
    assert "status: completed" in output
    assert "State Flow" in output
    assert "created -> completed" in output
    assert "Contexts" in output
    assert "0001" in output
    assert "Model Calls" in output
    assert "Tool Calls" in output
    assert "fake_tool: success" in output
    assert "Verification" in output
    assert "uv run pytest: success (exit_code=0)" in output
    assert "Failure Attribution" in output
    assert "未失败" in output


def test_cli_inspect_outputs_verification_evidence(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
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
    (episode_path / "episode.json").write_text(
        json.dumps(
            {
                "episode_version": "1.0",
                "created_at": "2026-06-19T00:00:00+00:00",
                "task_path": "task.yaml",
                "status": "failed",
                "provider": "fake",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "context-manifest.json").write_text(
        json.dumps({"summary": {"provider": "fake"}, "context_count": 0, "contexts": []}),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "failed"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    (episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps(
            {
                "command": "python fail.py",
                "status": "failed",
                "exit_code": 3,
                "timeout": False,
                "stdout_excerpt": "out evidence",
                "stderr_excerpt": "err evidence",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_path / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure": {
                    "category": "Verification Failure",
                    "stage": "verifying",
                    "evidence": "structured evidence",
                },
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "failure-attribution.md").write_text(
        "# Failure Attribution\n\n- category: Verification Failure\n",
        encoding="utf-8",
    )
    (episode_path / "environment.json").write_text(
        json.dumps({"workspace_root": str(tmp_path)}),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "python fail.py: failed (exit_code=3)" in output
    assert "stdout: out evidence" in output
    assert "stderr: err evidence" in output
    assert "Structured Failure" in output
    assert "category: Verification Failure" in output
    assert "stage: verifying" in output
    assert "structured evidence" in output


def test_cli_inspect_legacy_episode_without_episode_json_warns(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
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
    (episode_path / "failure-attribution.md").write_text("legacy", encoding="utf-8")

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "warning: episode.json missing; inspecting legacy episode" in output


def test_cli_inspect_unknown_episode_version_fails(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()
    (episode_path / "episode.json").write_text(
        json.dumps({"episode_version": "9.9"}),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "unsupported episode_version: 9.9" in output


def test_cli_inspect_fails_when_episode_json_missing_field(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_json = valid_episode_json(tmp_path)
    del episode_json["workspace_root"]
    write_minimal_episode(episode_path, episode_json=episode_json)

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: episode.json missing required field: workspace_root" in output


def test_cli_inspect_fails_when_episode_json_status_is_invalid(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(episode_path, episode_json=valid_episode_json(tmp_path, status="done-ish"))

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: episode.json status is invalid: done-ish" in output


def test_cli_inspect_fails_when_failure_json_category_is_unknown(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path, status="failed"),
        failure_json={
            "status": "failed",
            "failure": {
                "category": "Surprise Failure",
                "stage": "verifying",
                "evidence": "bad",
            },
        },
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: failure.json category is invalid: Surprise Failure" in output


def test_cli_inspect_fails_when_success_failure_json_has_failure(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": {"category": "Runtime Failure"}},
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: failure.json success record must have failure=null" in output


def test_cli_inspect_legacy_episode_without_failure_json_still_works(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(episode_path)

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "legacy episode without failure.json" in output


def test_cli_inspect_new_episode_missing_required_file_uses_validator(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "environment.json").unlink()

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "episode package missing required file: environment.json" in output


def test_cli_inspect_new_episode_invalid_jsonl_uses_validator(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "transcript.jsonl").write_text("{not json}\n", encoding="utf-8")

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "transcript.jsonl line 1 is not valid JSON" in output


def test_cli_inspect_fails_when_required_file_is_missing(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "missing required episode file" in output
    assert "context-manifest.json" in output
