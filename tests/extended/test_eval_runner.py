"""
tests/extended/test_eval_runner.py - 本地 eval runner 测试

验证导出的 eval case 可以被本地重新运行，并生成确定性回归报告。
"""

import json
from pathlib import Path

from haagent.mcp.client import McpClientManager
from haagent.runtime.evaluation.export import export_eval_case
from haagent.runtime.evaluation.runner import run_eval_path
from haagent.runtime.orchestration.orchestrator import RunOrchestrator


def write_task(path: Path) -> None:
    path.write_text(
        """
goal: Eval runner task
constraints:
  - Keep runner deterministic
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Eval runner can replay this case
verification_commands: []
""".strip(),
        encoding="utf-8",
    )


def write_eval_case(tmp_path: Path, name: str = "case.json") -> Path:
    task_path = tmp_path / f"{name}.task.yaml"
    write_task(task_path)
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    case_path = tmp_path / name
    case_path.write_text(
        json.dumps(export_eval_case(result.episode_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_path


def test_eval_runner_single_case_passes(tmp_path: Path) -> None:
    case_path = write_eval_case(tmp_path)

    report = run_eval_path(case_path, runs_root=tmp_path / ".eval-runs")

    assert report["total_count"] == 1
    assert report["passed_count"] == 1
    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["final_response_match"] is True
    assert result["basic_result_match"] is True
    assert result["expected_tool_uses"] == ["fake_tool"]
    assert result["actual_tool_uses"] == ["fake_tool"]
    assert result["missing_tool_uses"] == []
    assert result["unexpected_tool_uses"] == []
    assert result["episode_path"] is not None
    assert Path(result["episode_path"]).exists()
    assert result["failure_reason"] is None


def test_eval_runner_batch_manifest_runs_multiple_cases(tmp_path: Path) -> None:
    first = write_eval_case(tmp_path, "first.json")
    second = write_eval_case(tmp_path, "second.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "records": [
                    {"status": "success", "output_file": str(first)},
                    {"status": "success", "output_file": str(second)},
                ],
            },
        ),
        encoding="utf-8",
    )

    report = run_eval_path(manifest, runs_root=tmp_path / ".eval-runs")

    assert report["total_count"] == 2
    assert report["passed_count"] == 2
    assert [result["status"] for result in report["results"]] == ["passed", "passed"]


def test_eval_runner_failed_when_expected_tool_is_missing(tmp_path: Path) -> None:
    case_path = write_eval_case(tmp_path)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["expected_tool_uses"] = ["file_read"]
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")

    report = run_eval_path(case_path, runs_root=tmp_path / ".eval-runs")

    result = report["results"][0]
    assert result["status"] == "failed"
    assert result["expected_tool_uses"] == ["file_read"]
    assert result["actual_tool_uses"] == ["fake_tool"]
    assert result["missing_tool_uses"] == ["file_read"]
    assert result["unexpected_tool_uses"] == ["fake_tool"]
    assert "missing expected tool uses" in result["failure_reason"]


def test_eval_runner_failed_when_final_response_mismatches(tmp_path: Path) -> None:
    case_path = write_eval_case(tmp_path)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case["expectations"]["final_response"]["value"] = "not the fake final answer"
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")

    report = run_eval_path(case_path, runs_root=tmp_path / ".eval-runs")

    result = report["results"][0]
    assert result["status"] == "failed"
    assert result["final_response_match"] is False
    assert "final response did not contain expected text" in result["failure_reason"]


def test_eval_runner_corrupt_case_reports_error_without_stopping_batch(tmp_path: Path) -> None:
    good = write_eval_case(tmp_path, "good.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "records": [
                    {"status": "success", "output_file": str(bad)},
                    {"status": "success", "output_file": str(good)},
                ],
            },
        ),
        encoding="utf-8",
    )

    report = run_eval_path(manifest, runs_root=tmp_path / ".eval-runs")

    assert report["total_count"] == 2
    assert report["error_count"] == 1
    assert report["passed_count"] == 1
    assert [result["status"] for result in report["results"]] == ["error", "passed"]
    assert "missing task" in report["results"][0]["failure_reason"]


def test_builtin_eval_suite_is_discoverable_and_runs(tmp_path: Path) -> None:
    suite_path = Path("examples/evals")

    report = run_eval_path(suite_path, runs_root=tmp_path / ".eval-runs", max_turns=5)

    assert report["input_path"] == str(suite_path)
    assert report["total_count"] >= 5
    assert report["passed_count"] == report["total_count"]
    assert report["failed_count"] == 0
    assert report["error_count"] == 0
    assert {
        "builtin-file-read",
        "builtin-file-write",
        "builtin-code-run",
        "builtin-guardrail",
        "builtin-session-working-state",
    } <= {result["eval_id"] for result in report["results"]}


def test_chat_session_eval_does_not_start_user_mcp(monkeypatch, tmp_path: Path) -> None:
    config_dir = Path.home() / ".haagent"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "servers": {
                    "user-server": {
                        "type": "http",
                        "url": "http://127.0.0.1:9/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    async def fail_connect_all(self) -> None:
        raise AssertionError("deterministic eval must not start user MCP servers")

    monkeypatch.setattr(McpClientManager, "connect_all", fail_connect_all)

    report = run_eval_path(
        Path("examples/evals/session_working_state.json"),
        runs_root=tmp_path / ".eval-runs",
    )

    assert report["passed_count"] == 1
    assert report["error_count"] == 0


def test_eval_runner_replays_case_model_responses(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("marker=ALPHA-42\n", encoding="utf-8")
    case_path = tmp_path / "read.json"
    case_path.write_text(
        json.dumps(
            {
                "eval_id": "deterministic-read",
                "workspace_root": "workspace",
                "task": {
                    "goal": "Read note marker",
                    "constraints": [],
                    "allowed_tools": ["file_read"],
                    "acceptance_criteria": ["Answer with the marker."],
                    "verification_commands": [],
                },
                "expected_tool_uses": ["file_read"],
                "expectations": {
                    "final_status": "completed",
                    "final_response": {"mode": "contains", "value": "ALPHA-42"},
                },
                "model_responses": [
                    {
                        "content": "reading",
                        "tool_calls": [{"name": "file_read", "args": {"path": "note.txt"}}],
                    },
                    {"content": "The marker is ALPHA-42.", "tool_calls": []},
                ],
            },
        ),
        encoding="utf-8",
    )

    report = run_eval_path(case_path, runs_root=tmp_path / ".eval-runs", max_turns=3)

    result = report["results"][0]
    assert result["status"] == "passed"
    assert result["actual_tool_uses"] == ["file_read"]


def test_eval_runner_reports_context_expectation_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case_path = tmp_path / "session.json"
    case_path.write_text(
        json.dumps(
            {
                "eval_id": "session-context-mismatch",
                "case_type": "chat_session",
                "workspace_root": "workspace",
                "chat": {"prompts": ["First turn", "Second turn after resume"]},
                "expectations": {
                    "final_status": "completed",
                    "final_response": {"mode": "contains", "value": "second done"},
                    "context_contains": ["NOT_PRESENT_IN_CONTEXT"],
                },
                "model_responses": [
                    {"content": "first done", "tool_calls": []},
                    {"content": "second done", "tool_calls": []},
                ],
            },
        ),
        encoding="utf-8",
    )

    report = run_eval_path(case_path, runs_root=tmp_path / ".eval-runs", max_turns=3)

    result = report["results"][0]
    assert result["status"] == "failed"
    assert "context missing expected text" in result["failure_reason"]


def test_eval_runner_batch_summary_counts_failed_and_error(tmp_path: Path) -> None:
    passing = write_eval_case(tmp_path, "passing.json")
    failing = write_eval_case(tmp_path, "failing.json")
    failing_case = json.loads(failing.read_text(encoding="utf-8"))
    failing_case["expected_tool_uses"] = ["file_read"]
    failing.write_text(json.dumps(failing_case, ensure_ascii=False), encoding="utf-8")
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": "1.0",
                "records": [
                    {"status": "success", "output_file": str(passing)},
                    {"status": "success", "output_file": str(failing)},
                    {"status": "success", "output_file": str(corrupt)},
                ],
            },
        ),
        encoding="utf-8",
    )

    report = run_eval_path(manifest, runs_root=tmp_path / ".eval-runs")

    assert report["total_count"] == 3
    assert report["passed_count"] == 1
    assert report["failed_count"] == 1
    assert report["error_count"] == 1
