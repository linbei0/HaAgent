"""
tests/test_eval_runner.py - 本地 eval runner 测试

验证导出的 eval case 可以被本地重新运行，并生成确定性回归报告。
"""

import json
from pathlib import Path

from haagent.runtime.eval_export import export_eval_case
from haagent.runtime.eval_runner import run_eval_path
from haagent.runtime.orchestrator import RunOrchestrator


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
