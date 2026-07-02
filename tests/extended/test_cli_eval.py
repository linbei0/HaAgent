"""
tests/extended/test_cli_eval.py - eval runner CLI 测试

验证 haagent eval 命令可以运行 eval case 并写出 JSON 报告。
"""

import json
from pathlib import Path

from haagent import cli
from haagent.runtime.eval_export import export_eval_case
from haagent.runtime.orchestrator import RunOrchestrator


def write_task(path: Path) -> None:
    path.write_text(
        """
goal: CLI eval task
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Eval report is written
verification_commands: []
""".strip(),
        encoding="utf-8",
    )


def test_cli_eval_writes_output_report(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    run_result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(export_eval_case(run_result.episode_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "eval-report.json"

    exit_code = cli.main(
        [
            "eval",
            str(case_path),
            "--output",
            str(output_path),
            "--provider",
            "fake",
        ],
    )

    stdout = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert f"eval_report={output_path}" in stdout
    assert "total=1" in stdout
    assert "passed=1" in stdout
    assert "failed=0" in stdout
    assert "error=0" in stdout
    assert report["total_count"] == 1
    assert report["passed_count"] == 1
    assert report["results"][0]["status"] == "passed"


def test_cli_eval_stdout_summary_returns_failure_code_for_mismatch(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    run_result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    case_path = tmp_path / "case.json"
    case = export_eval_case(run_result.episode_path)
    case["expected_tool_uses"] = ["file_read"]
    case_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")

    exit_code = cli.main(["eval", str(case_path), "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "status=failed" in output
    assert "passed=0" in output
    assert "failed=1" in output
    assert "error=0" in output
    assert f"failure_case={case_path}" in output
    assert "failure_reason=missing expected tool uses" in output


def test_cli_eval_builtin_suite_reports_counts_and_report_path(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "builtin-report.json"

    exit_code = cli.main(
        [
            "eval",
            "examples/evals",
            "--output",
            str(output_path),
            "--provider",
            "fake",
        ],
    )

    stdout = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert f"eval_report={output_path}" in stdout
    assert "total=5" in stdout
    assert "passed=5" in stdout
    assert "failed=0" in stdout
    assert "error=0" in stdout
    assert report["total_count"] == 5
