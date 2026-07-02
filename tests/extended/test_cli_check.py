"""
tests/extended/test_cli_check.py - check 质量门禁 CLI 测试

验证 haagent check 能一键运行内置 eval suite，并可选执行 pytest。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent import cli
from haagent.runtime import checks


def test_cli_check_runs_builtin_eval_suite_successfully(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(["check", "--runs-root", str(tmp_path / ".runs")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status=passed" in output
    assert "eval_total=5" in output
    assert "eval_passed=5" in output
    assert "eval_failed=0" in output
    assert "eval_error=0" in output


def test_cli_check_custom_failed_eval_returns_nonzero(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "eval_id": "check-failing-case",
                "workspace_root": "workspace",
                "task": {
                    "goal": "Return a deterministic answer",
                    "constraints": [],
                    "allowed_tools": [],
                    "acceptance_criteria": ["Answer should match expectation."],
                    "verification_commands": [],
                },
                "expected_tool_uses": [],
                "expectations": {
                    "final_status": "completed",
                    "final_response": {"mode": "contains", "value": "expected text"},
                },
                "model_responses": [
                    {"content": "different text", "tool_calls": []},
                ],
            },
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "check",
            "--eval-path",
            str(case_path),
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "status=failed" in output
    assert "eval_total=1" in output
    assert "eval_failed=1" in output
    assert f"failure_case={case_path}" in output


def test_cli_check_writes_json_report(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "check-report.json"

    exit_code = cli.main(
        [
            "check",
            "--output",
            str(output_path),
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    stdout = capsys.readouterr().out
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert f"check_report={output_path}" in stdout
    assert report["status"] == "passed"
    assert report["checks"][0]["name"] == "eval"
    assert report["eval_report"]["total_count"] == 5
    assert "started_at" in report
    assert "finished_at" in report


def test_cli_check_pytest_uses_runner_without_real_pytest(tmp_path: Path, capsys, monkeypatch) -> None:
    calls = []

    def fake_pytest(command, cwd):
        calls.append((command, cwd))
        return checks.PytestResult(
            command=list(command),
            exit_code=0,
            stdout="12 passed in 0.10s\n",
            stderr="",
        )

    monkeypatch.setattr(checks, "run_pytest_command", fake_pytest)

    exit_code = cli.main(["check", "--pytest", "--runs-root", str(tmp_path / ".runs")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == [(["uv", "run", "pytest", "-q"], Path.cwd())]
    assert "pytest_exit_code=0" in output
    assert "pytest_summary=12 passed in 0.10s" in output
