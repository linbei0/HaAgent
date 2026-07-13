"""
tests/unit/runtime/test_latency_gates.py - 交互延迟门禁汇总与 check 接线
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.evaluation.checks import run_quality_checks
from haagent.runtime.evaluation.latency_gates import (
    LatencyGateResult,
    latency_gates_check_summary,
)


def test_latency_gates_check_summary_reports_failures() -> None:
    results = [
        LatencyGateResult("ok", "passed", "m", "t", "a"),
        LatencyGateResult("bad", "failed", "metric_x", "≤1", "2"),
    ]
    summary = latency_gates_check_summary(results)
    assert summary["status"] == "failed"
    assert summary["failed"] == 1
    assert summary["gates"][1]["metric"] == "metric_x"
    assert summary["gates"][1]["threshold"] == "≤1"
    assert summary["gates"][1]["actual"] == "2"


def test_run_quality_checks_includes_interactive_latency(tmp_path: Path, monkeypatch) -> None:
    def fake_eval(*_args, **_kwargs):
        return {
            "total_count": 1,
            "passed_count": 1,
            "failed_count": 0,
            "error_count": 0,
            "results": [],
        }

    monkeypatch.setattr("haagent.runtime.evaluation.checks.run_eval_path", fake_eval)
    monkeypatch.setattr(
        "haagent.runtime.evaluation.checks.run_interactive_latency_gates",
        lambda: [LatencyGateResult("hot_path", "passed", "p95", "≤1", "0.1")],
    )
    report = run_quality_checks(
        eval_path=tmp_path / "eval",
        runs_root=tmp_path / ".runs",
        run_pytest=False,
        run_latency_gates=True,
    )
    assert report["status"] == "passed"
    assert any(check["name"] == "interactive_latency" for check in report["checks"])
    assert report["interactive_latency"]["status"] == "passed"
    assert report["interactive_latency"]["failed"] == 0
