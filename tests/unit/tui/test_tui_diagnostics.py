"""
tests/unit/tui/test_tui_diagnostics.py - TUI 退出诊断的隐私与轮转合同

验证诊断仅保留生命周期与环境摘要，不能把聊天或凭据写进用户级日志。
"""

from __future__ import annotations

import json

from haagent.tui.application.diagnostics import TuiDiagnostics


def test_normal_exit_records_only_whitelisted_summary(tmp_path) -> None:
    diagnostics = TuiDiagnostics(config_dir=tmp_path)

    diagnostics.record_started()
    diagnostics.record_stopped(active_markdown_writers=2, pending_markdown_fragments=3)

    records = [json.loads(line) for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()]

    assert [record["event"] for record in records] == ["tui_started", "tui_stopped"]
    assert records[-1]["active_markdown_writers"] == 2
    assert records[-1]["pending_markdown_fragments"] == 3
    assert "prompt" not in records[-1]
    assert "workspace" not in records[-1]


def test_exception_diagnostic_excludes_sensitive_exception_text(tmp_path) -> None:
    diagnostics = TuiDiagnostics(config_dir=tmp_path)
    prompt = "用户请求：整理 C:\\private\\report.md"
    api_key = "sk-live-this-must-not-be-logged"

    try:
        raise RuntimeError(f"{prompt}; api_key={api_key}")
    except RuntimeError as error:
        diagnostics.record_unhandled_exception(error)

    content = diagnostics.log_path.read_text(encoding="utf-8")

    assert "unhandled_exception" in content
    assert "RuntimeError" in content
    assert prompt not in content
    assert api_key not in content
    assert "private\\report.md" not in content


def test_diagnostics_rotate_before_exceeding_maximum_size(tmp_path) -> None:
    diagnostics = TuiDiagnostics(config_dir=tmp_path, max_bytes=80, backup_count=1)

    diagnostics.record_started()
    diagnostics.record_stopped(active_markdown_writers=10, pending_markdown_fragments=20)

    assert diagnostics.log_path.exists()
    assert diagnostics.log_path.with_suffix(".log.1").exists()
