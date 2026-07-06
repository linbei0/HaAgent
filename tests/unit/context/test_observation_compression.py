"""
 tests/unit/context/test_observation_compression.py - 工具 observation 压缩测试

验证工具 observation 的确定性折叠和机器可读压缩记录。
"""

from __future__ import annotations

from haagent.context.compression.sections import compact_observation_with_record


def test_long_observation_collapses_with_record() -> None:
    raw_output = "HEAD-" + ("middle-" * 80) + "TAIL"
    observation = {
        "tool_name": "shell",
        "args": {"command": "pytest"},
        "result": {"status": "success", "stdout": raw_output, "stderr": ""},
    }

    compacted, record = compact_observation_with_record(
        observation,
        max_chars=80,
        head_chars=20,
        tail_chars=20,
    )

    assert compacted.startswith('{"status": "success"')
    assert "HEAD-" in compacted
    assert "TAIL" in compacted
    assert "...[collapsed " in compacted
    assert record.tool_name == "shell"
    assert record.kind == "observation"
    assert record.decision == "collapsed"
    assert record.reason == "observation_over_budget"
    assert record.original_chars > record.final_chars


def test_short_observation_is_selected_without_changes() -> None:
    observation = {
        "tool_name": "file_write",
        "args": {"path": "note.txt", "mode": "create"},
        "result": {"status": "success", "path": "note.txt", "bytes_written": 5},
    }

    compacted, record = compact_observation_with_record(observation, max_chars=500)

    assert compacted == (
        '{"status": "success", "path": "note.txt", "mode": "create", '
        '"bytes_written": 5, "created": null, "truncated": false}'
    )
    assert record.tool_name == "file_write"
    assert record.decision == "selected"
    assert record.reason == "within_budget"
    assert record.original_chars == record.final_chars
