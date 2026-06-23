"""
tests/test_observation_compaction.py - 工具 observation 压缩测试

验证工具 observation 的模型输入摘要保持紧凑且不包含完整大输出。
"""

from __future__ import annotations

import json

from haagent.context.observation_compaction import observation_summary


def _summary_json(observation: dict[str, object]) -> str:
    return json.dumps(observation_summary(observation), ensure_ascii=False, sort_keys=True)


def test_file_read_summary_keeps_bounded_excerpt() -> None:
    content = "file-read-start\n" + ("x" * 600) + "\nFILE_READ_TAIL"
    summary = observation_summary(
        {
            "tool_name": "file_read",
            "args": {"path": "notes.txt", "offset": 3, "limit": 20},
            "result": {"status": "success", "content": content},
        },
    )

    summary_json = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    assert summary["path"] == "notes.txt"
    assert summary["start_line"] == 4
    assert summary["truncated"] is True
    assert "file-read-start" in summary_json
    assert "FILE_READ_TAIL" not in summary_json
    assert content not in summary_json


def test_write_and_code_run_summaries_omit_full_input_payloads() -> None:
    secret_content = "FULL_WRITE_CONTENT_SHOULD_NOT_ENTER_MODEL"
    code = "print('CODE_SHOULD_NOT_ENTER_MODEL')"

    file_write_summary = _summary_json(
        {
            "tool_name": "file_write",
            "args": {"path": "notes.txt", "content": secret_content, "mode": "overwrite"},
            "result": {"status": "success", "bytes_written": 42, "created": False},
        },
    )
    code_run_summary = _summary_json(
        {
            "tool_name": "code_run",
            "args": {"code": code, "timeout_seconds": 5},
            "result": {
                "status": "error",
                "exit_code": 2,
                "stdout_excerpt": "stdout-start-" + ("o" * 600),
                "stderr_excerpt": "stderr-start-" + ("e" * 600),
                "script_path": ".haagent-tmp/code-run.py",
            },
        },
    )

    assert secret_content not in file_write_summary
    assert '"bytes_written": 42' in file_write_summary
    assert code not in code_run_summary
    assert '"script_path": ".haagent-tmp/code-run.py"' in code_run_summary
    assert "o" * 400 not in code_run_summary


def test_shell_and_request_user_input_summaries_are_bounded() -> None:
    stdout = "stdout-start-" + ("o" * 600) + "-STDOUT-TAIL"
    stderr = "stderr-start-" + ("e" * 600) + "-STDERR-TAIL"
    answer = "answer-start-" + ("a" * 600) + "-ANSWER-TAIL"

    shell_summary = _summary_json(
        {
            "tool_name": "shell",
            "args": {"command": "uv run pytest -q", "cwd": "."},
            "result": {"status": "error", "exit_code": 1, "stdout": stdout, "stderr": stderr},
        },
    )
    user_input_summary = _summary_json(
        {
            "tool_name": "request_user_input",
            "args": {"question": "Which file?", "reason": "Need target"},
            "result": {"status": "success", "answer": answer},
        },
    )

    assert "stdout-start-" in shell_summary
    assert "-STDOUT-TAIL" not in shell_summary
    assert "-STDERR-TAIL" not in shell_summary
    assert stdout not in shell_summary
    assert stderr not in shell_summary
    assert "answer-start-" in user_input_summary
    assert "-ANSWER-TAIL" not in user_input_summary
    assert '"answer_chars": 625' in user_input_summary


def test_patch_summaries_do_not_include_complete_patch_payloads() -> None:
    old_text = "old-start-" + ("o" * 600) + "-OLD-TAIL"
    new_text = "new-start-" + ("n" * 600) + "-NEW-TAIL"

    apply_patch_summary = _summary_json(
        {
            "tool_name": "apply_patch",
            "args": {"path": "app.py", "old_text": old_text, "new_text": new_text},
            "result": {"status": "success", "replacements": 1},
        },
    )
    apply_patch_set_summary = _summary_json(
        {
            "tool_name": "apply_patch_set",
            "args": {
                "replacements": [
                    {
                        "path": "app.py",
                        "old_text": "SECRET_OLD_TEXT_SHOULD_NOT_ENTER_MODEL",
                        "new_text": "SECRET_NEW_TEXT_SHOULD_NOT_ENTER_MODEL",
                    },
                ],
            },
            "result": {
                "status": "error",
                "replacement_count": 1,
                "error": {"type": "patch_text_not_found", "message": "old_text was not found"},
            },
        },
    )

    assert '"old_text_length": 619' in apply_patch_summary
    assert '"new_text_length": 619' in apply_patch_summary
    assert "-OLD-TAIL" not in apply_patch_summary
    assert "-NEW-TAIL" not in apply_patch_summary
    assert old_text not in apply_patch_summary
    assert new_text not in apply_patch_summary
    assert "SECRET_OLD_TEXT_SHOULD_NOT_ENTER_MODEL" not in apply_patch_set_summary
    assert "SECRET_NEW_TEXT_SHOULD_NOT_ENTER_MODEL" not in apply_patch_set_summary


def test_search_list_and_context_find_summaries_limit_repeated_results() -> None:
    matches = [
        {"path": "notes.txt", "line": index + 1, "column": 1, "text": f"match-{index:03d} " + ("m" * 40)}
        for index in range(30)
    ]
    tree = "\n".join(f"src/file_{index:03d}.py" for index in range(80))
    candidates = [
        {
            "path": f"src/file_{index:03d}.py",
            "line": index + 1,
            "excerpt": f"candidate-{index:03d} " + ("c" * 300),
        }
        for index in range(8)
    ]

    file_search_summary = _summary_json(
        {"tool_name": "file_search", "args": {"query": "match"}, "result": {"status": "success", "matches": matches}},
    )
    file_list_summary = _summary_json(
        {"tool_name": "file_list", "args": {"path": "."}, "result": {"status": "success", "tree": tree}},
    )
    context_find_summary = _summary_json(
        {
            "tool_name": "context_find",
            "args": {"query": "candidate"},
            "result": {"status": "success", "candidates": candidates},
        },
    )

    assert '"match_count": 30' in file_search_summary
    assert "match-029" not in file_search_summary
    assert "src/file_079.py" not in file_list_summary
    assert "candidate-000" in context_find_summary
    assert "candidate-007" not in context_find_summary
