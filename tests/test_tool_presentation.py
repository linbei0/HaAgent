from __future__ import annotations

from haagent.tools.presentation import summarize_tool_args, summarize_tool_result


def test_summarizes_file_read_args_and_result() -> None:
    args = {
        "path": "docs/" + ("nested/" * 30) + "guide.md",
        "offset": 10,
        "limit": 40,
        "keyword": "important\nkeyword",
    }
    result = {
        "path": "README.md",
        "start_line": 1,
        "end_line": 12,
        "line_count": 12,
        "truncated": "",
    }

    assert summarize_tool_args("file_read", args) == {
        "path": ("docs/" + ("nested/" * 30) + "guide.md")[:160] + "... [truncated]",
        "offset": 10,
        "limit": 40,
        "keyword": "important keyword",
    }
    assert summarize_tool_result("file_read", result) == {
        "path": "README.md",
        "start_line": 1,
        "end_line": 12,
        "line_count": 12,
        "truncated": False,
    }


def test_summarizes_file_write_args_and_result() -> None:
    assert summarize_tool_args(
        "file_write",
        {"path": "notes/today.md", "content": "hello\nworld", "mode": "append"},
    ) == {
        "content_chars": 11,
        "mode": "append",
        "path": "notes/today.md",
    }
    assert summarize_tool_result(
        "file_write",
        {"path": "notes/today.md", "mode": "append", "bytes_written": 11, "created": True},
    ) == {
        "path": "notes/today.md",
        "mode": "append",
        "bytes_written": 11,
        "created": True,
    }


def test_summarizes_apply_patch_args_and_result() -> None:
    assert summarize_tool_args(
        "apply_patch",
        {"path": "src/app.py", "old_text": "old", "new_text": "newer"},
    ) == {
        "new_text_chars": 5,
        "old_text_chars": 3,
        "path": "src/app.py",
    }
    assert summarize_tool_result("apply_patch", {"path": "src/app.py", "replacements": 2}) == {
        "path": "src/app.py",
        "replacements": 2,
    }


def test_summarizes_apply_patch_set_args_and_result() -> None:
    assert summarize_tool_args(
        "apply_patch_set",
        {
            "replacements": [
                {"path": "src/a.py", "old_text": "a", "new_text": "b"},
                {"path": "src/b.py", "old_text": "c", "new_text": "d"},
                "invalid",
            ]
        },
    ) == {"replacement_count": 3, "paths": ["src/a.py", "src/b.py"]}
    assert summarize_tool_result(
        "apply_patch_set",
        {"paths": ["src/a.py", "src/b.py"], "replacement_count": 2},
    ) == {"paths": ["src/a.py", "src/b.py"], "replacement_count": 2}


def test_summarizes_code_run_and_shell_output_excerpts() -> None:
    long_stdout = "x" * 301
    code_result = {
        "exit_code": 0,
        "stdout_excerpt": long_stdout,
        "stderr_excerpt": "",
        "truncated": "yes",
    }
    shell_result = {
        "exit_code": 1,
        "stdout_excerpt": "ok",
        "stderr_excerpt": "error\nline",
        "timeout": "",
        "truncated": False,
    }

    assert summarize_tool_args(
        "code_run",
        {"code": "print('hello')", "cwd": ".", "timeout_seconds": 5},
    ) == {"code_chars": 14, "cwd": ".", "timeout_seconds": 5}
    assert summarize_tool_result("code_run", code_result) == {
        "exit_code": 0,
        "stdout_excerpt": "x" * 300 + "... [truncated]",
        "stderr_excerpt": "none",
        "stdout_chars": 301,
        "stderr_chars": 0,
        "truncated": True,
    }
    assert summarize_tool_args(
        "shell",
        {"command": "uv run pytest", "cwd": ".", "timeout_seconds": 30},
    ) == {"command": "uv run pytest", "cwd": ".", "timeout_seconds": 30}
    assert summarize_tool_result("shell", shell_result) == {
        "exit_code": 1,
        "stdout_excerpt": "ok",
        "stderr_excerpt": "error line",
        "stdout_chars": 2,
        "stderr_chars": 10,
        "timeout": False,
        "truncated": False,
    }


def test_summarizes_request_user_input_args() -> None:
    assert summarize_tool_args(
        "request_user_input",
        {"question": "Proceed?\nPlease confirm.", "reason": ""},
    ) == {"question": "Proceed? Please confirm.", "reason": ""}


def test_unknown_tool_uses_key_fallbacks() -> None:
    assert summarize_tool_args("unknown_tool", {"z": 1, "a": 2}) == {"args_keys": ["a", "z"]}
    assert summarize_tool_result("unknown_tool", {"status": "ok", "value": 3}) == {
        "status": "ok",
        "result_keys": ["status", "value"],
    }
