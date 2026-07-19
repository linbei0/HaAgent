"""
tests/unit/runtime/test_loop_guidance.py - Agent loop 工具建议测试

验证 suggestion_for_observation 只产生"建议"，不做强制干预。
"""

from __future__ import annotations

from haagent.runtime.orchestration.loop_guidance import suggestion_for_observation


def _obs(tool_name: str, args: dict, result: dict) -> dict:
    return {"tool_name": tool_name, "args": args, "result": result}


# --- 成功路径：只对特定工具生成建议 ---

def test_grep_success_suggests_file_read() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "grep",
            {"pattern": "greet"},
            {"status": "success", "matches": [{"path": "src/app.py", "line": 1, "text": "def greet"}]},
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_success"
    assert "file_read" in suggestion.message
    assert "src/app.py" in suggestion.message


def test_grep_no_results_suggests_refine() -> None:
    suggestion = suggestion_for_observation(
        _obs("grep", {"pattern": "missing"}, {"status": "success", "matches": []})
    )

    assert suggestion is not None
    assert "file_list" in suggestion.message or "Refine" in suggestion.message


def test_grep_partial_no_results_does_not_claim_search_was_complete() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "grep",
            {"pattern": "missing"},
            {
                "status": "success",
                "matches": [],
                "partial": True,
                "guidance": "Narrow path or include and retry.",
            },
        )
    )

    assert suggestion is not None
    assert "incomplete" in suggestion.message.lower()
    assert "No matches found" not in suggestion.message


def test_grep_truncated_results_suggest_narrower_search() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "grep",
            {"pattern": "needle"},
            {
                "status": "success",
                "matches": [{"path": "src/app.py", "line": 1, "text": "needle"}],
                "truncated": True,
                "guidance": "Search results were truncated; narrow path or include and retry.",
            },
        )
    )

    assert suggestion is not None
    assert "truncated" in suggestion.message.lower()
    assert "path" in suggestion.message and "include" in suggestion.message


def test_grep_partial_results_with_matches_still_warns_search_is_incomplete() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "grep",
            {"pattern": "needle"},
            {
                "status": "success",
                "matches": [{"path": "src/app.py", "line": 1, "text": "needle"}],
                "partial": True,
                "guidance": "Search was incomplete; narrow path or include and retry.",
            },
        )
    )

    assert suggestion is not None
    assert "partial" in suggestion.message.lower()
    assert "path" in suggestion.message and "include" in suggestion.message


def test_file_write_success_suggests_read_back() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "file_write",
            {"path": "README.md"},
            {"status": "success", "path": "README.md"},
        )
    )

    assert suggestion is not None
    assert "README.md" in suggestion.message
    assert "verification" in suggestion.message or "reading back" in suggestion.message or "Read back" in suggestion.message


def test_request_user_input_success_suggests_continue() -> None:
    suggestion = suggestion_for_observation(
        _obs("request_user_input", {}, {"status": "success", "answer": "yes"})
    )

    assert suggestion is not None
    assert "same question" in suggestion.message or "continue" in suggestion.message.lower()


def test_file_read_success_produces_no_suggestion() -> None:
    """file_read 是中性操作，不应该产生建议来干预 Agent。"""
    suggestion = suggestion_for_observation(
        _obs(
            "file_read",
            {"path": "README.md"},
            {"status": "success", "path": "README.md", "content": "# Demo"},
        )
    )

    assert suggestion is None


def test_file_list_success_produces_no_suggestion() -> None:
    suggestion = suggestion_for_observation(
        _obs("file_list", {}, {"status": "success", "tree": "./"})
    )

    assert suggestion is None


# --- 错误路径：根据错误类型提供有用建议 ---

def test_file_read_error_with_suggestion_uses_suggested_path() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "file_read",
            {"path": "app.py"},
            {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "category": "argument", "message": "path does not exist: app.py", "retryable": False},
                "recovery": {"action": "use_tool", "message": "读取最相似的已存在文件。", "tool_name": "file_read", "args": {"path": "src/app.py"}},
            },
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_error"
    assert "src/app.py" in suggestion.message


def test_file_read_directory_error_suggests_file_list() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "file_read",
            {"path": "src"},
            {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "category": "argument", "message": "path must be a file: src", "retryable": False},
                "recovery": {"action": "use_tool", "message": "该路径是目录，请先列出目录内容。", "tool_name": "file_list", "args": {"path": "src", "max_depth": 1}},
            },
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_error"
    assert "file_list" in suggestion.message
    assert "src" in suggestion.message


def test_grep_file_path_success_suggests_file_read() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "grep",
            {"pattern": "needle", "path": "alpha.txt"},
            {
                "status": "success",
                "matches": [{"path": "alpha.txt", "line": 1, "text": "needle appears here"}],
            },
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_success"
    assert "file_read" in suggestion.message
    assert "alpha.txt" in suggestion.message


def test_file_list_missing_directory_suggests_parent_listing() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "file_list",
            {"path": "tools"},
            {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "category": "argument", "message": "path does not exist: tools", "retryable": False},
                "recovery": {"action": "use_tool", "message": "从最近存在的父目录重新定位目标。", "tool_name": "file_list", "args": {"path": ".", "max_depth": 2}},
            },
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_error"
    assert "file_list" in suggestion.message
    assert "max_depth" in suggestion.message


def test_apply_patch_miss_suggests_read_before_retry() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "apply_patch",
            {"path": "README.md", "old_text": "missing", "new_text": "new"},
            {
                "status": "error",
                "error": {"type": "patch_not_applied", "message": "old_text not found"},
            },
        )
    )

    assert suggestion is not None
    assert "README.md" in suggestion.message
    assert "narrow old_text" in suggestion.message


def test_apply_patch_set_not_unique_suggests_expand_context() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "apply_patch_set",
            {"replacements": [{"path": "README.md", "old_text": "same", "new_text": "new"}]},
            {
                "status": "error",
                "error": {"type": "patch_text_not_unique", "message": "old_text must match exactly once"},
                "replacement_count": 1,
                "replacements": [{"index": 0, "path": "README.md", "status": "error", "reason": "repeated"}],
            },
        )
    )

    assert suggestion is not None
    assert "README.md" in suggestion.message
    assert "unique" in suggestion.message.lower() or "longer" in suggestion.message.lower()


def test_unknown_execution_suggests_inspection_before_any_retry() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "shell",
            {},
            {
                "status": "error",
                "execution_state": "unknown",
                "error": {"type": "timeout", "message": "command timed out"},
            },
        )
    )

    assert suggestion is not None
    assert "Inspect" in suggestion.message
    assert "before retrying" in suggestion.message


def test_error_without_specific_handler_returns_none() -> None:
    """没有专门处理逻辑的错误不应伪造恢复建议。"""
    suggestion = suggestion_for_observation(
        _obs(
            "shell",
            {"command": "pytest"},
            {"status": "error", "exit_code": 1, "stderr": "AssertionError"},
        )
    )

    assert suggestion is None


# --- 关键回归测试：不再有强制终止逻辑 ---

def test_repeated_file_reads_never_force_stop() -> None:
    """重复读取文件不应该触发任何强制终止——这是最关键的回归测试。"""
    for _ in range(10):
        suggestion = suggestion_for_observation(
            _obs(
                "file_read",
                {"path": "README.md"},
                {"status": "success", "path": "README.md", "content": "# Demo"},
            )
        )
        assert suggestion is None, "file_read 不应该产生任何建议来干预 Agent"


def test_summary_task_with_read_only_tools_never_force_stop() -> None:
    """原始 bug 场景：'生成介绍文档' 不应该被强制终止。"""
    for tool_name in ["file_read", "file_read", "file_read", "file_list"]:
        suggestion = suggestion_for_observation(
            _obs(
                tool_name,
                {"path": "docs/something.md"} if tool_name == "file_read" else {},
                {"status": "success", "content": "..." if tool_name == "file_read" else ""},
            )
        )
        # file_read 和 file_list 都不应该产生建议
        if tool_name in {"file_read", "file_list"}:
            assert suggestion is None
