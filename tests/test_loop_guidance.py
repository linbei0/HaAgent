"""
tests/test_loop_guidance.py - Agent loop 工具建议测试

验证 suggestion_for_observation 只产生"建议"，不做强制干预。
"""

from __future__ import annotations

from haagent.runtime.loop_guidance import suggestion_for_observation


def _obs(tool_name: str, args: dict, result: dict) -> dict:
    return {"tool_name": tool_name, "args": args, "result": result}


# --- 成功路径：只对特定工具生成建议 ---

def test_file_search_success_suggests_file_read() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "file_search",
            {"query": "greet"},
            {"status": "success", "matches": [{"path": "src/app.py", "line": 1, "text": "def greet"}]},
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_success"
    assert "file_read" in suggestion.message
    assert "src/app.py" in suggestion.message


def test_file_search_no_results_suggests_refine() -> None:
    suggestion = suggestion_for_observation(
        _obs("file_search", {"query": "missing"}, {"status": "success", "matches": []})
    )

    assert suggestion is not None
    assert "file_list" in suggestion.message or "Refine" in suggestion.message


def test_context_find_success_suggests_read_candidate() -> None:
    suggestion = suggestion_for_observation(
        _obs(
            "context_find",
            {"query": "greeting"},
            {
                "status": "success",
                "candidates": [
                    {
                        "path": "src/app.py",
                        "line": 1,
                        "excerpt": "def greet",
                        "recommended_file_read": {"path": "src/app.py", "keyword": "greet", "limit": 80},
                    }
                ],
            },
        )
    )

    assert suggestion is not None
    assert "context_find" in suggestion.message
    assert "file_read" in suggestion.message
    assert "src/app.py" in suggestion.message


def test_context_find_empty_suggests_change_keywords() -> None:
    suggestion = suggestion_for_observation(
        _obs("context_find", {"query": "missing feature"}, {"status": "success", "candidates": []})
    )

    assert suggestion is not None
    assert "change keywords" in suggestion.message.lower() or "Change keywords" in suggestion.message
    assert "request_user_input" in suggestion.message


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
                "error": {"type": "tool_argument_invalid", "message": "path does not exist: app.py"},
                "suggestions": ["src/app.py"],
            },
        )
    )

    assert suggestion is not None
    assert suggestion.trigger == "tool_error"
    assert "src/app.py" in suggestion.message


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


def test_error_without_specific_handler_returns_none() -> None:
    """没有专门处理逻辑的错误不应该产生建议，让 SafetyGuard 处理。"""
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
