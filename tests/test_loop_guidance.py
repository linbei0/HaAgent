"""
tests/test_loop_guidance.py - Agent loop 推进策略测试

验证工具结果和 no-tool 回复会生成短小、可审计的下一步 guidance。
"""

from __future__ import annotations

from haagent.runtime.loop_guidance import (
    LoopGuidanceState,
    guidance_for_no_tool_response,
    guidance_for_observation,
)


def test_guidance_for_successful_file_search_selects_file_to_read() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "file_search",
            "args": {"query": "greet"},
            "result": {
                "status": "success",
                "matches": [{"path": "src/app.py", "line": 1, "text": "def greet"}],
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert guidance.status == "continue"
    assert "file_read" in guidance.message
    assert "src/app.py" in guidance.message


def test_repeated_read_only_file_read_requires_final_answer() -> None:
    state = LoopGuidanceState()

    first = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "README.md"},
            "result": {"status": "success", "path": "README.md", "content": "# Demo"},
        },
        state,
        goal="介绍一下项目",
    )
    second = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "README.md"},
            "result": {"status": "success", "path": "README.md", "content": "# Demo"},
        },
        state,
        goal="介绍一下项目",
    )

    assert first is not None
    assert first.status == "continue"
    assert second is not None
    assert second.status == "final_answer_required"
    assert second.trigger == "repeated_read_only_exploration"
    assert "final answer" in second.message
    assert "Do not call tools" in second.message
    assert "do not repeat" in second.message


def test_repeated_read_only_guidance_does_not_interrupt_edit_tasks() -> None:
    state = LoopGuidanceState()

    guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "README.md"},
            "result": {"status": "success", "path": "README.md", "content": "# Demo"},
        },
        state,
        goal="修改 README.md",
    )
    second = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "README.md"},
            "result": {"status": "success", "path": "README.md", "content": "# Demo"},
        },
        state,
        goal="修改 README.md",
    )

    assert second is not None
    assert second.status == "continue"


def test_guidance_for_successful_context_find_selects_candidate_to_read() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "context_find",
            "args": {"query": "greeting"},
            "result": {
                "status": "success",
                "candidates": [
                    {
                        "path": "src/app.py",
                        "line": 1,
                        "excerpt": "def greet",
                        "recommended_file_read": {"path": "src/app.py", "keyword": "greet", "limit": 80},
                    },
                ],
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert guidance.status == "continue"
    assert "context_find" in guidance.message
    assert "file_read" in guidance.message
    assert "src/app.py" in guidance.message


def test_guidance_for_file_read_after_file_change_pushes_final_answer() -> None:
    state = LoopGuidanceState(has_file_change=True)

    guidance = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "README.md"},
            "result": {
                "status": "success",
                "path": "README.md",
                "content": "# Demo\n\nTiny demo.\n\nTiny project.\n",
            },
        },
        state,
    )

    assert guidance is not None
    assert "If the read-back content satisfies the request" in guidance.message
    assert "produce the final answer" in guidance.message
    assert "do not keep editing" in guidance.message


def test_guidance_for_empty_context_find_changes_keywords_or_asks_user() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "context_find",
            "args": {"query": "missing feature"},
            "result": {"status": "success", "candidates": []},
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert guidance.status == "continue"
    assert "change keywords" in guidance.message
    assert "request_user_input" in guidance.message


def test_guidance_for_missing_file_prefers_suggestion_path() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "app.py"},
            "result": {
                "status": "error",
                "error": {
                    "type": "tool_argument_invalid",
                    "message": "path does not exist: app.py",
                },
                "suggestions": ["src/app.py"],
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert guidance.status == "handle_error"
    assert guidance.message == "File path failed; try the suggested path with file_read: src/app.py."


def test_guidance_for_patch_miss_reads_current_file_before_retry() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "apply_patch",
            "args": {"path": "README.md", "old_text": "missing", "new_text": "new"},
            "result": {
                "status": "error",
                "error": {"type": "patch_not_applied", "message": "old_text not found"},
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert "file_read README.md" in guidance.message
    assert "narrow old_text" in guidance.message


def test_guidance_for_patch_set_failure_reads_current_file_before_retry() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "apply_patch_set",
            "args": {
                "replacements": [
                    {"path": "README.md", "old_text": "missing", "new_text": "new"},
                    {"path": "src/app.py", "old_text": "x", "new_text": "y"},
                ],
            },
            "result": {
                "status": "error",
                "error": {"type": "patch_text_not_found", "message": "old_text was not found"},
                "replacement_count": 2,
                "replacements": [
                    {"index": 0, "path": "README.md", "status": "error", "reason": "old_text was not found"},
                    {"index": 1, "path": "src/app.py", "status": "skipped"},
                ],
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert "file_read README.md" in guidance.message
    assert "then retry apply_patch_set" in guidance.message


def test_guidance_for_patch_set_duplicate_match_expands_context() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "apply_patch_set",
            "args": {"replacements": [{"path": "README.md", "old_text": "same", "new_text": "new"}]},
            "result": {
                "status": "error",
                "error": {"type": "patch_text_not_unique", "message": "old_text must match exactly once"},
                "replacement_count": 1,
                "replacements": [
                    {"index": 0, "path": "README.md", "status": "error", "reason": "old_text repeated"}
                ],
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert "file_read README.md" in guidance.message
    assert "expand old_text context" in guidance.message


def test_guidance_for_shell_failure_uses_output_without_mechanical_retry() -> None:
    guidance = guidance_for_observation(
        {
            "tool_name": "shell",
            "args": {"command": "pytest -q"},
            "result": {
                "status": "error",
                "exit_code": 1,
                "stdout": "x" * 1000,
                "stderr": "AssertionError: bad value",
            },
        },
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert "Use stderr/stdout" in guidance.message
    assert "do not rerun the same command unchanged" in guidance.message
    assert "x" * 300 not in guidance.message


def test_consecutive_failures_require_new_strategy_or_user_input() -> None:
    state = LoopGuidanceState()
    first = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "missing.py"},
            "result": {"status": "error", "error": {"type": "tool_argument_invalid", "message": "missing"}},
        },
        state,
    )
    second = guidance_for_observation(
        {
            "tool_name": "file_read",
            "args": {"path": "missing.py"},
            "result": {"status": "error", "error": {"type": "tool_argument_invalid", "message": "missing"}},
        },
        state,
    )

    assert first is not None
    assert second is not None
    assert "Do not repeat the same failing tool call" in second.message
    assert "request_user_input" in second.message


def test_no_tool_review_pushes_file_modification_to_tools() -> None:
    guidance = guidance_for_no_tool_response(
        "Here is the code you should put in README.md:\n```markdown\nupdated\n```",
        "修改 README 文件",
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert guidance.status == "continue"
    assert "file_write/apply_patch" in guidance.message


def test_no_tool_review_pushes_unverified_completion_to_validation() -> None:
    guidance = guidance_for_no_tool_response(
        "Done, tests pass.",
        "修改 Python 文件并运行测试",
        LoopGuidanceState(),
    )

    assert guidance is not None
    assert "verify" in guidance.message
    assert "shell/code_run" in guidance.message


def test_no_tool_review_allows_normal_final_answer() -> None:
    guidance = guidance_for_no_tool_response(
        "Project has src/app.py and tests/test_app.py.",
        "总结项目结构",
        LoopGuidanceState(),
    )

    assert guidance is None
