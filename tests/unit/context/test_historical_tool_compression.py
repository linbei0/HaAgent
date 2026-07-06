"""
tests/unit/context/test_historical_tool_compression.py - 历史工具消息压缩测试

验证 artifact-backed 工具结果按新旧分层保留或降级，长文本结果按统一策略折叠。
"""

import json

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.messages import compress_historical_tool_messages


def _artifact_tool_message(index: int) -> dict[str, object]:
    path = f".runs/episode/artifacts/tool-results/tool-{index}.txt"
    return {
        "role": "tool",
        "tool_call_id": f"call_{index}",
        "name": "mcp__fixture__fetch",
        "content": json.dumps(
            {
                "kind": "tool_result_view",
                "tool_name": "mcp__fixture__fetch",
                "status": "success",
                "content": f"preview-{index}",
                "content_format": "text",
                "artifact": {
                    "path": path,
                    "original_chars": 13000 + index,
                    "preview_chars": 3000,
                },
                "truncated": True,
                "continuation_hint": f"Use file_read with path={path}",
            },
            ensure_ascii=False,
        ),
    }


def test_recent_three_artifact_backed_tool_messages_keep_previews() -> None:
    messages = [_artifact_tool_message(index) for index in range(4)]

    diagnostics = compress_historical_tool_messages(messages, derive_compression_budget(None))

    kept_payloads = [json.loads(message["content"]) for message in messages[1:]]
    assert [payload["content"] for payload in kept_payloads] == ["preview-1", "preview-2", "preview-3"]
    assert json.loads(messages[0]["content"])["content_format"] == "summary"
    assert diagnostics[0].stage == "historical_tool_message"
    assert diagnostics[0].decision == "artifact_summary"
    assert diagnostics[0].reason == "older_artifact_result"


def test_older_artifact_backed_message_becomes_path_summary() -> None:
    messages = [_artifact_tool_message(index) for index in range(5)]

    compress_historical_tool_messages(messages, derive_compression_budget(None))

    payload = json.loads(messages[0]["content"])
    assert payload["kind"] == "tool_result_view"
    assert payload["content_format"] == "summary"
    assert payload["content"] == (
        "mcp__fixture__fetch result saved at .runs/episode/artifacts/tool-results/tool-0.txt "
        "(13000 chars). Use file_read with path=.runs/episode/artifacts/tool-results/tool-0.txt"
    )
    assert payload["artifact"]["path"] == ".runs/episode/artifacts/tool-results/tool-0.txt"
    assert "preview-0" not in messages[0]["content"]


def test_non_artifact_long_tool_result_collapses_with_head_and_tail() -> None:
    budget = derive_compression_budget(None)
    long_text = "head " + ("x" * budget.tool_output_inline_chars) + " tail"
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "shell",
            "content": long_text,
        },
    ]
    diagnostics = compress_historical_tool_messages(messages, budget)

    assert messages[0]["content"].startswith("head ")
    assert messages[0]["content"].endswith(" tail")
    assert "collapsed" in messages[0]["content"]
    assert diagnostics[0].stage == "historical_tool_message"
    assert diagnostics[0].decision == "collapsed"
    assert diagnostics[0].reason == "long_text_result"
    assert diagnostics[0].original_chars == len(long_text)
    assert diagnostics[0].final_chars == len(messages[0]["content"])


def test_historical_compression_no_old_reason_name() -> None:
    budget = derive_compression_budget(None)
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "shell",
            "content": "x" * (budget.tool_output_inline_chars + 1),
        },
    ]

    diagnostics = compress_historical_tool_messages(messages, budget)

    assert all(diagnostic.reason != "old_tool_result_over_budget" for diagnostic in diagnostics)
