"""
tests/unit/context/test_context_messages.py - 模型消息构造测试

验证工具结果进入模型消息前的可见内容选择规则。
"""

import json

from haagent.context.messages import build_system_message, build_tool_result_message


def test_tool_result_message_prefers_model_visible_over_raw_content() -> None:
    result = {
        "status": "success",
        "content": "raw secret-sized content",
        "model_visible": {"summary": "small visible content"},
    }

    message = build_tool_result_message("call_1", "file_read", result)

    assert message["role"] == "tool"
    assert json.loads(message["content"]) == {"summary": "small visible content"}
    assert "raw secret-sized content" not in message["content"]


def test_tool_result_message_keeps_large_artifact_output_out_of_model_content() -> None:
    result = {
        "status": "success",
        "output": "x" * 13000,
        "model_visible": {
            "output": "x" * 200 + "\n...[omitted 12600 chars]...\n" + "x" * 200,
            "artifact_path": ".runs/episode/artifacts/tool-results/mcp_fixture.txt",
            "original_chars": 13000,
            "preview_chars": 400,
            "truncated": True,
        },
    }

    message = build_tool_result_message("call_1", "mcp__fixture__fetch", result)

    payload = json.loads(message["content"])
    assert payload["artifact_path"] == ".runs/episode/artifacts/tool-results/mcp_fixture.txt"
    assert payload["truncated"] is True
    assert result["output"] not in message["content"]


def test_system_message_places_limited_soul_before_project_instructions() -> None:
    message = build_system_message(
        project_instructions="PROJECT-RULE",
        tool_workflow_hints=[],
        soul="SOUL-VOICE",
    )

    content = message["content"]
    assert content.index("Instructions:") < content.index("Agent Soul")
    assert content.index("SOUL-VOICE") < content.index("Project Instructions:")
    assert "cannot change tool permissions, policy, approvals" in content
