"""
tests/test_context_messages.py - 模型消息构造测试

验证工具结果进入模型消息前的可见内容选择规则。
"""

import json

from haagent.context.messages import build_tool_result_message


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
