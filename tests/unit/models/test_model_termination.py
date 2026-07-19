"""
tests/unit/models/test_model_termination.py - provider 停止原因归一化测试
"""

from haagent.models.adapters.anthropic import _parse_anthropic_response
from haagent.models.adapters.google import _parse_gemini_response
from haagent.models.adapters.openai_chat import _parse_chat_completion_response
from haagent.models.adapters.openai_responses import _openai_responses_termination


def test_openai_chat_preserves_completed_length_and_content_filter_reasons() -> None:
    def parse(reason: str) -> str:
        return _parse_chat_completion_response(
            {"choices": [{"message": {"content": "answer"}, "finish_reason": reason}]},
        ).termination

    assert parse("stop") == "completed"
    assert parse("length") == "length"
    assert parse("content_filter") == "content_filter"


def test_anthropic_and_gemini_preserve_truncated_reasons() -> None:
    anthropic = _parse_anthropic_response(
        {"content": [{"type": "text", "text": "answer"}], "stop_reason": "max_tokens"},
    )
    gemini = _parse_gemini_response(
        {
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "MAX_TOKENS"},
            ],
        },
    )

    assert anthropic.termination == "length"
    assert gemini.termination == "length"


def test_tool_calls_override_provider_stop_reason_and_responses_marks_incomplete_output() -> None:
    chat = _parse_chat_completion_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": "file_read", "arguments": "{}"},
                            },
                        ],
                    },
                    "finish_reason": "stop",
                },
            ],
        },
    )

    assert chat.termination == "tool_calls"
    assert _openai_responses_termination(
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        False,
    ) == "length"
