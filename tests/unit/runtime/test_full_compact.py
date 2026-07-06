"""
tests/unit/runtime/test_full_compact.py - full compact 执行测试

验证 full compact executor 的模型调用、消息重建、失败边界和 tool pair 保护。
"""

from __future__ import annotations

import copy
import json

from haagent.models.gateway import ModelCallError, ModelResponse, ToolCall
from haagent.context.compression.full import FullCompactEligibility, maybe_full_compact_messages


class RecordingGateway:
    provider_name = "recording"

    def __init__(self, response: ModelResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, messages, tool_schemas):
        self.calls.append({"messages": copy.deepcopy(messages), "tool_schemas": copy.deepcopy(tool_schemas)})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def eligible() -> FullCompactEligibility:
    return FullCompactEligibility(
        eligible=True,
        reason="full_compact_candidate_after_deterministic_compaction",
        trigger_kind="full_compact_candidate",
        required_preserve_recent=2,
    )


def ineligible() -> FullCompactEligibility:
    return FullCompactEligibility(
        eligible=False,
        reason="deterministic_context_sufficient",
        trigger_kind=None,
        required_preserve_recent=2,
    )


def valid_summary_json() -> str:
    return json.dumps(
        {
            "task_focus": "finish full compact",
            "completed_work": ["contract exists"],
            "open_issues": [],
            "important_files": ["src/haagent/runtime/full_compact.py"],
            "tool_results": ["pytest target planned"],
            "constraints": ["preserve recent messages"],
            "verification": ["uv run pytest tests/unit/runtime/test_full_compact.py -q"],
            "risks": [],
        },
    )


def test_full_compact_eligibility_false_does_not_call_gateway_and_preserves_messages() -> None:
    messages = [{"role": "user", "content": "older"}, {"role": "assistant", "content": "recent"}]
    original = copy.deepcopy(messages)
    gateway = RecordingGateway(ModelResponse(content=valid_summary_json(), tool_calls=[]))

    result = maybe_full_compact_messages(messages=messages, eligibility=ineligible(), gateway=gateway, preserve_recent=2)

    assert result.applied is False
    assert result.reason == "deterministic_context_sufficient"
    assert result.messages == original
    assert result.messages is not messages
    assert gateway.calls == []
    assert messages == original


def test_full_compact_success_calls_gateway_with_older_messages_and_rebuilds_messages() -> None:
    messages = [{"role": "user", "content": f"older-{index}"} for index in range(4)]
    messages.extend(
        [
            {"role": "user", "content": "recent-user"},
            {"role": "assistant", "content": "recent-assistant"},
        ],
    )
    original_recent = copy.deepcopy(messages[-2:])
    gateway = RecordingGateway(ModelResponse(content=valid_summary_json(), tool_calls=[]))

    result = maybe_full_compact_messages(messages=messages, eligibility=eligible(), gateway=gateway, preserve_recent=2)

    assert result.applied is True
    assert result.reason == "applied"
    assert result.pre_message_count == 6
    assert result.post_message_count == 4
    assert result.older_message_count == 4
    assert result.preserved_recent_count == 2
    assert result.summary_chars > 0
    assert gateway.calls[0]["tool_schemas"] == []
    prompt_text = "\n".join(message["content"] for message in gateway.calls[0]["messages"])
    assert "older-0" in prompt_text
    assert "older-3" in prompt_text
    assert "recent-user" not in prompt_text
    assert result.messages[0]["content"] == "[full_compact_boundary older_messages=4 preserved_recent=2]"
    assert result.messages[1]["content"].startswith("Full Compact Summary:\n")
    assert result.messages[-2:] == original_recent
    assert "older-0" not in "\n".join(str(message.get("content", "")) for message in result.messages)


def test_full_compact_preserves_tool_pair_when_boundary_would_split_it() -> None:
    messages = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "file_read", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "file_read", "content": "tool result"},
        {"role": "assistant", "content": "after tool"},
    ]
    gateway = RecordingGateway(ModelResponse(content=valid_summary_json(), tool_calls=[]))

    result = maybe_full_compact_messages(messages=messages, eligibility=eligible(), gateway=gateway, preserve_recent=2)

    assert result.applied is True
    assert result.older_message_count == 1
    assert result.preserved_recent_count == 3
    assert result.messages[-3:] == messages[1:]


def test_full_compact_schema_invalid_does_not_pollute_messages() -> None:
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(5)]
    original = copy.deepcopy(messages)
    gateway = RecordingGateway(ModelResponse(content=json.dumps({"task_focus": "missing fields"}), tool_calls=[]))

    result = maybe_full_compact_messages(messages=messages, eligibility=eligible(), gateway=gateway, preserve_recent=2)

    assert result.applied is False
    assert result.reason == "schema_invalid"
    assert result.messages == original
    assert messages == original
    assert "schema_errors" in result.manifest


def test_full_compact_rejects_invalid_json_tool_calls_and_model_failures_without_pollution() -> None:
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(5)]
    original = copy.deepcopy(messages)

    invalid_json = maybe_full_compact_messages(
        messages=messages,
        eligibility=eligible(),
        gateway=RecordingGateway(ModelResponse(content="{not-json", tool_calls=[])),
        preserve_recent=2,
    )
    with_tool_call = maybe_full_compact_messages(
        messages=messages,
        eligibility=eligible(),
        gateway=RecordingGateway(ModelResponse(content=valid_summary_json(), tool_calls=[ToolCall(name="x", args={})])),
        preserve_recent=2,
    )
    failed = maybe_full_compact_messages(
        messages=messages,
        eligibility=eligible(),
        gateway=RecordingGateway(ModelCallError("provider unavailable")),
        preserve_recent=2,
    )

    assert invalid_json.reason == "summary_json_invalid"
    assert with_tool_call.reason == "summary_returned_tool_calls"
    assert failed.reason == "model_call_failed"
    assert invalid_json.messages == original
    assert with_tool_call.messages == original
    assert failed.messages == original
    assert messages == original
