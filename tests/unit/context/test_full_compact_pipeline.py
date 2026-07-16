"""
tests/unit/context/test_full_compact_pipeline.py - 统一 full compact 流水线测试

验证确定性压缩后按 token 压力自动触发 full compact，并在失败时保留可用消息。
"""

import copy
import json

from haagent.context.compression.full import (
    FullCompactEligibility,
    maybe_full_compact_messages,
)
from haagent.models.types import ModelCallError, ModelResponse


class RecordingGateway:
    provider_name = "recording"

    def __init__(self, response: ModelResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def generate(self, invocation, **kwargs):
        self.calls.append({"messages": copy.deepcopy(invocation.messages), "tool_schemas": copy.deepcopy(invocation.tool_schemas)})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_full_compact_failure_records_reason_and_returns_original_messages() -> None:
    messages = [{"role": "user", "content": f"message-{index}"} for index in range(8)]
    original = copy.deepcopy(messages)

    result = maybe_full_compact_messages(
        messages=messages,
        eligibility=FullCompactEligibility(True, "high_pressure_after_deterministic_compression", "auto_full_compact", 2),
        gateway=RecordingGateway(ModelCallError("provider unavailable")),
        preserve_recent=2,
    )

    assert result.applied is False
    assert result.reason == "model_call_failed"
    assert result.messages == original
    assert result.manifest["reason"] == "model_call_failed"


def test_full_compact_preserves_tool_call_result_pair_boundary() -> None:
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
    gateway = RecordingGateway(ModelResponse(content=_valid_summary_json(), tool_calls=[]))

    result = maybe_full_compact_messages(
        messages=messages,
        eligibility=FullCompactEligibility(True, "high_pressure_after_deterministic_compression", "auto_full_compact", 2),
        gateway=gateway,
        preserve_recent=2,
    )

    assert result.applied is True
    assert result.older_message_count == 1
    assert result.preserved_recent_count == 3
    assert result.messages[-3:] == messages[1:]


def _valid_summary_json() -> str:
    return json.dumps(
        {
            "task_focus": "compact old messages",
            "completed_work": [],
            "open_issues": [],
            "important_files": [],
            "tool_results": [],
            "constraints": [],
            "verification": [],
            "risks": [],
        },
    )
