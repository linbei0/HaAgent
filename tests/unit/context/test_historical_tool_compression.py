"""
tests/unit/context/test_historical_tool_compression.py - 历史工具消息压缩测试

验证工具结果首次进入模型后保持不可变，历史层不再二次折叠长文本。
KV-cache 稳定性测试覆盖完整模型视图，而非只覆盖 system 前缀。
"""

import json

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.messages import build_compressed_model_view


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
                "truncation": {"occurred": True},
                "continuation_hint": f"Use file_read with path={path}",
            },
            ensure_ascii=False,
        ),
    }


def test_artifact_backed_tool_messages_keep_immutable_previews() -> None:
    messages = [_artifact_tool_message(index) for index in range(4)]

    view, diagnostics = build_compressed_model_view(messages, derive_compression_budget(None))

    assert view == messages
    assert diagnostics == []
    assert all(view[index] is messages[index] for index in range(len(messages)))


def test_raw_long_tool_result_is_not_recollapsed() -> None:
    budget = derive_compression_budget(None)
    long_text = "head " + ("x" * 50_000) + " tail"
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "shell",
            "content": long_text,
        },
    ]
    view, diagnostics = build_compressed_model_view(messages, budget)

    assert view == messages
    assert diagnostics == []


def test_historical_compression_no_old_reason_name() -> None:
    budget = derive_compression_budget(None)
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "shell",
            "content": "x" * 50_001,
        },
    ]

    view, diagnostics = build_compressed_model_view(messages, budget)

    assert view == messages
    assert diagnostics == []


def test_append_only_prefix_stability() -> None:
    """连续两轮，前缀消息内容不变（KV-cache 稳定性核心保证）。"""
    budget = derive_compression_budget(None)
    system = {"role": "system", "content": "You are helpful."}
    user = {"role": "user", "content": "Read the file"}
    assistant = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
    tool_result = _artifact_tool_message(0)

    messages_turn_1 = [system, user, assistant, tool_result]
    view_1, _ = build_compressed_model_view(messages_turn_1, budget)

    # Turn 2: 追加新消息（模拟 append-only 日志）
    assistant_2 = {"role": "assistant", "content": "", "tool_calls": [{"id": "c2"}]}
    tool_result_2 = _artifact_tool_message(1)
    tool_result_3 = _artifact_tool_message(2)
    tool_result_4 = _artifact_tool_message(3)
    messages_turn_2 = messages_turn_1 + [assistant_2, tool_result_2, tool_result_3, tool_result_4]
    view_2, _ = build_compressed_model_view(messages_turn_2, budget)

    # 验证: 原始消息未被修改
    assert messages_turn_2[3] is tool_result  # 同一对象引用
    assert messages_turn_2[3]["content"] == tool_result["content"]

    # 完整旧视图必须成为新视图的精确前缀，工具结果不能在后续轮次改写。
    assert view_2[: len(view_1)] == view_1


def test_original_messages_never_mutated() -> None:
    """build_compressed_model_view 不修改原始消息列表中的任何元素。"""
    budget = derive_compression_budget(None)
    long_text = "y" * 50_100
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "c1", "name": "shell", "content": long_text},
        {"role": "tool", "tool_call_id": "c2", "name": "shell", "content": long_text},
    ]
    originals = [dict(m) for m in messages]

    view, diagnostics = build_compressed_model_view(messages, budget)

    assert diagnostics == []
    assert view == messages
    # 原始列表和元素均未被修改
    for i, (original, current) in enumerate(zip(originals, messages)):
        assert current == original, f"message at index {i} was mutated"
    assert view[1] is messages[1]
    assert view[2] is messages[2]
    assert view[0] is messages[0]
