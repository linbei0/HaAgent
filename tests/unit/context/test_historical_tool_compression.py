"""
tests/unit/context/test_historical_tool_compression.py - 历史工具消息压缩测试

验证 artifact-backed 工具结果按新旧分层保留或降级，长文本结果按统一策略折叠。
新增 KV-cache 稳定性测试：验证 append-only 前缀在连续轮次间保持不变。
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
                "truncated": True,
                "continuation_hint": f"Use file_read with path={path}",
            },
            ensure_ascii=False,
        ),
    }


def test_recent_three_artifact_backed_tool_messages_keep_previews() -> None:
    messages = [_artifact_tool_message(index) for index in range(4)]
    original_content_0 = messages[0]["content"]

    view, diagnostics = build_compressed_model_view(messages, derive_compression_budget(None))

    kept_payloads = [json.loads(message["content"]) for message in view[1:]]
    assert [payload["content"] for payload in kept_payloads] == ["preview-1", "preview-2", "preview-3"]
    assert json.loads(view[0]["content"])["content_format"] == "summary"
    assert diagnostics[0].stage == "historical_tool_message"
    assert diagnostics[0].decision == "artifact_summary"
    assert diagnostics[0].reason == "older_artifact_result"
    # 原始消息不被修改（append-only 保证）
    assert messages[0]["content"] == original_content_0


def test_older_artifact_backed_message_becomes_path_summary() -> None:
    messages = [_artifact_tool_message(index) for index in range(5)]
    original_content_0 = messages[0]["content"]

    view, _ = build_compressed_model_view(messages, derive_compression_budget(None))

    payload = json.loads(view[0]["content"])
    assert payload["kind"] == "tool_result_view"
    assert payload["content_format"] == "summary"
    assert payload["content"] == (
        "mcp__fixture__fetch result saved at .runs/episode/artifacts/tool-results/tool-0.txt "
        "(13000 chars). Use file_read with path=.runs/episode/artifacts/tool-results/tool-0.txt"
    )
    assert payload["artifact"]["path"] == ".runs/episode/artifacts/tool-results/tool-0.txt"
    assert "preview-0" not in view[0]["content"]
    # 原始消息不被修改
    assert messages[0]["content"] == original_content_0


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
    view, diagnostics = build_compressed_model_view(messages, budget)

    assert view[0]["content"].startswith("head ")
    assert view[0]["content"].endswith(" tail")
    assert "collapsed" in view[0]["content"]
    assert diagnostics[0].stage == "historical_tool_message"
    assert diagnostics[0].decision == "collapsed"
    assert diagnostics[0].reason == "long_text_result"
    assert diagnostics[0].original_chars == len(long_text)
    assert diagnostics[0].final_chars == len(view[0]["content"])
    # 原始消息不被修改
    assert messages[0]["content"] == long_text


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

    _, diagnostics = build_compressed_model_view(messages, budget)

    assert all(diagnostic.reason != "old_tool_result_over_budget" for diagnostic in diagnostics)


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

    # 验证: 视图中非 recent 窗口的消息在两轮间保持一致
    # Turn 1 只有 1 个 artifact（在 recent 窗口内），不压缩
    # Turn 2 有 4 个 artifact，第 1 个（index=3）离开 recent 窗口被压缩
    # 但 system/user/assistant 消息（index 0-2）始终不变
    for i in range(3):
        assert view_1[i] == view_2[i], f"prefix mismatch at index {i}"


def test_original_messages_never_mutated() -> None:
    """build_compressed_model_view 不修改原始消息列表中的任何元素。"""
    budget = derive_compression_budget(None)
    long_text = "y" * (budget.tool_output_inline_chars + 100)
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "c1", "name": "shell", "content": long_text},
        {"role": "tool", "tool_call_id": "c2", "name": "shell", "content": long_text},
    ]
    originals = [dict(m) for m in messages]

    view, diagnostics = build_compressed_model_view(messages, budget)

    assert len(diagnostics) == 2
    # 原始列表和元素均未被修改
    for i, (original, current) in enumerate(zip(originals, messages)):
        assert current == original, f"message at index {i} was mutated"
    # 视图中被压缩的消息是新对象
    assert view[1] is not messages[1]
    assert view[2] is not messages[2]
    # 未压缩消息保持原引用
    assert view[0] is messages[0]
