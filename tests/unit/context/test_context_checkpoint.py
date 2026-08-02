"""
tests/unit/context/test_context_checkpoint.py - 模型上下文 checkpoint 测试

验证 checkpoint 只在足够回收量下整段替换历史，保护工具调用配对并保留可回读证据。
"""

import json

from haagent.context.compression.budget import CompressionBudget
from haagent.context.compression.checkpoint import checkpoint_payload, maybe_checkpoint_messages


def _budget() -> CompressionBudget:
    return CompressionBudget(
        context_window_tokens=22_000,
        reserved_output_tokens=1_000,
        safety_buffer_tokens=1_000,
        available_input_tokens=20_000,
        context_builder_max_tokens=4_000,
    )


def _tool_pair(index: int, *, artifact: bool = True, size: int = 50_000) -> list[dict[str, object]]:
    tool_name = "shell"
    content = f"result-{index}-" + ("x" * size)
    artifact_value = (
        {
            "path": f".runs/episode/artifacts/tool-results/result-{index}.txt",
            "original_chars": len(content),
            "preview_chars": len(content),
        }
        if artifact
        else None
    )
    return [
        {
            "role": "assistant",
            "content": f"running tool {index}",
            "tool_calls": [{"id": f"call-{index}", "function": {"name": tool_name, "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "tool_call_id": f"call-{index}",
            "name": tool_name,
            "content": json.dumps(
                {
                    "kind": "tool_result_view",
                    "tool_name": tool_name,
                    "status": "success",
                    "content": content,
                    "content_format": "text",
                    "artifact": artifact_value,
                    "truncation": {
                        "occurred": artifact,
                        "reason": "model_input_budget" if artifact else None,
                        "original_chars": len(content),
                        "visible_chars": len(content),
                        "omitted_chars": 0,
                        "original_bytes": len(content.encode("utf-8")),
                        "visible_bytes": len(content.encode("utf-8")),
                        "omitted_bytes": 0,
                        "estimated_original_tokens": None,
                        "estimated_visible_tokens": None,
                        "estimated_omitted_tokens": None,
                        "artifact_path": artifact_value["path"] if artifact_value else None,
                        "recovery_hint": None,
                    },
                    "continuation_hint": None,
                },
            ),
        },
    ]


def test_checkpoint_replaces_one_contiguous_prefix_and_preserves_recent_tool_pairs() -> None:
    prefix = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: inspect the project"},
    ]
    messages = [*prefix]
    for index in range(1, 7):
        messages.extend(_tool_pair(index))
    original = json.loads(json.dumps(messages))

    result = maybe_checkpoint_messages(
        messages=messages,
        budget=_budget(),
        epoch=0,
        artifact_writer=lambda tool_name, content: f"artifacts/{tool_name}-{len(content)}.txt",
    )

    assert result.applied is True
    assert result.epoch == 1
    assert result.messages[:2] == prefix
    checkpoint = checkpoint_payload(result.messages[2])
    assert checkpoint is not None
    assert checkpoint["kind"] == "context_checkpoint"
    assert checkpoint["epoch"] == 1
    assert checkpoint["compacted_message_count"] >= 2
    assert checkpoint["artifact_refs"]
    # 最近消息原样保留，边界不能拆开 assistant/tool 对。
    assert result.messages[-4:] == original[-4:]
    assert result.diagnostic["tokens_reclaimed"] > 0
    assert messages == original


def test_checkpoint_preserves_token_budgeted_tail_and_marks_history_as_system_context() -> None:
    budget = CompressionBudget(
        context_window_tokens=80_000,
        reserved_output_tokens=4_000,
        safety_buffer_tokens=4_000,
        available_input_tokens=72_000,
        context_builder_max_tokens=14_000,
        checkpoint_preserve_recent_messages=6,
        checkpoint_preserve_recent_tokens=12_000,
    )
    prefix = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: inspect the project"},
    ]
    messages = [*prefix]
    for index in range(20):
        messages.extend(_tool_pair(index, size=16_000))
    original = json.loads(json.dumps(messages))

    result = maybe_checkpoint_messages(
        messages=messages,
        budget=budget,
        epoch=0,
        artifact_writer=lambda tool_name, content: f"artifacts/{tool_name}-{len(content)}.txt",
    )

    assert result.applied is True
    assert result.messages[2]["role"] == "system"
    assert result.messages[2]["content"].startswith("Runtime historical context checkpoint")
    # 尾部预算至少保留 6 条，并完整保留最近工具配对。
    preserved_recent = result.messages[3:]
    assert len(preserved_recent) >= 6
    assert result.messages[-len(preserved_recent) :] == original[-len(preserved_recent) :]


def test_checkpoint_materializes_inline_tool_output_before_forgetting_it() -> None:
    messages = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: inspect the project"},
    ]
    for index in range(1, 7):
        messages.extend(_tool_pair(index, artifact=False))
    saved: list[tuple[str, str]] = []

    result = maybe_checkpoint_messages(
        messages=messages,
        budget=_budget(),
        epoch=0,
        artifact_writer=lambda tool_name, content: (
            saved.append((tool_name, content)) or f"artifacts/{tool_name}-{len(saved)}.txt"
        ),
    )

    assert result.applied is True
    # artifact=None 的 ToolResultView 已经完成模型投影，checkpoint 不再二次落盘。
    assert saved == []
    checkpoint = checkpoint_payload(result.messages[2])
    assert checkpoint is not None
    assert all(ref["path"].startswith("artifacts/") for ref in checkpoint["artifact_refs"])


def test_checkpoint_does_not_run_without_pressure() -> None:
    messages = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: small task"},
        *_tool_pair(1, size=3_000),
    ]

    result = maybe_checkpoint_messages(
        messages=messages,
        budget=_budget(),
        epoch=0,
        artifact_writer=lambda tool_name, content: "unused.txt",
    )

    assert result.applied is False
    assert result.epoch == 0
    assert result.messages == messages
    assert result.diagnostic["reason"] == "within_input_budget"


def test_checkpoint_skips_small_reclaim_after_cache_generation_change() -> None:
    messages = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: inspect the project"},
    ]
    for index in range(12):
        content = f"result-{index}-" + ("x" * 3_000)
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "running",
                    "tool_calls": [
                        {"id": f"call-{index}", "function": {"name": "shell", "arguments": "{}"}},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "name": "shell",
                    "content": json.dumps(
                        {
                            "kind": "tool_result_view",
                            "tool_name": "shell",
                            "status": "success",
                            "content": content,
                            "content_format": "text",
                            "artifact": {
                                "path": f"artifacts/result-{index}.txt",
                                "original_chars": len(content),
                                "preview_chars": len(content),
                            },
                            "truncation": {"occurred": True},
                        },
                    ),
                },
            ],
        )

    small_budget = CompressionBudget(
        context_window_tokens=7_000,
        reserved_output_tokens=1_000,
        safety_buffer_tokens=1_000,
        available_input_tokens=5_000,
        context_builder_max_tokens=1_000,
    )
    result = maybe_checkpoint_messages(
        messages=messages,
        budget=small_budget,
        epoch=0,
        artifact_writer=lambda tool_name, content: f"artifacts/{tool_name}.txt",
    )

    assert result.applied is False
    assert result.diagnostic["reason"] == "insufficient_reclaim"


def test_repeated_checkpoint_keeps_model_visible_index_bounded_and_archives_full_index() -> None:
    previous_payload = {
        "kind": "context_checkpoint",
        "version": 1,
        "epoch": 1,
        "compacted_message_count": 80,
        "records": [{"type": "tool", "summary": f"record-{index}"} for index in range(40)],
        "artifact_refs": [
            {
                "tool_name": "shell",
                "path": f"artifacts/old-{index}.txt",
                "digest": f"sha256:old-{index}",
                "original_chars": 7_000,
            }
            for index in range(40)
        ],
        "archive": {
            "path": "artifacts/checkpoint-1.txt",
            "digest": "sha256:checkpoint-1",
            "record_count": 80,
            "artifact_ref_count": 80,
            "parent_archive_count": 0,
        },
    }
    messages = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "Task:\ngoal: inspect the project"},
        {"role": "user", "content": json.dumps(previous_payload)},
    ]
    for index in range(5, 11):
        messages.extend(_tool_pair(index))
    saved: list[tuple[str, str]] = []

    result = maybe_checkpoint_messages(
        messages=messages,
        budget=_budget(),
        epoch=1,
        artifact_writer=lambda tool_name, content: (
            saved.append((tool_name, content)) or f"artifacts/{tool_name}-{len(saved)}.json"
        ),
    )

    assert result.applied is True
    checkpoint = checkpoint_payload(result.messages[2])
    assert checkpoint is not None
    assert len(checkpoint["records"]) <= 24
    assert len(checkpoint["artifact_refs"]) <= 24
    assert checkpoint["archive"]["path"].startswith("artifacts/context-checkpoint-")
    archive = json.loads(next(content for tool_name, content in saved if tool_name == "context-checkpoint"))
    assert archive["kind"] == "context_checkpoint_archive"
    assert archive["parent_archives"][0]["path"] == "artifacts/checkpoint-1.txt"
    assert len(archive["records"]) > len(checkpoint["records"])
