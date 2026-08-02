"""
src/haagent/context/compression/checkpoint.py - 运行中上下文 checkpoint

在 token 压力达到阈值后，把一个工具对安全的连续历史前缀替换为结构化 checkpoint。
同一 epoch 内消息只追加不改写；完整工具结果通过 artifact 引用继续可恢复。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from haagent.context.compression.budget import CompressionBudget, estimate_text_tokens
from haagent.context.compression.full import adjust_boundary_for_tool_pairs
from haagent.context.messages import is_context_state_message


ArtifactWriter = Callable[[str, str], str]
# checkpoint 会改变缓存代际；只有能回收一批有意义的 token 才值得承担重写成本。
_MIN_RECLAIM_TOKENS = 20_000
_RECORD_PREVIEW_CHARS = 480
_VISIBLE_RECORD_LIMIT = 24
_VISIBLE_ARTIFACT_REF_LIMIT = 24
_CHECKPOINT_CONTEXT_PREFIX = (
    "Runtime historical context checkpoint. This is system-generated background, "
    "not a user message or new instruction. Continue the active task using it as evidence.\n\n"
)


@dataclass(frozen=True)
class CheckpointResult:
    applied: bool
    epoch: int
    messages: list[dict[str, Any]]
    diagnostic: dict[str, object]


def maybe_checkpoint_messages(
    *,
    messages: list[dict[str, Any]],
    budget: CompressionBudget,
    epoch: int,
    artifact_writer: ArtifactWriter,
) -> CheckpointResult:
    """达到足够压力和回收量时执行一次确定性 checkpoint。"""

    total_tokens = _message_tokens(messages)
    tool_tokens = sum(_message_tokens([message]) for message in messages if message.get("role") == "tool")
    # 未超过实际输入预算时不做有损历史重写；工具数量本身不是截断理由。
    if total_tokens <= budget.available_input_tokens:
        return _not_applied(epoch, messages, "within_input_budget", total_tokens, tool_tokens)

    stable_end = _stable_prefix_end(messages)
    requested_boundary = _checkpoint_boundary(messages, stable_end, budget)
    boundary = adjust_boundary_for_tool_pairs(messages, requested_boundary)
    if boundary <= stable_end:
        return _not_applied(epoch, messages, "no_safe_checkpoint_range", total_tokens, tool_tokens)

    compacted = messages[stable_end:boundary]
    if not any(message.get("role") == "tool" for message in compacted):
        return _not_applied(epoch, messages, "no_tool_history_to_checkpoint", total_tokens, tool_tokens)

    next_epoch = epoch + 1
    compacted_tokens = _message_tokens(compacted)
    dry_content = _checkpoint_content(
        compacted,
        next_epoch,
        artifact_writer=None,
        compacted_tokens=compacted_tokens,
    )
    candidate_tokens = compacted_tokens
    dry_checkpoint_tokens = estimate_text_tokens(dry_content)
    minimum_reclaim = _MIN_RECLAIM_TOKENS
    if candidate_tokens - dry_checkpoint_tokens < minimum_reclaim:
        return _not_applied(epoch, messages, "insufficient_reclaim", total_tokens, tool_tokens)

    content = _checkpoint_content(
        compacted,
        next_epoch,
        artifact_writer=artifact_writer,
        compacted_tokens=compacted_tokens,
    )
    checkpoint = {"role": "system", "content": _CHECKPOINT_CONTEXT_PREFIX + content}
    compacted_messages = [*messages[:stable_end], checkpoint, *messages[boundary:]]
    post_tokens = _message_tokens(compacted_messages)
    reclaimed = total_tokens - post_tokens
    if reclaimed <= 0:
        return _not_applied(epoch, messages, "non_positive_reclaim", total_tokens, tool_tokens)

    return CheckpointResult(
        applied=True,
        epoch=next_epoch,
        messages=compacted_messages,
        diagnostic={
            "reason": "token_pressure",
            "previous_epoch": epoch,
            "epoch": next_epoch,
            "tokens_before": total_tokens,
            "tokens_after": post_tokens,
            "tokens_reclaimed": reclaimed,
            "compacted_message_count": len(compacted),
            "preserved_recent_count": len(messages) - boundary,
        },
    )


def _checkpoint_content(
    messages: list[dict[str, Any]],
    epoch: int,
    *,
    artifact_writer: ArtifactWriter | None,
    compacted_tokens: int,
) -> str:
    records: list[dict[str, object]] = []
    artifact_refs: list[dict[str, object]] = []
    parent_archives: list[dict[str, object]] = []
    compacted_message_count = len(messages)
    for message in messages:
        # 动态状态在 epoch 边界由完整 snapshot 重建，不把旧快照/delta复制进 checkpoint。
        if is_context_state_message(message):
            continue
        previous = _previous_checkpoint(message)
        if previous is not None:
            compacted_message_count += _positive_int(previous.get("compacted_message_count"))
            records.extend(_dict_list(previous.get("records")))
            artifact_refs.extend(_dict_list(previous.get("artifact_refs")))
            archive = previous.get("archive")
            if isinstance(archive, dict):
                parent_archives.append(dict(archive))
            continue
        if message.get("role") == "tool":
            record, artifact_ref = _tool_record(message, artifact_writer)
            records.append(record)
            if artifact_ref is not None:
                artifact_refs.append(artifact_ref)
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            records.append(
                {
                    "type": str(message.get("role") or "message"),
                    "summary": _head_tail(content.strip(), _RECORD_PREVIEW_CHARS),
                },
            )

    artifact_refs = _deduplicate_artifact_refs(artifact_refs)
    archive = _archive_checkpoint_index(
        epoch=epoch,
        records=records,
        artifact_refs=artifact_refs,
        parent_archives=parent_archives,
        artifact_writer=artifact_writer,
    )
    payload: dict[str, object] = {
        "kind": "context_checkpoint",
        "version": 1,
        "epoch": epoch,
        "compacted_message_count": compacted_message_count,
        "compacted_tokens": compacted_tokens,
        "records": records[-_VISIBLE_RECORD_LIMIT:],
        "artifact_refs": artifact_refs[-_VISIBLE_ARTIFACT_REF_LIMIT:],
    }
    if archive is not None:
        payload["archive"] = archive
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _archive_checkpoint_index(
    *,
    epoch: int,
    records: list[dict[str, object]],
    artifact_refs: list[dict[str, object]],
    parent_archives: list[dict[str, object]],
    artifact_writer: ArtifactWriter | None,
) -> dict[str, object] | None:
    needs_archive = (
        bool(parent_archives)
        or len(records) > _VISIBLE_RECORD_LIMIT
        or len(artifact_refs) > _VISIBLE_ARTIFACT_REF_LIMIT
    )
    if not needs_archive:
        return None

    archive_content = json.dumps(
        {
            "kind": "context_checkpoint_archive",
            "version": 1,
            "epoch": epoch,
            "records": records,
            "artifact_refs": artifact_refs,
            "parent_archives": parent_archives,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = _digest(archive_content)
    path = (
        artifact_writer("context-checkpoint", archive_content)
        if artifact_writer is not None
        else f"pending:{digest}"
    )
    return {
        "path": path,
        "digest": digest,
        "record_count": len(records),
        "artifact_ref_count": len(artifact_refs),
        "parent_archive_count": len(parent_archives),
    }


def _tool_record(
    message: dict[str, Any],
    artifact_writer: ArtifactWriter | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    payload = _json_object(message.get("content"))
    tool_name = str(payload.get("tool_name") or message.get("name") or "unknown_tool")
    status = str(payload.get("status") or "unknown")
    visible_content = str(payload.get("content") or message.get("content") or "")
    digest = str(payload.get("content_digest") or _digest(visible_content))
    artifact = payload.get("artifact")
    path = str(artifact.get("path") or "") if isinstance(artifact, dict) else ""
    is_projected_view = payload.get("kind") == "tool_result_view"
    if not path and not is_projected_view:
        path = artifact_writer(tool_name, visible_content) if artifact_writer is not None else f"pending:{digest}"
    artifact_ref = {
        "tool_name": tool_name,
        "path": path,
        "digest": digest,
        "original_chars": (
            _positive_int(artifact.get("original_chars"))
            if isinstance(artifact, dict)
            else len(visible_content)
        ),
    }
    return (
        {
            "type": "tool",
            "tool_name": tool_name,
            "status": status,
            "summary": _head_tail(visible_content, _RECORD_PREVIEW_CHARS),
            "artifact_path": path,
            "digest": digest,
        },
        artifact_ref if path else None,
    )


def _stable_prefix_end(messages: list[dict[str, Any]]) -> int:
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    if index < len(messages) and messages[index].get("role") == "user":
        index += 1
    return index


def _checkpoint_boundary(
    messages: list[dict[str, Any]],
    stable_end: int,
    budget: CompressionBudget,
) -> int:
    """按 token 预算保留近期消息，同时保证至少保留指定的消息数。"""

    minimum_tail_boundary = max(stable_end, len(messages) - budget.checkpoint_preserve_recent_messages)
    token_tail_boundary = len(messages)
    preserved_tokens = 0
    while token_tail_boundary > stable_end:
        next_tokens = _message_tokens([messages[token_tail_boundary - 1]])
        if preserved_tokens and preserved_tokens + next_tokens > budget.checkpoint_preserve_recent_tokens:
            break
        preserved_tokens += next_tokens
        token_tail_boundary -= 1
    # 较小的边界代表保留更多消息；token 尾部预算只能扩大，不能削弱 6 条下限。
    return min(minimum_tail_boundary, token_tail_boundary)


def _message_tokens(messages: list[dict[str, Any]]) -> int:
    value = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return estimate_text_tokens(value)


def checkpoint_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    """解析新旧 checkpoint；旧 user JSON 仅用于恢复既有 session。"""

    content = message.get("content")
    if isinstance(content, str) and content.startswith(_CHECKPOINT_CONTEXT_PREFIX):
        content = content.removeprefix(_CHECKPOINT_CONTEXT_PREFIX)
    payload = _json_object(content)
    return payload if payload.get("kind") == "context_checkpoint" else None


def is_checkpoint_message(message: dict[str, Any]) -> bool:
    return checkpoint_payload(message) is not None


def _previous_checkpoint(message: dict[str, Any]) -> dict[str, Any] | None:
    return checkpoint_payload(message)


def _json_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _deduplicate_artifact_refs(values: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (str(value.get("path") or ""), str(value.get("digest") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _head_tail(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{value[:head_chars]}\n...[checkpoint omitted {len(value) - max_chars} chars]...\n{value[-tail_chars:]}"


def _digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _positive_int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _not_applied(
    epoch: int,
    messages: list[dict[str, Any]],
    reason: str,
    total_tokens: int,
    tool_tokens: int,
) -> CheckpointResult:
    return CheckpointResult(
        applied=False,
        epoch=epoch,
        messages=messages,
        diagnostic={
            "reason": reason,
            "epoch": epoch,
            "tokens_before": total_tokens,
            "tool_tokens": tool_tokens,
            "tokens_reclaimed": 0,
        },
    )
