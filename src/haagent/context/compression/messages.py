"""
src/haagent/context/compression/messages.py - 历史工具消息压缩

按类型压缩历史 tool message，保留最近 artifact 预览，降级更早 artifact 结果。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from haagent.context.compression.budget import CompressionBudget, estimate_text_tokens
from haagent.context.compression.diagnostics import CompressionDiagnostic


class HistoricalToolCompressionPolicy:
    def __init__(self, budget: CompressionBudget) -> None:
        self.budget = budget


def compress_historical_tool_messages(
    messages: list[dict[str, Any]],
    budget: CompressionBudget,
    writer: object | None = None,
    turn: int | None = None,
    emit_event: Callable[[dict[str, object]], None] | None = None,
) -> list[CompressionDiagnostic]:
    artifact_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool" and _tool_result_view_payload(message) is not None
    ]
    recent_artifact_indices = set(artifact_indices[-budget.artifact_recent_preview_count :])
    diagnostics: list[CompressionDiagnostic] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        payload = _tool_result_view_payload(message)
        diagnostic: CompressionDiagnostic | None = None
        if payload is not None:
            if index in recent_artifact_indices:
                continue
            diagnostic = _summarize_artifact_payload(message, payload)
        elif field_diagnostic := _collapse_json_tool_result_fields(message, content, budget):
            diagnostic = field_diagnostic
        elif len(content) > _historical_text_limit(budget):
            collapsed = _collapse_text_head_tail(
                content,
                head_chars=budget.historical_collapse_head_chars,
                tail_chars=budget.historical_collapse_tail_chars,
            )
            message["content"] = collapsed
            diagnostic = CompressionDiagnostic(
                stage="historical_tool_message",
                subject=str(message.get("name", "unknown_tool")),
                decision="collapsed",
                reason="long_text_result",
                original_chars=len(content),
                final_chars=len(collapsed),
                original_tokens=estimate_text_tokens(content),
                final_tokens=estimate_text_tokens(collapsed),
            )
        if diagnostic is None:
            continue
        diagnostics.append(diagnostic)
        event = _diagnostic_event(diagnostic, turn=turn, message_index=index)
        if writer is not None and hasattr(writer, "append_transcript"):
            writer.append_transcript({"event": "compression_diagnostic", **event})
        if emit_event is not None:
            emit_event({"event_type": "compression_diagnostic", **event})
    return diagnostics


def _collapse_json_tool_result_fields(
    message: dict[str, Any],
    content: str,
    budget: CompressionBudget,
) -> CompressionDiagnostic | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    changed = False
    field_limit = budget.historical_collapse_head_chars + budget.historical_collapse_tail_chars + 200
    for key in ("content", "output", "stdout", "stderr"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) <= field_limit:
            continue
        collapsed = _collapse_text_head_tail(
            value,
            head_chars=budget.historical_collapse_head_chars,
            tail_chars=budget.historical_collapse_tail_chars,
        )
        payload[key] = collapsed
        payload["truncated"] = True
        changed = True
    if not changed:
        return None
    message["content"] = json.dumps(payload, ensure_ascii=False)
    return CompressionDiagnostic(
        stage="historical_tool_message",
        subject=str(message.get("name", "unknown_tool")),
        decision="collapsed",
        reason="long_text_result",
        original_chars=len(content),
        final_chars=len(message["content"]),
        original_tokens=estimate_text_tokens(content),
        final_tokens=estimate_text_tokens(message["content"]),
    )


def _summarize_artifact_payload(message: dict[str, Any], payload: dict[str, Any]) -> CompressionDiagnostic | None:
    original_content = str(message.get("content", ""))
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        return None
    path = str(artifact.get("path", ""))
    if not path:
        return None
    original_chars = _int_value(artifact.get("original_chars"))
    tool_name = str(payload.get("tool_name") or message.get("name") or "unknown_tool")
    hint = str(payload.get("continuation_hint") or f"Use file_read with path={path}")
    summary = f"{tool_name} result saved at {path} ({original_chars} chars). {hint}"
    payload["content"] = summary
    payload["content_format"] = "summary"
    payload["truncated"] = True
    message["content"] = json.dumps(payload, ensure_ascii=False)
    return CompressionDiagnostic(
        stage="historical_tool_message",
        subject=tool_name,
        decision="artifact_summary",
        reason="older_artifact_result",
        original_chars=len(original_content),
        final_chars=len(message["content"]),
        original_tokens=estimate_text_tokens(original_content),
        final_tokens=estimate_text_tokens(message["content"]),
        artifact_path=path,
    )


def _tool_result_view_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") == "tool_result_view" and isinstance(payload.get("artifact"), dict):
        return payload
    if "artifact_path" in payload:
        path = str(payload.get("artifact_path") or "")
        if not path:
            return None
        return {
            "kind": "tool_result_view",
            "tool_name": str(message.get("name", "unknown_tool")),
            "status": "success",
            "content": str(payload.get("output") or payload.get("content") or ""),
            "content_format": "text",
            "artifact": {
                "path": path,
                "original_chars": _int_value(payload.get("original_chars")),
                "preview_chars": _int_value(payload.get("preview_chars")),
            },
            "truncated": bool(payload.get("truncated", True)),
            "continuation_hint": payload.get("continuation_hint"),
        }
    return None


def _historical_text_limit(budget: CompressionBudget) -> int:
    return budget.tool_output_inline_chars


def _collapse_text_head_tail(text: str, *, head_chars: int, tail_chars: int) -> str:
    if len(text) <= head_chars + tail_chars:
        return text
    omitted = len(text) - head_chars - tail_chars
    return f"{text[:head_chars].rstrip()}\n...[collapsed {omitted} chars]...\n{text[-tail_chars:].lstrip()}"


def _diagnostic_event(
    diagnostic: CompressionDiagnostic,
    *,
    turn: int | None,
    message_index: int,
) -> dict[str, object]:
    event = diagnostic.to_dict()
    event["message_index"] = message_index
    if turn is not None:
        event["turn"] = turn
    return event


def _int_value(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0
