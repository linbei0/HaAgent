"""
src/haagent/context/compression/tool_results.py - 工具结果模型可见压缩

统一生成 ToolResultView，负责长工具结果 artifact 落盘、预览和 provider 渲染。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from haagent.context.compression.budget import CompressionBudget, derive_compression_budget
from haagent.context.compression.diagnostics import CompressionDiagnostic

ArtifactWriter = Callable[[str, str], str]


@dataclass(frozen=True)
class ToolResultArtifact:
    path: str
    original_chars: int
    preview_chars: int


@dataclass(frozen=True)
class ToolResultView:
    kind: str
    tool_name: str
    status: str
    content: str
    content_format: str
    artifact: ToolResultArtifact | None
    truncated: bool
    continuation_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def prepare_tool_result_for_model(
    tool_name: str,
    result: dict[str, Any],
    budget: CompressionBudget | None,
    artifact_writer: ArtifactWriter,
) -> dict[str, Any]:
    if result.get("status") != "success":
        return result
    active_budget = budget or derive_compression_budget(None)
    candidate = _model_visible_candidate(tool_name, result)
    if candidate is None:
        return result
    content, content_format = candidate
    if len(content) <= active_budget.tool_output_inline_chars:
        return {
            **result,
            "model_visible": _view_dict(
                tool_name=tool_name,
                status=str(result.get("status", "success")),
                content=content,
                content_format=content_format,
                artifact=None,
                truncated=bool(result.get("truncated", False)),
                continuation_hint=None,
            ),
        }
    artifact_path = artifact_writer(tool_name, content)
    preview = _head_tail_preview(content, active_budget.tool_output_preview_chars)
    return {
        **result,
        "model_visible": _view_dict(
            tool_name=tool_name,
            status=str(result.get("status", "success")),
            content=preview,
            content_format=content_format,
            artifact=ToolResultArtifact(
                path=artifact_path,
                original_chars=len(content),
                preview_chars=len(preview),
            ),
            truncated=True,
            continuation_hint=f"Use file_read with path={artifact_path} to inspect the full tool output.",
        ),
        "compression_diagnostics": [
            CompressionDiagnostic(
                stage="tool_output_artifact",
                subject=tool_name,
                decision="offloaded",
                reason="tool_result_over_inline_budget",
                original_chars=len(content),
                final_chars=len(preview),
                artifact_path=artifact_path,
            ).to_dict(),
        ],
    }


def render_tool_result_view(view: ToolResultView | dict[str, Any]) -> str:
    payload = view.to_dict() if isinstance(view, ToolResultView) else dict(view)
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def _model_visible_candidate(tool_name: str, result: dict[str, Any]) -> tuple[str, str] | None:
    existing = result.get("model_visible")
    if isinstance(existing, dict):
        if existing.get("kind") == "tool_result_view":
            return None
        content = existing.get("content")
        if tool_name == "web_fetch" and isinstance(content, str):
            return content, str(existing.get("content_format") or "text")
    if tool_name.startswith("mcp__") and isinstance(result.get("output"), str):
        return str(result["output"]), "text"
    return None


def _view_dict(
    *,
    tool_name: str,
    status: str,
    content: str,
    content_format: str,
    artifact: ToolResultArtifact | None,
    truncated: bool,
    continuation_hint: str | None,
) -> dict[str, Any]:
    return ToolResultView(
        kind="tool_result_view",
        tool_name=tool_name,
        status=status,
        content=content,
        content_format=content_format,
        artifact=artifact,
        truncated=truncated,
        continuation_hint=continuation_hint,
    ).to_dict()


def _head_tail_preview(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n...[model-visible content truncated]...\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return value[:max_chars]
    head = keep // 2
    tail = keep - head
    return f"{value[:head].rstrip()}{marker}{value[-tail:].lstrip()}"
