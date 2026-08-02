"""
src/haagent/context/compression/tool_results.py - 工具结果模型可见视图

所有可能进入模型的长工具结果只在这里生成一次模型视图；完整结果仍保留在工具 trace 或 artifact 中。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from haagent.context.compression.budget import CompressionBudget, derive_compression_budget, estimate_text_tokens
from haagent.context.compression.diagnostics import CompressionDiagnostic

ArtifactWriter = Callable[[str, str], str]
_PROCESS_METADATA_KEYS = (
    "job_id",
    "job_status",
    "stream",
    "logs_truncated",
    "command",
    "cwd",
    "duration_seconds",
    "waited_seconds",
    "error_message",
    "exit_code",
    "timeout",
    "redacted",
)


@dataclass(frozen=True)
class ToolResultArtifact:
    path: str
    original_chars: int
    preview_chars: int
    original_bytes: int | None = None
    preview_bytes: int | None = None


@dataclass(frozen=True)
class ToolResultTruncation:
    occurred: bool
    reason: str | None
    original_chars: int
    visible_chars: int
    omitted_chars: int
    original_bytes: int
    visible_bytes: int
    omitted_bytes: int
    estimated_original_tokens: int | None
    estimated_visible_tokens: int | None
    estimated_omitted_tokens: int | None
    artifact_path: str | None
    recovery_hint: str | None


@dataclass(frozen=True)
class ToolResultView:
    kind: str
    tool_name: str
    status: str
    content: str
    content_format: str
    artifact: ToolResultArtifact | None
    representation_version: int
    content_digest: str
    truncation: ToolResultTruncation
    source_scope: dict[str, Any] | None = None
    continuation_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_tool_result_for_model(
    tool_name: str,
    result: dict[str, Any],
    budget: CompressionBudget | None,
    artifact_writer: ArtifactWriter,
) -> dict[str, Any]:
    """把工具结果投影成一次、可恢复且带精确统计的模型视图。"""

    active_budget = budget or derive_compression_budget(None)
    if process_view := _prepare_process_output_view(tool_name, result, active_budget, artifact_writer):
        return process_view

    candidate = _model_visible_candidate(tool_name, result)
    if candidate is None:
        candidate = _large_unprojected_candidate(result, _inline_char_budget(active_budget))
    if candidate is None:
        return result

    content, content_format, source_scope = candidate
    max_chars = _inline_char_budget(active_budget)
    preview, visible_source, omitted_chars = _head_tail_preview(content, max_chars)
    occurred = omitted_chars > 0
    artifact_path: str | None = None
    reason: str | None = "model_input_budget" if occurred else None
    recovery_hint: str | None = None
    artifact: ToolResultArtifact | None = None
    projected_result = result

    if occurred:
        try:
            written_path = artifact_writer(tool_name, content)
            if not isinstance(written_path, str) or not written_path.strip():
                raise OSError("artifact writer returned an empty path")
            artifact_path = written_path
        except Exception as error:
            # artifact 落盘失败时显式返回错误，不能伪装成成功并静默丢数据。
            reason = "artifact_write_failed"
            recovery_hint = "完整结果保存失败；请使用更窄的命令、offset/limit 或 grep 重新获取。"
            projected_result = {
                **result,
                "status": "error",
                "error": {
                    "type": "tool_output_artifact_write_failed",
                    "message": f"完整工具结果保存失败: {error}",
                },
            }
        else:
            recovery_hint = f"使用 file_read 读取完整结果: {artifact_path}"
            artifact = ToolResultArtifact(
                path=artifact_path,
                original_chars=len(content),
                preview_chars=len(visible_source),
                original_bytes=len(content.encode("utf-8")),
                preview_bytes=len(visible_source.encode("utf-8")),
            )

    truncation = _build_truncation(
        content,
        visible_source=visible_source,
        omitted_chars=omitted_chars,
        occurred=occurred,
        reason=reason,
        artifact_path=artifact_path,
        recovery_hint=recovery_hint,
    )
    view = _view_dict(
        tool_name=tool_name,
        status=str(projected_result.get("status", "success")),
        content=preview,
        content_format=content_format,
        artifact=artifact,
        content_digest=_content_digest(content),
        truncation=truncation,
        source_scope=source_scope,
        continuation_hint=recovery_hint,
    )
    diagnostic = CompressionDiagnostic(
        stage="tool_output_artifact" if occurred else "tool_output_view",
        subject=tool_name,
        decision="offloaded" if occurred else "preserved",
        reason=reason or "within_model_input_budget",
        original_chars=len(content),
        final_chars=len(visible_source),
        artifact_path=artifact_path,
    ).to_dict()
    return {**projected_result, "model_visible": view, "compression_diagnostics": [diagnostic]}


def _prepare_process_output_view(
    tool_name: str,
    result: dict[str, Any],
    budget: CompressionBudget,
    artifact_writer: ArtifactWriter,
) -> dict[str, Any] | None:
    """把 shell/code_run/job 的 stdout/stderr 一次性投影，并保留精确流统计。"""

    if not _is_process_output(result):
        return None

    stream_values = {
        "stdout": str(result.get("stdout") or result.get("stdout_excerpt") or ""),
        "stderr": str(result.get("stderr") or result.get("stderr_excerpt") or ""),
    }
    original_chars = {
        name: _nonnegative_int(result.get(f"{name}_original_chars"), len(value))
        for name, value in stream_values.items()
    }
    original_bytes = {
        name: _nonnegative_int(
            result.get(f"{name}_original_bytes"),
            len(value.encode("utf-8")),
        )
        for name, value in stream_values.items()
    }
    max_chars = _inline_char_budget(budget)
    visible_streams = _bounded_stream_values(stream_values, max_chars)
    payload: dict[str, Any] = {
        "stdout": visible_streams["stdout"][0],
        "stderr": visible_streams["stderr"][0],
    }
    for key in _PROCESS_METADATA_KEYS:
        if key in result:
            payload[key] = result[key]
    preview = json.dumps(payload, ensure_ascii=False)
    visible_sources = {name: pair[1] for name, pair in visible_streams.items()}
    visible_chars = sum(len(value) for value in visible_sources.values())
    visible_bytes = sum(len(value.encode("utf-8")) for value in visible_sources.values())
    total_chars = sum(original_chars.values())
    total_bytes = sum(original_bytes.values())
    omitted_chars = max(0, total_chars - visible_chars)
    omitted_bytes = max(0, total_bytes - visible_bytes)
    capture_paths = [
        str(result[key])
        for key in ("stdout_artifact_path", "stderr_artifact_path")
        if result.get(key)
    ]
    artifact_path: str | None = capture_paths[0] if capture_paths else None
    reason: str | None = None
    recovery_hint: str | None = None
    artifact: ToolResultArtifact | None = None
    projected_result = result

    if omitted_chars > 0 or omitted_bytes > 0 or capture_paths:
        reason = "process_capture_limit" if capture_paths or result.get("truncated") else "model_input_budget"
        if artifact_path is None:
            full_payload = {
                "stdout": stream_values["stdout"],
                "stderr": stream_values["stderr"],
                **{
                    key: result[key]
                    for key in _PROCESS_METADATA_KEYS
                    if key in result
                },
            }
            try:
                written_path = artifact_writer(
                    tool_name,
                    json.dumps(full_payload, ensure_ascii=False),
                )
                if not isinstance(written_path, str) or not written_path.strip():
                    raise OSError("artifact writer returned an empty path")
                artifact_path = written_path
            except Exception as error:
                reason = "artifact_write_failed"
                recovery_hint = "完整进程输出保存失败；请重新运行更窄的命令并使用 head/tail/grep。"
                projected_result = {
                    **result,
                    "status": "error",
                    "error": {
                        "type": "tool_output_artifact_write_failed",
                        "message": f"完整进程输出保存失败: {error}",
                    },
                }
        if artifact_path is not None:
            recovery_hint = "使用 file_read 读取完整进程输出: " + ", ".join(
                [*capture_paths, artifact_path] if artifact_path not in capture_paths else capture_paths,
            )
            artifact = ToolResultArtifact(
                path=artifact_path,
                original_chars=total_chars,
                preview_chars=visible_chars,
                original_bytes=total_bytes,
                preview_bytes=visible_bytes,
            )
    truncation = ToolResultTruncation(
        occurred=omitted_chars > 0 or omitted_bytes > 0,
        reason=reason,
        original_chars=total_chars,
        visible_chars=visible_chars,
        omitted_chars=omitted_chars,
        original_bytes=total_bytes,
        visible_bytes=visible_bytes,
        omitted_bytes=omitted_bytes,
        estimated_original_tokens=(total_chars + 3) // 4 if total_chars else 0,
        estimated_visible_tokens=(visible_chars + 3) // 4 if visible_chars else 0,
        estimated_omitted_tokens=max(
            0,
            ((total_chars + 3) // 4 if total_chars else 0)
            - ((visible_chars + 3) // 4 if visible_chars else 0),
        ),
        artifact_path=artifact_path,
        recovery_hint=recovery_hint,
    )
    view = _view_dict(
        tool_name=tool_name,
        status=str(projected_result.get("status", "success")),
        content=preview,
        content_format="json",
        artifact=artifact,
        content_digest=_content_digest(json.dumps(payload, ensure_ascii=False)),
        truncation=truncation,
        source_scope=None,
        continuation_hint=recovery_hint,
    )
    return {
        **projected_result,
        "model_visible": view,
        "compression_diagnostics": [
            CompressionDiagnostic(
                stage="process_output_artifact" if artifact_path else "tool_output_view",
                subject=tool_name,
                decision="offloaded" if artifact_path else "preserved",
                reason=reason or "within_model_input_budget",
                original_chars=total_chars,
                final_chars=visible_chars,
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


def _model_visible_candidate(
    tool_name: str,
    result: dict[str, Any],
) -> tuple[str, str, dict[str, Any] | None] | None:
    existing = result.get("model_visible")
    if isinstance(existing, dict):
        if existing.get("kind") == "tool_result_view":
            return None
        content = existing.get("content")
        if isinstance(content, str):
            return content, str(existing.get("content_format") or "text"), existing.get("source_scope")

    if tool_name.startswith("mcp__") and isinstance(result.get("output"), str):
        return str(result["output"]), "text", None

    # task_output 的正文不使用 stdout 字段，仍必须进入同一预算入口。
    if tool_name == "task_output" and isinstance(result.get("output"), str):
        return _json_payload(result), "json", None
    if isinstance(result.get("tree"), str):
        return _json_payload(result), "json", None
    if isinstance(result.get("matches"), list):
        return _json_payload(result), "json", None
    if any(isinstance(result.get(key), str) for key in ("stdout_excerpt", "stderr_excerpt")):
        return _json_payload(result), "json", None
    return None


def _large_unprojected_candidate(
    result: dict[str, Any],
    max_chars: int,
) -> tuple[str, str, dict[str, Any] | None] | None:
    """兜底覆盖没有专用 model_visible 的长 output/content，短结果保留原工具契约。"""

    for key in ("output", "content"):
        value = result.get(key)
        if isinstance(value, str) and len(value) > max_chars:
            return value, "text", None
    for key in ("tree", "matches"):
        value = result.get(key)
        if value is None:
            continue
        candidate = _json_payload(result)
        if len(candidate) > max_chars:
            return candidate, "json", None
    return None


def _view_dict(
    *,
    tool_name: str,
    status: str,
    content: str,
    content_format: str,
    artifact: ToolResultArtifact | None,
    content_digest: str,
    truncation: ToolResultTruncation,
    source_scope: dict[str, Any] | None,
    continuation_hint: str | None,
) -> dict[str, Any]:
    return ToolResultView(
        kind="tool_result_view",
        tool_name=tool_name,
        status=status,
        content=content,
        content_format=content_format,
        artifact=artifact,
        representation_version=2,
        content_digest=content_digest,
        truncation=truncation,
        source_scope=source_scope,
        continuation_hint=continuation_hint,
    ).to_dict()


def _json_payload(result: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"model_visible", "compression_diagnostics"}
    }
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


def _content_digest(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _inline_char_budget(budget: CompressionBudget) -> int:
    """以稳定四字符近似换算 token；真正 token 数保留为 estimated 字段。"""
    inline_tokens = min(
        max(0, int(budget.tool_output_inline_tokens)),
        max(0, int(budget.available_input_tokens)),
    )
    return max(1, inline_tokens * 4)


def _head_tail_preview(value: str, max_chars: int) -> tuple[str, str, int]:
    """返回 (含省略提示的预览、实际保留源文本、省略字符数)。"""
    if len(value) <= max_chars:
        return value, value, 0

    head = max_chars // 2
    tail = max_chars - head
    for _ in range(100):
        omitted = len(value) - head - tail
        marker = f"\n...[{omitted} characters omitted]...\n"
        keep = max_chars - len(marker)
        if keep <= 0:
            visible_source = value[:max_chars]
            return visible_source, visible_source, len(value) - len(visible_source)
        next_head = keep // 2
        next_tail = keep - next_head
        if (next_head, next_tail) == (head, tail):
            break
        head, tail = next_head, next_tail
    omitted = len(value) - head - tail
    marker = f"\n...[{omitted} characters omitted]...\n"
    visible_source = value[:head] + value[-tail:]
    return f"{value[:head]}{marker}{value[-tail:]}", visible_source, omitted


def _bounded_stream_values(streams: dict[str, str], max_chars: int) -> dict[str, tuple[str, str]]:
    total = sum(len(value) for value in streams.values())
    if total <= max_chars:
        return {name: (value, value) for name, value in streams.items()}
    nonempty = [name for name, value in streams.items() if value]
    if not nonempty:
        return {name: ("", "") for name in streams}
    if len(nonempty) == 1:
        capacities = {nonempty[0]: max_chars}
    else:
        first, second = nonempty[:2]
        first_capacity = max(1, min(len(streams[first]) - 1, max_chars * len(streams[first]) // total))
        capacities = {first: first_capacity, second: max(1, max_chars - first_capacity)}
    result: dict[str, tuple[str, str]] = {}
    for name, value in streams.items():
        if not value:
            result[name] = ("", "")
            continue
        preview, visible_source, _omitted = _head_tail_preview(value, capacities.get(name, max_chars))
        result[name] = (preview, visible_source)
    return result


def _build_truncation(
    content: str,
    *,
    visible_source: str,
    omitted_chars: int,
    occurred: bool,
    reason: str | None,
    artifact_path: str | None,
    recovery_hint: str | None,
) -> ToolResultTruncation:
    original_chars = len(content)
    visible_chars = len(visible_source)
    original_bytes = len(content.encode("utf-8"))
    visible_bytes = len(visible_source.encode("utf-8"))
    original_tokens = estimate_text_tokens(content)
    visible_tokens = estimate_text_tokens(visible_source)
    return ToolResultTruncation(
        occurred=occurred,
        reason=reason,
        original_chars=original_chars,
        visible_chars=visible_chars,
        omitted_chars=omitted_chars,
        original_bytes=original_bytes,
        visible_bytes=visible_bytes,
        omitted_bytes=max(0, original_bytes - visible_bytes),
        estimated_original_tokens=original_tokens,
        estimated_visible_tokens=visible_tokens,
        estimated_omitted_tokens=max(0, original_tokens - visible_tokens),
        artifact_path=artifact_path,
        recovery_hint=recovery_hint,
    )


def _is_process_output(result: dict[str, Any]) -> bool:
    return any(
        key in result
        for key in (
            "stdout",
            "stderr",
            "stdout_excerpt",
            "stderr_excerpt",
            "stdout_artifact_path",
            "stderr_artifact_path",
        )
    )


def _nonnegative_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return max(0, default)
