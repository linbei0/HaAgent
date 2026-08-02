"""
tests/unit/context/test_tool_result_visibility.py - 工具结果模型可见合同测试

验证统一 ToolResultView 负责工具长结果落盘、预览和渲染。
"""

import json
import hashlib

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.budget import CompressionBudget
from haagent.context.compression.tool_results import (
    prepare_tool_result_for_model,
    render_tool_result_view,
)
from haagent.context.messages import build_tool_result_message


def test_long_mcp_output_becomes_tool_result_view_with_artifact() -> None:
    saved: dict[str, str] = {}
    output = "start " + ("middle " * 7000) + "important tail"

    def artifact_writer(tool_name: str, content: str) -> str:
        saved["tool_name"] = tool_name
        saved["content"] = content
        return ".runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt"

    result = prepare_tool_result_for_model(
        "mcp__fixture__fetch",
        {"status": "success", "output": output},
        derive_compression_budget(None),
        artifact_writer,
    )

    view = result["model_visible"]
    assert view["kind"] == "tool_result_view"
    assert view["tool_name"] == "mcp__fixture__fetch"
    assert view["content_format"] == "text"
    assert view["artifact"]["path"] == ".runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt"
    assert view["artifact"]["original_chars"] == len(output)
    assert view["artifact"]["preview_chars"] == view["truncation"]["visible_chars"]
    assert view["truncation"]["occurred"] is True
    assert view["representation_version"] == 2
    assert view["truncation"]["original_chars"] == len(output)
    assert view["truncation"]["omitted_chars"] == len(output) - view["truncation"]["visible_chars"]
    assert view["truncation"]["artifact_path"] == view["artifact"]["path"]
    assert view["content_digest"] == f"sha256:{hashlib.sha256(output.encode('utf-8')).hexdigest()}"
    assert "start" in view["content"]
    assert "important tail" in view["content"]
    assert len(view["content"]) <= 40_000
    assert view["truncation"]["estimated_visible_tokens"] <= 10_000
    assert saved == {"tool_name": "mcp__fixture__fetch", "content": output}


def test_13522_chars_are_preserved_when_model_budget_is_sufficient() -> None:
    content = "x" * 13_522
    result = prepare_tool_result_for_model(
        "mcp__fixture__fetch",
        {"status": "success", "output": content},
        derive_compression_budget(None),
        lambda _tool_name, _content: "unused.txt",
    )

    view = result["model_visible"]
    assert view["content"] == content
    assert view["truncation"]["occurred"] is False
    assert view["artifact"] is None
    assert view["truncation"]["occurred"] is False
    assert view["truncation"]["omitted_chars"] == 0


def test_truncation_reports_exact_counts_under_small_budget() -> None:
    content = "x" * 13_522
    budget = CompressionBudget(
        context_window_tokens=4_000,
        reserved_output_tokens=500,
        safety_buffer_tokens=500,
        available_input_tokens=1_500,
        context_builder_max_tokens=500,
        tool_output_inline_tokens=1_000,
    )
    result = prepare_tool_result_for_model(
        "mcp__fixture__fetch",
        {"status": "success", "output": content},
        budget,
        lambda _tool_name, _content: "artifact.txt",
    )

    truncation = result["model_visible"]["truncation"]
    assert truncation["occurred"] is True
    assert truncation["original_chars"] == 13_522
    assert truncation["visible_chars"] + truncation["omitted_chars"] == 13_522
    assert truncation["original_bytes"] == 13_522
    assert truncation["visible_bytes"] + truncation["omitted_bytes"] == 13_522


def test_artifact_write_failure_is_returned_as_explicit_tool_error() -> None:
    content = "x" * 50_000

    def failing_writer(_tool_name: str, _content: str) -> str:
        raise OSError("disk full")

    result = prepare_tool_result_for_model(
        "mcp__fixture__fetch",
        {"status": "success", "output": content},
        derive_compression_budget(None),
        failing_writer,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_output_artifact_write_failed"
    assert result["model_visible"]["artifact"] is None
    assert result["model_visible"]["truncation"]["reason"] == "artifact_write_failed"


def test_process_output_view_keeps_exact_stream_counts_and_valid_json() -> None:
    stdout = "o" * 13_000
    stderr = "中" * 5_000
    budget = CompressionBudget(
        context_window_tokens=4_000,
        reserved_output_tokens=500,
        safety_buffer_tokens=500,
        available_input_tokens=1_500,
        context_builder_max_tokens=500,
        tool_output_inline_tokens=1_000,
    )
    result = prepare_tool_result_for_model(
        "shell",
        {
            "status": "success",
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": 0,
        },
        budget,
        lambda _tool_name, _content: "process-output.json",
    )

    view = result["model_visible"]
    truncation = view["truncation"]
    assert truncation["original_chars"] == len(stdout) + len(stderr)
    assert truncation["visible_chars"] + truncation["omitted_chars"] == truncation["original_chars"]
    assert truncation["original_bytes"] == len(stdout.encode()) + len(stderr.encode())
    assert truncation["visible_bytes"] + truncation["omitted_bytes"] == truncation["original_bytes"]
    payload = json.loads(view["content"])
    assert set(payload) >= {"stdout", "stderr", "exit_code"}


def test_build_tool_result_message_renders_view_without_raw_output() -> None:
    raw_output = "x" * 13_000
    result = {
        "status": "success",
        "output": raw_output,
        "model_visible": {
            "kind": "tool_result_view",
            "tool_name": "mcp__fixture__fetch",
            "status": "success",
            "content": "small visible preview",
            "content_format": "text",
            "artifact": {
                "path": ".runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt",
                "original_chars": len(raw_output),
                "preview_chars": len("small visible preview"),
            },
            "truncation": {
                "occurred": True,
                "reason": "model_input_budget",
                "original_chars": len(raw_output),
                "visible_chars": len("small visible preview"),
                "omitted_chars": len(raw_output) - len("small visible preview"),
                "original_bytes": len(raw_output.encode("utf-8")),
                "visible_bytes": len("small visible preview".encode("utf-8")),
                "omitted_bytes": len(raw_output.encode("utf-8")) - len("small visible preview".encode("utf-8")),
                "estimated_original_tokens": None,
                "estimated_visible_tokens": None,
                "estimated_omitted_tokens": None,
                "artifact_path": ".runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt",
                "recovery_hint": "Use file_read with path=.runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt",
            },
            "continuation_hint": "Use file_read with path=.runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt",
        },
    }

    message = build_tool_result_message("call_1", "mcp__fixture__fetch", result)

    payload = json.loads(message["content"])
    assert payload["kind"] == "tool_result_view"
    assert payload["artifact"]["path"] == ".runs/episode/artifacts/tool-results/mcp_fixture_fetch.txt"
    assert raw_output not in message["content"]


def test_render_tool_result_view_accepts_dataclass_or_dict() -> None:
    rendered = render_tool_result_view(
        {
            "kind": "tool_result_view",
            "tool_name": "fake_tool",
            "status": "success",
            "content": "ok",
            "content_format": "text",
            "artifact": None,
            "truncation": {"occurred": False},
            "continuation_hint": None,
        },
    )

    assert json.loads(rendered)["content"] == "ok"
