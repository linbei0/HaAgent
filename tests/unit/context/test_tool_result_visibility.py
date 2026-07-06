"""
tests/unit/context/test_tool_result_visibility.py - 工具结果模型可见合同测试

验证统一 ToolResultView 负责工具长结果落盘、预览和渲染。
"""

import json

from haagent.context.compression.budget import derive_compression_budget
from haagent.context.compression.tool_results import (
    prepare_tool_result_for_model,
    render_tool_result_view,
)
from haagent.context.messages import build_tool_result_message


def test_long_mcp_output_becomes_tool_result_view_with_artifact() -> None:
    saved: dict[str, str] = {}
    output = "start " + ("middle " * 2200) + "important tail"

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
    assert view["artifact"]["preview_chars"] == len(view["content"])
    assert view["truncated"] is True
    assert "start" in view["content"]
    assert "important tail" in view["content"]
    assert len(view["content"]) <= 3_000
    assert saved == {"tool_name": "mcp__fixture__fetch", "content": output}


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
            "truncated": True,
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
            "truncated": False,
            "continuation_hint": None,
        },
    )

    assert json.loads(rendered)["content"] == "ok"
