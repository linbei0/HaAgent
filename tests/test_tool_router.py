"""
tests/test_tool_router.py - ToolRouter 本地工具行为测试

验证工具授权、trace 写入、文件工具和 shell 工具的结构化结果。
"""

import json
from pathlib import Path

from agent_foundry.episode import EpisodeWriter
from agent_foundry.tools import ToolRouter


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Route a tool
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    return EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)


def test_tool_router_runs_fake_tool_and_writes_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["fake_tool"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("fake_tool", {"value": 42})

    assert result == {"status": "success", "args": {"value": 42}}
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(trace)
    assert record["tool_name"] == "fake_tool"
    assert record["status"] == "success"
    assert record["result"] == result


def test_tool_router_rejects_disallowed_tool(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=[], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("fake_tool", {})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_not_allowed"
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(trace)
    assert record["tool_name"] == "fake_tool"
    assert record["status"] == "error"


def test_file_read_supports_offset_and_limit(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "offset": 1, "limit": 2})

    assert result["status"] == "success"
    assert result["content"] == "one\ntwo\n"
    assert result["offset"] == 1
    assert result["limit"] == 2


def test_file_search_finds_matching_text_and_writes_trace(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("needle appears here\n", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("nothing useful\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_search"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_search", {"query": "needle"})

    assert result["status"] == "success"
    assert any("alpha.txt" in match["path"] for match in result["matches"])
    trace_lines = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    assert json.loads(trace_lines[0])["tool_name"] == "file_search"


def test_apply_patch_rejects_path_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch(
        "apply_patch",
        {"path": str(outside), "old_text": "old", "new_text": "new"},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "path_outside_workspace"
    assert outside.read_text(encoding="utf-8") == "old"


def test_apply_patch_replaces_unique_text_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "change.txt"
    target.write_text("before\nold value\nafter\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch(
        "apply_patch",
        {"path": "change.txt", "old_text": "old value", "new_text": "new value"},
    )

    assert result["status"] == "success"
    assert target.read_text(encoding="utf-8") == "before\nnew value\nafter\n"


def test_shell_captures_exit_code_stdout_and_stderr(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["shell"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch(
        "shell",
        {
            "command": "python -c \"import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)\"",
            "timeout_seconds": 5,
        },
    )

    assert result["status"] == "error"
    assert result["exit_code"] == 3
    assert "out" in result["stdout"]
    assert "err" in result["stderr"]
    assert result["error"]["type"] == "command_failed"
