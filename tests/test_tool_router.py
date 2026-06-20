"""
tests/test_tool_router.py - ToolRouter 本地工具行为测试

验证工具授权、trace 写入、文件工具和 shell 工具的结构化结果。
"""

import json
from pathlib import Path

from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.tools.registry import TOOL_REGISTRY
from agentfoundry.tools.router import ToolRouter


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
    assert record["policy"] == {
        "tool_name": "fake_tool",
        "risk_level": "low",
        "action": "allow",
        "reason": "policy allows low risk tool fake_tool",
        "approval": {
            "required": False,
            "status": "not_required",
            "reason": "approval not required for low risk tool fake_tool",
        },
    }


def test_tool_router_handlers_match_tool_registry(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=list(TOOL_REGISTRY),
        episode_writer=writer,
        workspace_root=tmp_path,
    )

    assert set(router._handlers) == set(TOOL_REGISTRY)


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


def test_tool_router_rejects_missing_required_argument_and_writes_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_search"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_search", {})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert "missing required argument: query" in result["error"]["message"]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "file_search"
    assert record["status"] == "error"
    assert record["error"]["type"] == "tool_argument_invalid"


def test_tool_router_rejects_argument_type_mismatch(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "offset": "1"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert "argument offset must be integer" in result["error"]["message"]


def test_tool_router_rejects_extra_argument_when_schema_disallows_it(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_search"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_search", {"query": "needle", "extra": True})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert "unexpected argument: extra" in result["error"]["message"]


def test_tool_router_does_not_call_handler_when_schema_validation_fails(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args):
        calls.append(args)
        return {"status": "success"}

    router._handlers["file_read"] = handler

    result = router.dispatch("file_read", {"path": "notes.txt", "extra": True})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert calls == []


def test_tool_router_denies_high_risk_shell_before_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["shell"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args):
        calls.append(args)
        return {"status": "success"}

    router._handlers["shell"] = handler

    result = router.dispatch("shell", {"command": "echo ok", "timeout_seconds": 1.5})

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert "policy denies high risk tool shell" in result["error"]["message"]
    assert "approval not allowed" in result["error"]["message"]
    assert calls == []
    record = _read_single_tool_call(writer)
    assert record["status"] == "error"
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"] == {
        "required": True,
        "status": "missing",
        "reason": "approval not allowed for high risk tool shell",
    }
    assert record["error"]["type"] == "policy_denied"


def test_tool_router_unknown_tool_keeps_existing_failure_semantics(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["mystery_tool"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("mystery_tool", {})

    assert result["status"] == "error"
    assert result["error"]["type"] == "unknown_tool"


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


def test_apply_patch_is_denied_before_handler(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args):
        calls.append(args)
        return {"status": "success"}

    router._handlers["apply_patch"] = handler

    result = router.dispatch(
        "apply_patch",
        {"path": str(outside), "old_text": "old", "new_text": "new"},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert calls == []
    assert outside.read_text(encoding="utf-8") == "old"
    record = _read_single_tool_call(writer)
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"]["required"] is True
    assert record["policy"]["approval"]["reason"] == "approval not allowed for high risk tool apply_patch"
    assert record["error"]["type"] == "policy_denied"


def test_apply_patch_denial_writes_tool_call_error(tmp_path: Path) -> None:
    target = tmp_path / "change.txt"
    target.write_text("before\nold value\nafter\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch(
        "apply_patch",
        {"path": "change.txt", "old_text": "old value", "new_text": "new value"},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert target.read_text(encoding="utf-8") == "before\nold value\nafter\n"
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "apply_patch"
    assert record["status"] == "error"
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"]["status"] == "missing"
    assert record["error"]["type"] == "policy_denied"


def test_shell_policy_denial_writes_tool_call_error(tmp_path: Path) -> None:
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
    assert result["error"]["type"] == "policy_denied"
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "shell"
    assert record["status"] == "error"
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"]["required"] is True
    assert record["policy"]["approval"]["status"] == "missing"
    assert record["error"]["type"] == "policy_denied"


def test_policy_denies_high_risk_tool_and_records_reason(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["shell"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("shell", {"command": "python -c \"print('ok')\"", "timeout_seconds": 5})

    assert result["status"] == "error"
    record = _read_single_tool_call(writer)
    assert record["policy"] == {
        "tool_name": "shell",
        "risk_level": "high",
        "action": "deny",
        "reason": "policy denies high risk tool shell",
        "approval": {
            "required": True,
            "status": "missing",
            "reason": "approval not allowed for high risk tool shell",
        },
    }
    assert record["error"]["type"] == "policy_denied"


def test_policy_allowed_high_risk_tool_still_denies_but_records_allowed_missing(
    tmp_path: Path,
) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
    )
    calls = []

    def handler(args):
        calls.append(args)
        return {"status": "success"}

    router._handlers["shell"] = handler

    result = router.dispatch("shell", {"command": "echo blocked"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert "approval allowed but missing" in result["error"]["message"]
    assert calls == []
    record = _read_single_tool_call(writer)
    assert record["policy"]["approval"] == {
        "required": True,
        "status": "missing",
        "reason": "approval allowed but missing for high risk tool shell",
    }


def test_shell_denial_happens_before_argument_validation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["shell"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch(
        "shell",
        {
            "timeout_seconds": "slow",
        },
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"


def _read_single_tool_call(writer: EpisodeWriter) -> dict[str, object]:
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    return json.loads(trace)
