"""
tests/test_tool_router.py - ToolRouter 本地工具行为测试

验证工具授权、trace 写入、文件工具和 shell 工具的结构化结果。
"""

import json
from pathlib import Path

from haagent.runtime.human_interaction import HumanInteractionResponse
from haagent.runtime.episode import EpisodeWriter
from haagent.tools.registry import TOOL_REGISTRY
from haagent.tools.router import ToolRouter
from haagent.tools.shell import shell


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
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["line_count"] == 4


def test_file_read_keyword_reads_near_first_match(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("zero\none\nneedle here\nthree\nfour\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "keyword": "needle", "limit": 3})

    assert result["status"] == "success"
    assert result["keyword"] == "needle"
    assert result["start_line"] == 2
    assert result["end_line"] == 4
    assert result["content"] == "one\nneedle here\nthree\n"


def test_file_read_keyword_miss_returns_structured_error(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "keyword": "needle"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "keyword_not_found"
    assert "needle" in result["error"]["message"]


def test_file_read_missing_path_returns_short_argument_error(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "note.txt"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert result["error"]["message"] == "path does not exist: note.txt; path is relative to workspace_root"
    assert result["suggestions"] == ["notes.txt"]
    assert str(tmp_path) not in result["error"]["message"]


def test_file_list_defaults_to_workspace_root_and_writes_trace(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {})

    assert result["status"] == "success"
    assert result["path"] == "."
    assert result["max_depth"] == 2
    assert result["max_entries"] == 100
    assert result["truncated"] is False
    assert "README.md" in result["tree"]
    assert "src/" in result["tree"]
    assert "src/app.py" in result["tree"]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "file_list"
    assert record["status"] == "success"
    assert record["result"] == result


def test_file_list_supports_relative_path(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "outside.py").write_text("value = 2\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {"path": "pkg"})

    assert result["status"] == "success"
    assert result["path"] == "pkg"
    assert "module.py" in result["tree"]
    assert "outside.py" not in result["tree"]


def test_file_list_rejects_path_outside_workspace_root(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {"path": ".."})

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "message": "path must stay inside workspace_root; path is relative to workspace_root",
    }


def test_file_list_truncates_many_files(tmp_path: Path) -> None:
    for index in range(10):
        (tmp_path / f"file_{index:02d}.txt").write_text("x\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {"max_entries": 3})

    assert result["status"] == "success"
    assert result["entry_count"] == 3
    assert result["truncated"] is True
    assert "file_00.txt" in result["tree"]
    assert "file_03.txt" not in result["tree"]


def test_file_list_skips_noise_directories_by_default(tmp_path: Path) -> None:
    for directory in [".git", ".runs", ".smoke-runs", ".venv", "__pycache__", "node_modules", "dist", "build"]:
        noise_dir = tmp_path / directory
        noise_dir.mkdir()
        (noise_dir / "noise.txt").write_text("ignore me\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {})

    assert result["status"] == "success"
    assert "src/app.py" in result["tree"]
    assert "noise.txt" not in result["tree"]
    assert set(result["skipped_dirs"]) == {
        ".git",
        ".runs",
        ".smoke-runs",
        ".venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    }


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


def test_file_search_missing_root_returns_short_argument_error(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_search"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_search", {"query": "needle", "root": "missing"})

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "message": 'root does not exist: missing; root is relative to workspace_root; use "." or omit root',
    }
    assert str(tmp_path) not in result["error"]["message"]


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


def test_apply_patch_missing_path_returns_short_argument_error(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch"],
        approved_tools=["apply_patch"],
    )

    result = router.dispatch(
        "apply_patch",
        {"path": "missing.txt", "old_text": "old", "new_text": "new"},
    )

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "message": "path does not exist: missing.txt; path is relative to workspace_root",
    }
    assert str(tmp_path) not in result["error"]["message"]


def test_file_write_create_overwrite_and_append(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    created = router.dispatch("file_write", {"path": "notes.txt", "content": "hello\n", "mode": "create"})
    create_existing = router.dispatch("file_write", {"path": "notes.txt", "content": "nope", "mode": "create"})
    overwritten = router.dispatch("file_write", {"path": "notes.txt", "content": "new", "mode": "overwrite"})
    appended = router.dispatch("file_write", {"path": "notes.txt", "content": "\nmore", "mode": "append"})

    assert created["status"] == "success"
    assert created["created"] is True
    assert created["bytes_written"] == len("hello\n".encode("utf-8"))
    assert create_existing["status"] == "error"
    assert create_existing["error"]["type"] == "file_exists"
    assert overwritten["status"] == "success"
    assert overwritten["created"] is False
    assert appended["status"] == "success"
    assert appended["created"] is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\nmore"
    trace_lines = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool_name"] for line in trace_lines] == ["file_write"] * 4


def test_file_write_rejects_workspace_escape_and_missing_parent(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    escaped = router.dispatch("file_write", {"path": "../file_write_outside.txt", "content": "x", "mode": "create"})
    missing_parent = router.dispatch("file_write", {"path": "missing/notes.txt", "content": "x", "mode": "create"})

    assert escaped["status"] == "error"
    assert escaped["error"]["type"] == "tool_argument_invalid"
    assert missing_parent["status"] == "error"
    assert missing_parent["error"]["type"] == "tool_argument_invalid"
    assert not (tmp_path.parent / "file_write_outside.txt").exists()


def test_file_write_append_requires_existing_file(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch("file_write", {"path": "notes.txt", "content": "x", "mode": "append"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "file_not_found"


def test_file_write_denied_before_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_write"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_write", {"path": "notes.txt", "content": "x", "mode": "create"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert not (tmp_path / "notes.txt").exists()
    record = _read_single_tool_call(writer)
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"]["reason"] == "approval not allowed for high risk tool file_write"


def test_code_run_success_nonzero_timeout_and_truncation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["code_run"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["code_run"],
        approved_tools=["code_run"],
    )

    success = router.dispatch("code_run", {"code": "print('ok')", "timeout_seconds": 5})
    failed = router.dispatch("code_run", {"code": "import sys\nprint('bad')\nsys.exit(7)", "timeout_seconds": 5})
    timeout = router.dispatch("code_run", {"code": "import time\ntime.sleep(2)", "timeout_seconds": 0.1})
    long_output = router.dispatch("code_run", {"code": "print('x' * 5000)", "timeout_seconds": 5})

    assert success["status"] == "success"
    assert success["exit_code"] == 0
    assert success["stdout_excerpt"] == "ok\n"
    assert success["script_path"].startswith(".haagent-tmp/")
    assert failed["status"] == "error"
    assert failed["exit_code"] == 7
    assert failed["error"]["type"] == "code_run_failed"
    assert timeout["status"] == "error"
    assert timeout["exit_code"] is None
    assert timeout["error"]["type"] == "timeout"
    assert long_output["status"] == "success"
    assert long_output["truncated"] is True
    assert len(long_output["stdout_excerpt"]) < 5000


def test_code_run_cwd_is_workspace_bound(tmp_path: Path) -> None:
    subdir = tmp_path / "pkg"
    subdir.mkdir()
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["code_run"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["code_run"],
        approved_tools=["code_run"],
    )

    success = router.dispatch(
        "code_run",
        {"code": "from pathlib import Path\nprint(Path.cwd().name)", "cwd": "pkg", "timeout_seconds": 5},
    )
    escaped = router.dispatch("code_run", {"code": "print('x')", "cwd": "..", "timeout_seconds": 5})

    assert success["status"] == "success"
    assert success["stdout_excerpt"].strip() == "pkg"
    assert escaped["status"] == "error"
    assert escaped["error"]["type"] == "tool_argument_invalid"


def test_code_run_denied_before_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["code_run"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("code_run", {"code": "print('x')", "timeout_seconds": 5})

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert not (tmp_path / ".haagent-tmp").exists()
    record = _read_single_tool_call(writer)
    assert record["policy"]["approval"]["reason"] == "approval not allowed for high risk tool code_run"


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


def test_request_user_input_tool_uses_interaction_handler_and_writes_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["request_user_input"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )
    requests = []

    def interaction_handler(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="Use src/app.py")

    result = router.dispatch(
        "request_user_input",
        {"question": "Which file should I edit?", "reason": "Need target"},
        interaction_handler=interaction_handler,
    )

    assert result == {
        "status": "success",
        "question": "Which file should I edit?",
        "answer": "Use src/app.py",
        "answer_chars": len("Use src/app.py"),
    }
    assert len(requests) == 1
    assert requests[0].interaction_type == "user_input"
    assert requests[0].tool_name == "request_user_input"
    assert requests[0].question == "Which file should I edit?"
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "request_user_input"
    assert record["status"] == "success"
    assert record["policy"]["action"] == "allow"


def test_high_risk_tool_with_missing_approval_prompts_and_runs_when_granted(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
    )
    requests = []

    def interaction_handler(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="yes")

    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "approved", "mode": "create"},
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "success"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "approved"
    assert len(requests) == 1
    assert requests[0].interaction_type == "approval"
    assert requests[0].tool_name == "file_write"
    assert requests[0].args_summary == {"content_chars": 8, "mode": "create", "path": "notes.txt"}
    record = _read_single_tool_call(writer)
    assert record["status"] == "success"
    assert record["policy"]["action"] == "allow"
    assert record["policy"]["approval"] == {
        "required": True,
        "status": "granted",
        "reason": "approval granted for high risk tool file_write",
    }


def test_high_risk_tool_denial_does_not_call_handler_or_modify_workspace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
    )

    def interaction_handler(request):
        return HumanInteractionResponse(approved=False, answer="no")

    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "denied", "mode": "create"},
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert not (tmp_path / "notes.txt").exists()
    record = _read_single_tool_call(writer)
    assert record["status"] == "error"
    assert record["policy"]["action"] == "deny"
    assert record["policy"]["approval"] == {
        "required": True,
        "status": "denied",
        "reason": "approval denied for high risk tool file_write",
    }
    assert record["error"]["type"] == "approval_denied"


def test_approved_high_risk_tool_runs_handler_and_records_granted(
    tmp_path: Path,
) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
        approved_tools=["shell"],
    )
    calls = []

    def handler(args):
        calls.append(args)
        return {"status": "success", "approved": True}

    router._handlers["shell"] = handler

    result = router.dispatch("shell", {"command": "echo approved"})

    assert result == {"status": "success", "approved": True}
    assert calls == [{"command": "echo approved"}]
    record = _read_single_tool_call(writer)
    assert record["status"] == "success"
    assert record["policy"]["action"] == "allow"
    assert record["policy"]["approval"] == {
        "required": True,
        "status": "granted",
        "reason": "approval granted for high risk tool shell",
    }


def test_shell_uses_workspace_root_when_cwd_is_missing(tmp_path: Path) -> None:
    result = shell({"command": _print_cwd_command(), "timeout_seconds": 5}, tmp_path)

    assert result["status"] == "success"
    assert result["stdout"].strip() == str(tmp_path.resolve())


def test_shell_uses_workspace_root_when_cwd_is_dot(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "cwd": ".", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "success"
    assert result["stdout"].strip() == str(tmp_path.resolve())


def test_shell_runs_in_workspace_relative_subdirectory(tmp_path: Path) -> None:
    subdir = tmp_path / "src"
    subdir.mkdir()

    result = shell(
        {"command": _print_cwd_command(), "cwd": "src", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "success"
    assert result["stdout"].strip() == str(subdir.resolve())


def test_shell_rejects_missing_directory_cwd_with_argument_error(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "cwd": "missing", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    message = result["error"]["message"]
    assert "cwd does not exist" in result["error"]["message"]
    assert 'cwd is relative to workspace_root; use "." or omit cwd for workspace root' in message


def test_shell_rejects_cwd_outside_workspace_root(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "cwd": "..", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    message = result["error"]["message"]
    assert "cwd must stay inside workspace_root" in message
    assert 'cwd is relative to workspace_root; use "." or omit cwd for workspace root' in message


def test_shell_rejects_non_positive_timeout_with_argument_error(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "timeout_seconds": 0},
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "message": "timeout_seconds must be positive",
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


def _print_cwd_command() -> str:
    return "python -c \"from pathlib import Path; print(Path.cwd().resolve())\""
