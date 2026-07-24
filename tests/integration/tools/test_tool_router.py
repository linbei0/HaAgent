"""
tests/integration/tools/test_tool_router.py - ToolRouter 本地工具行为测试

验证工具授权、trace 写入、文件工具和 shell 工具的结构化结果。
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from haagent.context.messages import build_tool_result_message
from haagent.mcp.runtime import McpRuntimeTimeoutError
from haagent.runtime.execution.human_interaction import HumanInteractionResponse
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import ReplaySafety, RetryController, RetryOperation, RetryPolicy
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.path_policy import ExternalRoot, PathPolicy
from haagent.runtime.session.attachments import ImageAttachment
from haagent.skills import SkillSettings
from haagent.tools.registry import TOOL_REGISTRY
from haagent.tools.registry import ToolDefinition, default_tool_runtime_registry
from haagent.tools.router import ToolRouter
from haagent.tools.base import tool_error
from haagent.tools import file_tools
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


class FakeSandboxBackend:
    def __init__(self) -> None:
        self.shell_commands = []
        self.python_scripts = []

    def metadata(self):
        raise AssertionError("metadata is not needed in this test")

    def run_shell(self, command):
        from haagent.runtime.execution.command import CommandResult

        self.shell_commands.append(command)
        return CommandResult(
            command=command.command,
            status="success",
            exit_code=0,
            stdout="sandbox shell",
            stderr="",
            stdout_excerpt="sandbox shell",
            stderr_excerpt="",
            stdout_truncated=False,
            stderr_truncated=False,
            truncated=False,
            timeout=False,
            redacted=False,
            duration_seconds=0.01,
            timeout_seconds=command.timeout_seconds,
        )

    def run_python(self, script_path, command):
        from haagent.runtime.execution.command import CommandResult

        self.python_scripts.append((script_path, command))
        return CommandResult(
            command=f"python {script_path}",
            status="success",
            exit_code=0,
            stdout="sandbox python",
            stderr="",
            stdout_excerpt="sandbox python",
            stderr_excerpt="",
            stdout_truncated=False,
            stderr_truncated=False,
            truncated=False,
            timeout=False,
            redacted=False,
            duration_seconds=0.01,
            timeout_seconds=command.timeout_seconds,
        )

    def close(self):
        return None


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


def test_file_list_skips_inaccessible_subdirectories(tmp_path: Path, monkeypatch) -> None:
    blocked = tmp_path / ".tmp" / "pytest"
    blocked.mkdir(parents=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    original_iterdir = Path.iterdir

    def fake_iterdir(path: Path):
        if path == blocked:
            raise PermissionError("access denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    result = file_tools.file_list(
        {"path": ".", "max_depth": 3, "max_entries": 20},
        tmp_path,
    )

    assert result["status"] == "success"
    assert ".tmp/pytest" in result["skipped_dirs"]
    assert "README.md" in result["tree"]


def test_start_memory_update_sets_runtime_flag_and_writes_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["start_memory_update"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )

    result = router.dispatch("start_memory_update", {"reason": "用户给出长期偏好"})

    assert result == {
        "status": "success",
        "memory_update_requested": True,
        "reason": "用户给出长期偏好",
    }
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "start_memory_update"
    assert record["result"]["memory_update_requested"] is True


def test_skill_list_returns_metadata_without_skill_body(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    skill_dir = home / ".haagent" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review workflow.\n---\n\nSECRET BODY",
        encoding="utf-8",
    )
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["skill_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("skill_list", {})

    assert result["status"] == "success"
    assert {
        "name": "review",
        "description": "Review workflow.",
        "source": "user",
        "command_name": "review",
        "user_invocable": True,
        "disable_model_invocation": False,
    } in result["skills"]
    assert any(skill["source"] == "builtin" for skill in result["skills"])
    assert "SECRET BODY" not in json.dumps(result, ensure_ascii=False)


def test_skill_read_returns_skill_content_and_writes_trace(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    skill_dir = home / ".haagent" / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Review\nRead files before reviewing.\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["skill_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("skill_read", {"name": "review"})

    assert result["status"] == "success"
    assert result["name"] == "Review"
    assert "Read files before reviewing." in result["content"]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "skill_read"
    assert record["status"] == "success"


def test_skill_read_blocks_model_disabled_skill(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    skill_dir = home / ".haagent" / "skills" / "grill-me"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: grill-me\ndescription: User-only session.\ndisable-model-invocation: true\n---\n\n# Body\n",
        encoding="utf-8",
    )
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["skill_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("skill_read", {"name": "grill-me"})

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "skill_model_invocation_disabled",
        "category": "execution",
        "message": "skill can only be invoked explicitly by the user: /grill-me",
        "retryable": False,
    }


def test_project_skill_list_marks_untrusted_project_roots(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    skill_dir = repo / ".haagent" / "skills" / "local"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# local\nlocal workflow\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["skill_list"],
        episode_writer=writer,
        workspace_root=repo,
        skill_settings=SkillSettings(version=1, trusted_project_roots=()),
    )

    result = router.dispatch("skill_list", {})

    assert result["status"] == "success"
    assert result["skills"]
    assert all(skill["source"] == "builtin" for skill in result["skills"])
    assert not any(skill["name"] == "local" for skill in result["skills"])
    assert result["blocked_project_skill_roots"] == [str((repo / ".haagent" / "skills").resolve())]


def test_skill_market_search_runs_through_router_and_writes_trace(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_skill_market_search(args: dict[str, object]) -> dict[str, object]:
        calls.append(args)
        return {
            "status": "success",
            "query": args["query"],
            "results": [
                {
                    "result_id": "skills_sh-1",
                    "provider": "skills_sh",
                    "name": "csv",
                    "source": "vercel-labs/bash-tool",
                    "summary": "",
                    "detail_url": "https://skills.sh/vercel-labs/bash-tool/csv",
                    "installable": True,
                    "quality": {},
                }
            ],
            "warnings": [],
        }

    monkeypatch.setattr(
        "haagent.tools.contributions.skills.skill_market_search",
        fake_skill_market_search,
    )
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["skill_market_search"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("skill_market_search", {"query": "csv", "providers": ["skills_sh"], "limit": 1})

    assert calls == [{"query": "csv", "providers": ["skills_sh"], "limit": 1}]
    assert result["status"] == "success"
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "skill_market_search"
    assert record["status"] == "success"
    assert record["result"] == result


def test_medium_risk_web_fetch_runs_without_high_risk_approval(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["web_fetch"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {
            "status": "success",
            "final_url": args["url"],
            "status_code": 200,
            "content_type": "text/plain",
            "content": "ok",
            "truncated": False,
        }

    router._handlers["web_fetch"] = handler

    result = router.dispatch("web_fetch", {"url": "https://example.com/"})

    assert result["status"] == "success"
    assert calls == [{"url": "https://example.com/"}]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "web_fetch"
    assert record["policy"] == {
        "tool_name": "web_fetch",
        "risk_level": "medium",
        "action": "allow",
        "reason": "policy allows medium risk tool web_fetch",
        "approval": {
            "required": False,
            "status": "not_required",
            "reason": "approval not required for medium risk tool web_fetch",
        },
    }


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
    router = ToolRouter(allowed_tools=["grep"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("grep", {})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert "missing required argument: pattern" in result["error"]["message"]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "grep"
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
    router = ToolRouter(allowed_tools=["grep"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("grep", {"pattern": "needle", "extra": True})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert "unexpected argument: extra" in result["error"]["message"]


def test_tool_router_does_not_call_handler_when_schema_validation_fails(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success"}

    router._handlers["file_read"] = handler

    result = router.dispatch("file_read", {"path": "notes.txt", "extra": True})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert calls == []


def test_tool_guardrail_blocks_shell_secret_exfiltration_before_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
        approved_tools=["shell"],
    )
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success"}

    router._handlers["shell"] = handler

    result = router.dispatch("shell", {"command": "cat ~/.ssh/id_rsa && echo $OPENAI_API_KEY"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "guardrail_denied"
    assert "guardrail shell_secret_exfiltration" in result["error"]["message"]
    assert calls == []
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "shell"
    assert record["status"] == "error"
    assert record["error"]["type"] == "guardrail_denied"
    assert record["guardrail"]["scope"] == "tool_input"
    assert record["guardrail"]["rule_id"] == "shell_secret_exfiltration"


def test_tool_router_denies_high_risk_shell_before_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["shell"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args, _context=None):
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


class FakeMcpRuntime:
    def __init__(self, output: str = "echo:hi") -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.output = output

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> str:
        del kwargs
        self.calls.append((server_name, tool_name, arguments))
        return self.output

    def list_resources(self) -> list[object]:
        return []

    def read_resource(self, server_name: str, uri: str) -> str:
        return "resource text"


class TimeoutMcpRuntime(FakeMcpRuntime):
    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, object],
        **kwargs: object,
    ) -> str:
        raise McpRuntimeTimeoutError("MCP tool fixture.echo timed out after 0.05 seconds")


def test_record_skipped_writes_trace_without_running_handler(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_read"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )
    skipped = {
        "status": "error",
        "error": {
            "type": "tool_call_skipped",
            "message": (
                "tool call was not started because an earlier call "
                "in the same model response failed"
            ),
        },
        "execution_state": "not_started",
    }

    result = router.record_skipped("file_read", {"path": "a.txt"}, skipped)

    assert result == skipped
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "file_read"
    assert record["status"] == "error"
    assert record["error"] == skipped["error"]
    assert record["policy"] is None
    assert record["duration_seconds"] == 0


def test_record_skipped_rejects_non_skipped_result(tmp_path: Path) -> None:
    from haagent.tools.base import ToolRoutingError

    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_read"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )

    with pytest.raises(ToolRoutingError, match="record_skipped only accepts"):
        router.record_skipped("file_read", {}, {"status": "success"})


def test_tool_router_dispatches_dynamic_mcp_tool_through_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
    )
    runtime = FakeMcpRuntime()
    router = ToolRouter(
        allowed_tools=["mcp__fixture__echo"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approved_tools=["mcp__fixture__echo"],
        tool_registry=default_tool_runtime_registry({"mcp__fixture__echo": dynamic}),
        mcp_runtime=runtime,
    )

    result = router.dispatch("mcp__fixture__echo", {"text": "hi"})

    assert result["status"] == "success"
    assert result["output"] == "echo:hi"
    assert result["model_visible"]["kind"] == "tool_result_view"
    assert result["model_visible"]["content"] == "echo:hi"
    assert result["model_visible"]["artifact"] is None
    assert runtime.calls == [("fixture", "echo", {"text": "hi"})]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "mcp__fixture__echo"
    assert record["status"] == "success"
    assert record["policy"]["risk_level"] == "high"


def test_tool_router_offloads_large_mcp_output_for_model_visibility(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    dynamic = ToolDefinition(
        name="mcp__fixture__fetch",
        description="Fetch remote content",
        risk_level="high",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
        execution_effect="external_effect",
    )
    large_output = "start " + ("middle " * 2200) + "important tail"
    runtime = FakeMcpRuntime(output=large_output)
    router = ToolRouter(
        allowed_tools=["mcp__fixture__fetch"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approved_tools=["mcp__fixture__fetch"],
        tool_registry=default_tool_runtime_registry({"mcp__fixture__fetch": dynamic}),
        mcp_runtime=runtime,
    )

    result = router.dispatch("mcp__fixture__fetch", {"url": "https://example.com"})

    visible = result["model_visible"]
    assert visible["kind"] == "tool_result_view"
    assert visible["truncated"] is True
    assert visible["artifact"]["original_chars"] == len(large_output)
    assert "start" in visible["content"]
    assert "important tail" in visible["content"]
    assert len(visible["content"]) < len(large_output)
    artifact_path = tmp_path / visible["artifact"]["path"]
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == large_output
    message = build_tool_result_message("call_large", "mcp__fixture__fetch", result)
    assert large_output not in message["content"]
    assert str(visible["artifact"]["path"]) in message["content"]


def test_tool_router_reports_dynamic_mcp_timeout_as_tool_error(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={"type": "object", "properties": {}, "required": []},
        execution_effect="external_effect",
    )
    router = ToolRouter(
        allowed_tools=["mcp__fixture__echo"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approved_tools=["mcp__fixture__echo"],
        tool_registry=default_tool_runtime_registry({"mcp__fixture__echo": dynamic}),
        mcp_runtime=TimeoutMcpRuntime(),
    )

    result = router.dispatch("mcp__fixture__echo", {})

    assert result["status"] == "error"
    assert result["execution_state"] == "unknown"
    assert result["error"] == {
        "type": "mcp_timeout",
        "category": "execution",
        "message": "MCP tool fixture.echo timed out after 0.05 seconds",
        "retryable": False,
    }


def test_tool_router_executes_handler_once_through_retry_controller(tmp_path: Path) -> None:
    class RecordingRetryController(RetryController):
        def __init__(self) -> None:
            super().__init__()
            self.operations: list[RetryOperation] = []

        def execute(self, operation: RetryOperation, invoke, **kwargs):
            self.operations.append(operation)
            return invoke()

    writer = make_writer(tmp_path)
    retry_controller = RecordingRetryController()
    router = ToolRouter(
        allowed_tools=["fake_tool"],
        episode_writer=writer,
        workspace_root=tmp_path,
        retry_controller=retry_controller,
    )
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success", "handled": True}

    router._handlers["fake_tool"] = handler

    result = router.dispatch("fake_tool", {"value": 42})

    assert result == {"status": "success", "handled": True}
    assert calls == [{"value": 42}]
    assert retry_controller.operations == [
        RetryOperation("tool.fake_tool", ReplaySafety.NEVER_REPLAY),
    ]


def test_safe_read_tool_recovers_from_retryable_failure_and_traces_attempts(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_read"],
        episode_writer=writer,
        workspace_root=tmp_path,
        retry_controller=RetryController(
            RetryPolicy(
                max_attempts=3,
                minimum_delay_seconds=0,
                base_delay_seconds=0,
                throttling_base_delay_seconds=0,
                max_delay_seconds=0,
            ),
            sleep=lambda _seconds: None,
            random_value=lambda: 0,
        ),
    )
    calls = 0

    def handler(_args, _context=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tool_error("temporary_io_error", "temporary read interruption", retryable=True)
        return {"status": "success", "content": "recovered"}

    router._handlers["file_read"] = handler

    result = router.dispatch("file_read", {"path": "notes.txt"})

    assert result == {"status": "success", "content": "recovered"}
    assert calls == 2
    trace = json.loads((writer.path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert trace["attempt_count"] == 2
    assert trace["recovered_after_retry"] is True
    assert [event["category"] for event in trace["retry_events"]] == ["transient"]


def test_write_tool_never_replays_even_if_handler_marks_failure_retryable(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
        retry_controller=RetryController(
            RetryPolicy(max_attempts=3, minimum_delay_seconds=0),
            sleep=lambda _seconds: None,
        ),
    )
    calls = 0

    def handler(_args, _context=None):
        nonlocal calls
        calls += 1
        return tool_error("temporary_io_error", "temporary write interruption", retryable=True)

    router._handlers["file_write"] = handler

    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "hello", "mode": "create"},
    )

    assert result["status"] == "error"
    assert calls == 1
    trace = json.loads((writer.path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert trace["attempt_count"] == 1
    assert "retry_events" not in trace


def test_malformed_handler_result_is_reported_as_contract_failure(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["fake_tool"], episode_writer=writer, workspace_root=tmp_path)
    router._handlers["fake_tool"] = lambda _args, _context=None: {
        "status": "error",
        "error": {"type": "broken"},
    }

    result = router.dispatch("fake_tool", {"value": 1})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_contract_invalid"
    assert result["error"]["category"] == "contract"
    assert result["recovery"]["action"] == "stop"


def test_approved_high_risk_tool_validates_arguments_before_execution(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
    )
    calls = []
    requests = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success"}

    def interaction_handler(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="yes")

    router._handlers["shell"] = handler

    result = router.dispatch("shell", {}, interaction_handler=interaction_handler)

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert len(requests) == 1
    assert calls == []


def test_load_image_attachment_returns_registered_history_image(tmp_path: Path) -> None:
    session_root = tmp_path / "session"
    image_path = session_root / "attachments" / "img-one.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-bytes")
    attachment = ImageAttachment(
        id="img-one",
        filename="img-one.png",
        mime_type="image/png",
        size_bytes=11,
        width=2,
        height=1,
        sha256="a" * 64,
        relative_path="attachments/img-one.png",
        base_path=str(session_root),
    )
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["load_image_attachment"],
        episode_writer=writer,
        workspace_root=tmp_path,
        image_attachment_history=[attachment],
    )

    result = router.dispatch("load_image_attachment", {"image_id": "img-one"})

    assert result["status"] == "success"
    assert result["loaded_image_attachment"] == {
        **attachment.to_dict(),
        "type": "image_attachment",
        "path": str(image_path.resolve()),
    }
    assert result["model_visible"]["image_id"] == "img-one"
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(trace)
    assert record["tool_name"] == "load_image_attachment"
    assert record["status"] == "success"
    assert "path" not in record["result"]["loaded_image_attachment"]
    assert record["result"]["loaded_image_attachment"]["relative_path"] == "attachments/img-one.png"
    assert "base64" not in trace


def test_load_image_attachment_reports_unknown_image_id(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["load_image_attachment"],
        episode_writer=writer,
        workspace_root=tmp_path,
        image_attachment_history=[],
    )

    result = router.dispatch("load_image_attachment", {"image_id": "missing"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "image_attachment_not_found"
    record = _read_single_tool_call(writer)
    assert record["status"] == "error"
    assert record["error"]["type"] == "image_attachment_not_found"


def test_shell_uses_sandbox_backend_when_provided(tmp_path: Path) -> None:
    backend = FakeSandboxBackend()
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
        approved_tools=["shell"],
        sandbox_backend=backend,
    )

    result = router.dispatch("shell", {"command": "echo hi", "timeout_seconds": 5})

    assert result["status"] == "success"
    assert result["stdout_excerpt"] == "sandbox shell"
    assert backend.shell_commands[0].command == "echo hi"
    assert backend.shell_commands[0].cwd == tmp_path.resolve()
    assert backend.shell_commands[0].timeout_seconds == 5


def test_code_run_uses_sandbox_backend_and_system_temp_script(tmp_path: Path) -> None:
    backend = FakeSandboxBackend()
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["code_run"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["code_run"],
        approved_tools=["code_run"],
        sandbox_backend=backend,
    )

    result = router.dispatch("code_run", {"code": "print('hi')", "timeout_seconds": 5})

    assert result["status"] == "success"
    assert result["stdout_excerpt"] == "sandbox python"
    script_path, command = backend.python_scripts[0]
    # Fake backend 在执行瞬间看到脚本；handler 返回后临时文件应已清理。
    assert not script_path.exists()
    assert not script_path.is_relative_to(tmp_path.resolve())
    assert command.cwd == tmp_path.resolve()
    assert not (tmp_path / ".haagent-tmp").exists()


def test_tool_router_does_not_swallow_run_cancelled_for_mcp_tool(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    token = CancellationToken()
    token.cancel()
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={"type": "object", "properties": {}, "required": []},
        execution_effect="external_effect",
    )
    router = ToolRouter(
        allowed_tools=["mcp__fixture__echo"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approved_tools=["mcp__fixture__echo"],
        cancellation_token=token,
        tool_registry=default_tool_runtime_registry({"mcp__fixture__echo": dynamic}),
        mcp_runtime=FakeMcpRuntime(),
    )

    with pytest.raises(RunCancelled):
        router.dispatch("mcp__fixture__echo", {})
    record = _read_single_tool_call(writer)
    assert record["status"] == "error"
    assert record["error"]["type"] == "RunCancelled"


def test_dynamic_mcp_tool_defaults_to_high_risk_approval(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={"type": "object", "properties": {}, "required": []},
        execution_effect="external_effect",
    )
    router = ToolRouter(
        allowed_tools=["mcp__fixture__echo"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["mcp__fixture__echo"],
        tool_registry=default_tool_runtime_registry({"mcp__fixture__echo": dynamic}),
        mcp_runtime=FakeMcpRuntime(),
    )

    result = router.dispatch("mcp__fixture__echo", {})

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    record = _read_single_tool_call(writer)
    assert record["policy"]["approval"]["status"] == "missing"


def test_mcp_resource_tools_dispatch_through_runtime(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["read_mcp_resource"],
        episode_writer=writer,
        workspace_root=tmp_path,
        tool_registry=default_tool_runtime_registry(),
        mcp_runtime=FakeMcpRuntime(),
    )

    result = router.dispatch("read_mcp_resource", {"server": "fixture", "uri": "fixture://hello"})

    assert result == {"status": "success", "output": "resource text"}


def test_file_read_supports_offset_and_limit(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "offset": 1, "limit": 2})

    assert result["status"] == "success"
    assert result["path"] == "notes.txt"
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


def test_file_read_model_visible_includes_range_and_truncation_diagnostics(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("zero\none\ntwo\nthree\nfour\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "offset": 1, "limit": 2})

    visible = result["model_visible"]
    assert visible == {
        "path": "notes.txt",
        "offset": 1,
        "limit": 2,
        "keyword": None,
        "start_line": 2,
        "end_line": 3,
        "line_count": 5,
        "content": "one\ntwo\n",
        "truncated": True,
        "truncation_reason": "requested_range_excludes_file_lines",
    }


def test_file_read_raw_content_is_not_serialized_to_tool_message_when_model_visible_exists(tmp_path: Path) -> None:
    from haagent.context.messages import build_tool_result_message

    target = tmp_path / "notes.txt"
    target.write_text("raw-only-line\nvisible-line\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "notes.txt", "offset": 0, "limit": 1})
    result["model_visible"]["content"] = "visible excerpt"

    message = build_tool_result_message("call_1", "file_read", result)

    assert "visible excerpt" in message["content"]
    assert "raw-only-line" not in message["content"]


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
    assert result["recovery"]["action"] == "use_tool"
    assert result["recovery"]["tool_name"] == "file_read"
    assert result["recovery"]["args"] == {"path": "notes.txt"}
    assert str(tmp_path) not in result["error"]["message"]


def test_file_read_directory_error_suggests_file_list(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_read", {"path": "src"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert result["error"]["message"] == "path must be a file: src; path is relative to workspace_root"
    assert result["recovery"]["action"] == "use_tool"
    assert result["recovery"]["tool_name"] == "file_list"
    assert result["recovery"]["args"] == {"path": "src", "max_depth": 1}


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
    assert result["error"]["type"] == "approval_required"
    assert "用户确认" in result["error"]["message"]


def test_file_read_allow_once_requests_external_directory_again(tmp_path: Path) -> None:
    external = tmp_path.parent / "haagent-external-read"
    external.mkdir()
    target = external / "notes.txt"
    target.write_text("external content", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="once")

    first = router.dispatch(
        "file_read",
        {"path": str(target)},
        interaction_handler=interaction,
    )
    second = router.dispatch(
        "file_read",
        {"path": str(target)},
        interaction_handler=interaction,
    )

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert len(requests) == 2
    assert requests[0].tool_name == "external_directory"
    assert requests[0].args_summary["access"] == "read"


def test_file_read_always_reuses_external_directory_authorization(tmp_path: Path) -> None:
    external = tmp_path.parent / "haagent-external-read-always"
    external.mkdir()
    target = external / "notes.txt"
    target.write_text("external content", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=tmp_path)
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="always")

    first = router.dispatch("file_read", {"path": str(target)}, interaction_handler=interaction)
    second = router.dispatch("file_read", {"path": str(target)})

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert len(requests) == 1


def test_file_read_haagent_own_artifact_skips_external_permission(tmp_path: Path) -> None:
    """HaAgent 自身 episode artifacts/tool-results 的只读回读不触发外部目录权限。

    HaAgent 主动把超长工具结果落盘到 ~/.haagent/runs/episodes/**/artifacts/tool-results/
    并引导模型用 file_read 回读；该路径在 workspace 外但属于内部可信内容。
    """
    artifact_root = (
        tmp_path
        / ".haagent"
        / "runs"
        / "episodes"
        / "2026"
        / "07"
        / "24"
        / "session-x"
        / "turn-1"
        / "artifacts"
        / "tool-results"
    )
    artifact_root.mkdir(parents=True)
    target = artifact_root / "mcp__exa__web_search_exa-abcd1234.txt"
    target.write_text("full web search output", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    writer = make_writer(workspace)
    router = ToolRouter(allowed_tools=["file_read"], episode_writer=writer, workspace_root=workspace)
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=False, answer="once")

    result = router.dispatch(
        "file_read",
        {"path": str(target)},
        interaction_handler=interaction,
    )

    assert result["status"] == "success"
    assert "full web search output" in result["content"]
    assert requests == []


def test_file_write_haagent_own_artifact_still_requires_permission(tmp_path: Path) -> None:
    """写入 HaAgent artifacts 目录不被豁免，仍走外部目录审批。"""
    artifact_root = (
        tmp_path
        / ".haagent"
        / "runs"
        / "episodes"
        / "2026"
        / "07"
        / "24"
        / "session-x"
        / "turn-1"
        / "artifacts"
        / "tool-results"
    )
    artifact_root.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    writer = make_writer(workspace)
    router = ToolRouter(allowed_tools=["file_write"], episode_writer=writer, workspace_root=workspace)

    result = router.dispatch(
        "file_write",
        {"path": str(artifact_root / "evil.txt"), "content": "x"},
    )

    assert result["status"] == "error"
    # 写入不被只读豁免覆盖，policy 层直接拒绝（无审批入口时不升级）。
    assert result["error"]["type"] in {"approval_required", "policy_denied"}


def test_external_read_root_allows_file_read_but_denies_file_write(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    target = external / "notes.txt"
    target.write_text("hello\n", encoding="utf-8")
    writer = make_writer(project)
    router = ToolRouter(
        allowed_tools=["file_read", "file_write"],
        episode_writer=writer,
        workspace_root=project,
        path_policy=PathPolicy(
            project_root=project,
            external_roots=[ExternalRoot(path=external, access="read", source="user", created_at="now")],
        ),
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    read_result = router.dispatch("file_read", {"path": str(target)})
    write_result = router.dispatch(
        "file_write",
        {"path": str(target), "content": "changed\n", "mode": "overwrite"},
    )

    assert read_result["status"] == "success"
    assert read_result["content"] == "hello\n"
    assert write_result["status"] == "error"
    assert write_result["error"]["type"] == "approval_required"
    assert "用户确认" in write_result["error"]["message"]
    assert target.read_text(encoding="utf-8") == "hello\n"


def test_external_authorized_grep_returns_absolute_readable_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    target = external / "notes.txt"
    target.write_text("external needle\n", encoding="utf-8")
    writer = make_writer(project)
    router = ToolRouter(
        allowed_tools=["grep", "file_read"],
        episode_writer=writer,
        workspace_root=project,
        path_policy=PathPolicy(
            project_root=project,
            external_roots=[ExternalRoot(path=external, access="read", source="user", created_at="now")],
        ),
    )

    searched = router.dispatch("grep", {"pattern": "needle", "path": str(external)})

    assert searched["status"] == "success"
    assert searched["matches"][0]["path"] == str(target.resolve())
    read_back = router.dispatch("file_read", {"path": searched["matches"][0]["path"]})
    assert read_back["status"] == "success"
    assert read_back["content"] == "external needle\n"


def test_full_access_mode_allows_file_write_outside_workspace_and_traces_mode(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    target = external / "notes.txt"
    writer = make_writer(project)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=project,
        path_policy=PathPolicy(project_root=project, permission_mode="full_access"),
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {"path": str(target), "content": "hello\n", "mode": "create"},
    )

    assert result["status"] == "success"
    assert target.read_text(encoding="utf-8") == "hello\n"
    record = _read_single_tool_call(writer)
    assert record["path_policy"]["permission_mode"] == "full_access"


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
    for directory in [".git", ".runs", ".venv", "__pycache__", "node_modules", "dist", "build"]:
        noise_dir = tmp_path / directory
        noise_dir.mkdir()
        (noise_dir / "noise.txt").write_text("ignore me\n", encoding="utf-8")
    (tmp_path / ".smoke-runs").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["file_list"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("file_list", {})

    assert result["status"] == "success"
    assert "src/app.py" in result["tree"]
    assert ".smoke-runs" in result["tree"]
    assert "noise.txt" not in result["tree"]
    assert set(result["skipped_dirs"]) == {
        ".git",
        ".runs",
        ".venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
    }


def test_grep_finds_matching_text_and_writes_trace(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("needle appears here\n", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("nothing useful\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["grep"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("grep", {"pattern": "needle"})

    assert result["status"] == "success"
    assert result["matches"][0]["path"] == "alpha.txt"
    trace_lines = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == 1
    assert json.loads(trace_lines[0])["tool_name"] == "grep"


def test_grep_python_fallback_respects_ignore_files_and_noise_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "blocked.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("needle\n", encoding="utf-8")
    original_which = file_tools.shutil.which
    monkeypatch.setattr(file_tools.shutil, "which", lambda name: None if name == "rg" else original_which(name))

    result = file_tools.grep({"pattern": "needle"}, tmp_path)

    assert result["status"] == "success"
    assert [match["path"] for match in result["matches"]] == ["kept.txt"]
    assert result["partial"] is False


def test_grep_default_command_does_not_override_ripgrep_ignore_rules(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "kept.txt").write_text("nothing\n", encoding="utf-8")
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    monkeypatch.setattr(file_tools.shutil, "which", lambda name: "rg" if name == "rg" else None)
    monkeypatch.setattr(file_tools.subprocess, "run", fake_run)

    result = file_tools.grep({"pattern": "needle"}, tmp_path)

    assert result["status"] == "success"
    assert "**/*" not in captured
    assert "--glob" not in captured


def test_grep_explicit_glob_keeps_fixed_noise_directories_excluded(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "src" / "note.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "noise.py").write_text("needle\n", encoding="utf-8")

    result = file_tools.grep({"pattern": "needle", "include": "*.py"}, tmp_path)

    assert result["status"] == "success"
    assert [match["path"] for match in result["matches"]] == ["src/app.py"]


def test_grep_permission_warning_returns_partial_matches(tmp_path: Path, monkeypatch) -> None:
    match_path = tmp_path / "kept.txt"
    match_path.write_text("needle\n", encoding="utf-8")
    output = json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": str(match_path)},
                "lines": {"text": "needle\n"},
                "line_number": 1,
                "submatches": [{"start": 0}],
            },
        },
    )

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout=output,
            stderr=f"rg: {tmp_path / '.tmp' / 'pytest'}: Access is denied. (os error 5)\n",
        )

    monkeypatch.setattr(file_tools.shutil, "which", lambda name: "rg" if name == "rg" else None)
    monkeypatch.setattr(file_tools.subprocess, "run", fake_run)

    result = file_tools.grep({"pattern": "needle"}, tmp_path)

    assert result["status"] == "success"
    assert result["partial"] is True
    assert result["matches"][0]["path"] == "kept.txt"
    assert result["warnings"][0]["type"] == "permission_denied"
    assert result["skipped_paths"] == [".tmp/pytest"]


def test_grep_non_permission_ripgrep_error_is_not_hidden(tmp_path: Path) -> None:
    result = file_tools.grep({"pattern": "["}, tmp_path)

    assert result["status"] == "error"
    assert result["error"]["type"] in {"search_failed", "tool_argument_invalid"}


def test_grep_timeout_preserves_partial_stdout_and_adds_guidance(tmp_path: Path, monkeypatch) -> None:
    match_path = tmp_path / "kept.txt"
    match_path.write_text("needle\n", encoding="utf-8")
    output = json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": str(match_path)},
                "lines": {"text": "needle\n"},
                "line_number": 1,
                "submatches": [{"start": 0}],
            },
        },
    )

    def fake_run(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, 2, output=output, stderr="")

    monkeypatch.setattr(file_tools.shutil, "which", lambda name: "rg" if name == "rg" else None)
    monkeypatch.setattr(file_tools.subprocess, "run", fake_run)

    result = file_tools.grep({"pattern": "needle", "timeout_seconds": 2}, tmp_path)

    assert result["status"] == "success"
    assert result["partial"] is True
    assert result["matches"][0]["path"] == "kept.txt"
    assert result["warnings"][0]["type"] == "timeout"
    assert "path" in result["guidance"] and "include" in result["guidance"]


def test_grep_python_fallback_uses_git_ignore_and_fixed_exclusions(tmp_path: Path, monkeypatch) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required for fallback ignore verification")
    subprocess.run([git, "init", "-q", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "noise.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "kept.txt").write_text("needle\n", encoding="utf-8")
    real_which = file_tools.shutil.which
    monkeypatch.setattr(file_tools.shutil, "which", lambda name: None if name == "rg" else real_which(name))

    result = file_tools.grep({"pattern": "needle"}, tmp_path)

    assert result["status"] == "success"
    assert result["search_backend"] == "python"
    assert [match["path"] for match in result["matches"]] == ["kept.txt"]


def test_grep_unreadable_path_is_a_hard_failure(tmp_path: Path, monkeypatch) -> None:
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    original_iterdir = Path.iterdir

    def fake_iterdir(path: Path):
        if path == blocked:
            raise PermissionError("access denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    result = file_tools.grep({"pattern": "needle", "path": "blocked"}, tmp_path)

    assert result["status"] == "error"
    assert result["error"]["type"] == "search_failed"


def test_grep_missing_path_returns_short_argument_error(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["grep"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("grep", {"pattern": "needle", "path": "missing"})

    assert result["status"] == "error"
    assert result["error"]["type"] == "tool_argument_invalid"
    assert result["error"]["category"] == "argument"
    assert result["error"]["message"] == 'path does not exist: missing; path is relative to workspace_root and may be a directory or file; use "." or omit path'
    assert result["error"]["retryable"] is False
    assert result["recovery"]["action"] == "correct_arguments"
    assert str(tmp_path) not in result["error"]["message"]


def test_grep_file_path_searches_single_file(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("needle appears here\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["grep"], episode_writer=writer, workspace_root=tmp_path)

    result = router.dispatch("grep", {"pattern": "needle", "path": "alpha.txt"})

    assert result["status"] == "success"
    assert result["matches"] == [
        {"path": "alpha.txt", "line": 1, "column": 1, "text": "needle appears here"},
    ]


def test_apply_patch_is_denied_before_handler(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args, _context=None):
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
    assert result["error"]["type"] == "tool_argument_invalid"
    assert result["error"]["category"] == "argument"
    assert result["error"]["message"] == "path does not exist: missing.txt; path is relative to workspace_root"
    assert result["error"]["retryable"] is False
    assert result["recovery"]["action"] == "correct_arguments"
    assert str(tmp_path) not in result["error"]["message"]


def test_apply_patch_set_replaces_multiple_unique_snippets_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\nold one\nomega\n", encoding="utf-8")
    second.write_text("before\nold two\nafter\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    result = router.dispatch(
        "apply_patch_set",
        {
            "replacements": [
                {"path": "first.txt", "old_text": "old one", "new_text": "new one"},
                {"path": "second.txt", "old_text": "old two", "new_text": "new two"},
            ],
        },
    )

    assert result["status"] == "success"
    assert result["replacement_count"] == 2
    assert result["changed_files"] == [
        {"path": str(first), "change_type": "modified", "additions": 1, "deletions": 1, "replacements": 1},
        {"path": str(second), "change_type": "modified", "additions": 1, "deletions": 1, "replacements": 1},
    ]
    assert [item["status"] for item in result["replacements"]] == ["success", "success"]
    assert first.read_text(encoding="utf-8") == "alpha\nnew one\nomega\n"
    assert second.read_text(encoding="utf-8") == "before\nnew two\nafter\n"


def test_apply_patch_set_single_failure_writes_no_partial_results(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\nold one\nomega\n", encoding="utf-8")
    second.write_text("before\nunchanged\nafter\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    result = router.dispatch(
        "apply_patch_set",
        {
            "replacements": [
                {"path": "first.txt", "old_text": "old one", "new_text": "new one"},
                {"path": "second.txt", "old_text": "missing", "new_text": "new two"},
            ],
        },
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "patch_text_not_found"
    assert [item["status"] for item in result["replacements"]] == ["ready", "error"]
    assert first.read_text(encoding="utf-8") == "alpha\nold one\nomega\n"
    assert second.read_text(encoding="utf-8") == "before\nunchanged\nafter\n"


def test_apply_patch_set_duplicate_match_fails_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "repeat.txt"
    target.write_text("same\nsame\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    result = router.dispatch(
        "apply_patch_set",
        {"replacements": [{"path": "repeat.txt", "old_text": "same", "new_text": "done"}]},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "patch_text_not_unique"
    assert result["replacements"][0]["match_count"] == 2
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


def test_apply_patch_set_rejects_workspace_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "apply_patch_set_outside.txt"
    outside.write_text("old", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    result = router.dispatch(
        "apply_patch_set",
        {"replacements": [{"path": "../apply_patch_set_outside.txt", "old_text": "old", "new_text": "new"}]},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_required"
    assert outside.read_text(encoding="utf-8") == "old"


def test_apply_patch_set_batches_external_directory_permission(tmp_path: Path) -> None:
    first_dir = tmp_path.parent / f"{tmp_path.name}-external-a"
    second_dir = tmp_path.parent / f"{tmp_path.name}-external-b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "first.txt"
    second = second_dir / "second.txt"
    first.write_text("old-a", encoding="utf-8")
    second.write_text("old-b", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="once")

    result = router.dispatch(
        "apply_patch_set",
        {
            "replacements": [
                {"path": str(first), "old_text": "old-a", "new_text": "new-a"},
                {"path": str(second), "old_text": "old-b", "new_text": "new-b"},
            ],
        },
        interaction_handler=interaction,
    )

    assert result["status"] == "success"
    external_requests = [request for request in requests if request.tool_name == "external_directory"]
    assert len(external_requests) == 1
    assert set(external_requests[0].args_summary["directories"]) == {
        str(first_dir.resolve()),
        str(second_dir.resolve()),
    }
    assert first.read_text(encoding="utf-8") == "new-a"
    assert second.read_text(encoding="utf-8") == "new-b"


def test_apply_patch_set_policy_denial_happens_before_handler(tmp_path: Path) -> None:
    target = tmp_path / "change.txt"
    target.write_text("old", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["apply_patch_set"], episode_writer=writer, workspace_root=tmp_path)
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        target.write_text("new", encoding="utf-8")
        return {"status": "success"}

    router._handlers["apply_patch_set"] = handler

    result = router.dispatch(
        "apply_patch_set",
        {"replacements": [{"path": "change.txt", "old_text": "old", "new_text": "new"}]},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "policy_denied"
    assert calls == []
    assert target.read_text(encoding="utf-8") == "old"


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
    assert created["changed_files"] == [
        {
            "path": str(tmp_path / "notes.txt"),
            "change_type": "added",
            "additions": 1,
            "deletions": 0,
            "bytes_written": len("hello\n".encode("utf-8")),
        },
    ]
    assert create_existing["status"] == "error"
    assert create_existing["error"]["type"] == "file_exists"
    assert overwritten["status"] == "success"
    assert overwritten["created"] is False
    assert overwritten["changed_files"] == [
        {
            "path": str(tmp_path / "notes.txt"),
            "change_type": "modified",
            "additions": 1,
            "deletions": 1,
            "bytes_written": len("new".encode("utf-8")),
        },
    ]
    assert appended["status"] == "success"
    assert appended["created"] is False
    assert appended["changed_files"] == [
        {
            "path": str(tmp_path / "notes.txt"),
            "change_type": "modified",
            "additions": 1,
            "deletions": 0,
            "bytes_written": len("\nmore".encode("utf-8")),
        },
    ]
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "new\nmore"
    trace_lines = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool_name"] for line in trace_lines] == ["file_write"] * 4


def test_file_write_requests_edit_diff_before_writing_when_handler_is_available(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    requests = []

    def interaction_handler(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True, answer="once")

    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "hello\n", "mode": "create"},
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "success"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello\n"
    assert len(requests) == 1
    request = requests[0]
    assert request.interaction_type == "edit_diff"
    assert request.tool_name == "file_write"
    assert request.args_summary["path"] == "notes.txt"
    assert request.args_summary["change_type"] == "added"
    assert request.args_summary["additions"] == 1
    assert request.args_summary["deletions"] == 0
    assert "+hello" in request.args_summary["diff_preview"]


def test_file_write_edit_diff_denial_does_not_modify_workspace(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    def interaction_handler(request):
        return HumanInteractionResponse(approved=False, answer="deny")

    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "new\n", "mode": "overwrite"},
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert target.read_text(encoding="utf-8") == "old\n"


def test_apply_patch_set_edit_diff_denial_preserves_atomicity(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha\nold one\nomega\n", encoding="utf-8")
    second.write_text("before\nold two\nafter\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    def interaction_handler(request):
        assert request.interaction_type == "edit_diff"
        assert request.args_summary["replacement_count"] == 2
        assert "-old one" in request.args_summary["diff_preview"]
        assert "+new two" in request.args_summary["diff_preview"]
        return HumanInteractionResponse(approved=False, answer="deny")

    result = router.dispatch(
        "apply_patch_set",
        {
            "replacements": [
                {"path": "first.txt", "old_text": "old one", "new_text": "new one"},
                {"path": "second.txt", "old_text": "old two", "new_text": "new two"},
            ],
        },
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert first.read_text(encoding="utf-8") == "alpha\nold one\nomega\n"
    assert second.read_text(encoding="utf-8") == "before\nold two\nafter\n"


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
    assert escaped["error"]["type"] == "approval_required"
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


def test_file_write_blocks_formal_memory_store_path_after_approval(tmp_path: Path) -> None:
    memory_dir = tmp_path / ".haagent" / "memory"
    memory_dir.mkdir(parents=True)
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {
            "path": ".haagent/memory/facts.jsonl",
            "content": '{"title":"用户偏好","body":"以后用中文回答。"}\n',
            "mode": "create",
        },
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "memory_store_path_denied"
    assert "候选确认流程" in result["error"]["message"]
    assert not (memory_dir / "facts.jsonl").exists()
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "file_write"
    assert record["status"] == "error"
    assert record["error"]["type"] == "memory_store_path_denied"


def test_file_write_allows_profile_named_workspace_file_after_approval(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {
            "path": "user_profile.md",
            "content": "# 用户档案\n\n- 名字：小明\n- 爱好：唱跳rap篮球\n",
            "mode": "create",
        },
    )

    assert result["status"] == "success"
    assert (tmp_path / "user_profile.md").read_text(encoding="utf-8").startswith("# 用户档案")


def test_file_write_allows_memory_named_workspace_file_after_approval(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {
            "path": "memory.md",
            "content": "plain text with no profile vocabulary",
            "mode": "create",
        },
    )

    assert result["status"] == "success"
    assert (tmp_path / "memory.md").read_text(encoding="utf-8") == "plain text with no profile vocabulary"


def test_apply_patch_blocks_formal_memory_store_path(tmp_path: Path) -> None:
    target = tmp_path / ".haagent" / "memory" / "facts.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text('{"title":"old"}\n', encoding="utf-8")
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
        {
            "path": ".haagent/memory/facts.jsonl",
            "old_text": '{"title":"old"}\n',
            "new_text": '{"title":"new"}\n',
        },
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "memory_store_path_denied"
    assert target.read_text(encoding="utf-8") == '{"title":"old"}\n'


def test_apply_patch_set_blocks_formal_memory_store_path_without_partial_write(tmp_path: Path) -> None:
    memory_target = tmp_path / ".haagent" / "memory" / "facts.jsonl"
    memory_target.parent.mkdir(parents=True)
    memory_target.write_text('{"title":"old"}\n', encoding="utf-8")
    normal_target = tmp_path / "notes.txt"
    normal_target.write_text("old normal\n", encoding="utf-8")
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["apply_patch_set"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["apply_patch_set"],
        approved_tools=["apply_patch_set"],
    )

    result = router.dispatch(
        "apply_patch_set",
        {
            "replacements": [
                {"path": "notes.txt", "old_text": "old normal\n", "new_text": "new normal\n"},
                {
                    "path": ".haagent/memory/facts.jsonl",
                    "old_text": '{"title":"old"}\n',
                    "new_text": '{"title":"new"}\n',
                },
            ],
        },
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "memory_store_path_denied"
    assert memory_target.read_text(encoding="utf-8") == '{"title":"old"}\n'
    assert normal_target.read_text(encoding="utf-8") == "old normal\n"


def test_shell_does_not_scan_command_for_formal_memory_api_strings(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
        approved_tools=["shell"],
    )
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success"}

    router._handlers["shell"] = handler

    result = router.dispatch(
        "shell",
        {
            "command": (
                "python -c \"from haagent.memory.store import MemoryStore; "
                "MemoryStore(workspace_root='x').confirm_candidate(queue, 'cand_1')\""
            ),
            "timeout_seconds": 5,
        },
    )

    assert result["status"] == "success"
    assert calls == [
        {
            "command": (
                "python -c \"from haagent.memory.store import MemoryStore; "
                "MemoryStore(workspace_root='x').confirm_candidate(queue, 'cand_1')\""
            ),
            "timeout_seconds": 5,
        }
    ]
    record = _read_single_tool_call(writer)
    assert record["tool_name"] == "shell"
    assert record["status"] == "success"


def test_code_run_does_not_scan_code_for_formal_memory_api_strings(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["code_run"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["code_run"],
        approved_tools=["code_run"],
    )
    calls = []

    def handler(args, _context=None):
        calls.append(args)
        return {"status": "success"}

    router._handlers["code_run"] = handler

    result = router.dispatch(
        "code_run",
        {
            "code": (
                "from haagent.memory.candidates import CandidateQueue\n"
                "from haagent.memory.store import MemoryStore\n"
                "store = MemoryStore(workspace_root='x')\n"
                "store.create_candidate(CandidateQueue('s'), scope='user', category='facts')\n"
            ),
            "timeout_seconds": 5,
        },
    )

    assert result["status"] == "success"
    assert calls == [
        {
            "code": (
                "from haagent.memory.candidates import CandidateQueue\n"
                "from haagent.memory.store import MemoryStore\n"
                "store = MemoryStore(workspace_root='x')\n"
                "store.create_candidate(CandidateQueue('s'), scope='user', category='facts')\n"
            ),
            "timeout_seconds": 5,
        }
    ]


def test_file_write_allows_normal_readme_creation(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {"path": "README.md", "content": "# Project\n\n普通项目说明。\n", "mode": "create"},
    )

    assert result["status"] == "success"
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Project\n\n普通项目说明。\n"


def test_file_write_allows_memory_like_content_in_normal_project_file(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {
            "path": "README.md",
            "content": "# Project\n\n用户偏好：以后用中文回答。\n",
            "mode": "create",
        },
    )

    assert result["status"] == "success"
    assert (tmp_path / "README.md").exists()


def test_file_write_secret_content_is_blocked_by_guardrail_not_memory_routing(tmp_path: Path) -> None:
    secret = "sk-test1234567890abcdef1234567890abcdef"
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )

    result = router.dispatch(
        "file_write",
        {"path": "memories.json", "content": f'{{"token":"{secret}"}}', "mode": "create"},
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "guardrail_denied"
    assert not (tmp_path / "memories.json").exists()


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
    success_script = Path(success["script_path"])
    assert success_script.is_absolute()
    assert not success_script.exists()
    assert not (tmp_path / ".haagent-tmp").exists()
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
    absolute_success = router.dispatch(
        "code_run",
        {
            "code": "from pathlib import Path\nprint(Path.cwd().name)",
            "cwd": str(subdir.resolve()),
            "timeout_seconds": 5,
        },
    )
    escaped = router.dispatch("code_run", {"code": "print('x')", "cwd": "..", "timeout_seconds": 5})

    assert success["status"] == "success"
    assert success["stdout_excerpt"].strip() == "pkg"
    assert absolute_success["status"] == "success"
    assert absolute_success["stdout_excerpt"].strip() == "pkg"
    assert escaped["status"] == "error"
    assert escaped["error"]["type"] == "approval_required"


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

    def handler(args, _context=None):
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


def test_router_dispatches_file_write_through_bound_handler(tmp_path: Path) -> None:
    """file_write 必须走 handler map，不能在 Router 内按工具名旁路实现。"""
    from haagent.tools.catalog import ToolExecutionContext

    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    calls: list[tuple[dict, object]] = []
    interaction = object()

    def handler(args, context: ToolExecutionContext):
        calls.append((args, context.interaction_handler))
        return {
            "status": "success",
            "path": str(tmp_path / "notes.txt"),
            "mode": "create",
            "bytes_written": 5,
            "created": True,
            "changed_files": [],
        }

    router._handlers["file_write"] = handler
    result = router.dispatch(
        "file_write",
        {"path": "notes.txt", "content": "hello", "mode": "create"},
        interaction_handler=interaction,  # type: ignore[arg-type]
    )

    assert result["status"] == "success"
    assert len(calls) == 1
    assert calls[0][0]["path"] == "notes.txt"
    assert calls[0][1] is interaction


def test_router_dispatches_request_user_input_through_bound_handler(tmp_path: Path) -> None:
    """request_user_input 必须走 handler map，不能在 dispatch 中单独分支。"""
    from haagent.tools.catalog import ToolExecutionContext

    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["request_user_input"],
        episode_writer=writer,
        workspace_root=tmp_path,
    )
    calls: list[tuple[dict, object]] = []
    interaction = object()

    def handler(args, context: ToolExecutionContext):
        calls.append((args, context.interaction_handler))
        return {
            "status": "success",
            "question": args["question"],
            "answer": "bound",
            "answer_chars": 5,
        }

    router._handlers["request_user_input"] = handler
    result = router.dispatch(
        "request_user_input",
        {"question": "Which file?"},
        interaction_handler=interaction,  # type: ignore[arg-type]
    )

    assert result["answer"] == "bound"
    assert len(calls) == 1
    assert calls[0][1] is interaction


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
    assert len(requests) == 2
    assert requests[0].interaction_type == "approval"
    assert requests[0].tool_name == "file_write"
    assert {
        key: requests[0].args_summary[key]
        for key in ("content_chars", "mode", "path")
    } == {"content_chars": 8, "mode": "create", "path": "notes.txt"}
    assert requests[0].args_summary["permission_patterns"]
    assert requests[0].args_summary["permission_always"]
    assert requests[1].interaction_type == "edit_diff"
    assert requests[1].tool_name == "file_write"
    assert requests[1].args_summary["path"] == "notes.txt"
    assert requests[1].args_summary["additions"] == 1
    assert requests[1].args_summary["deletions"] == 0
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

    def handler(args, _context=None):
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
    assert result["stdout_excerpt"].strip() == str(tmp_path.resolve())


def test_shell_uses_workspace_root_when_cwd_is_dot(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "cwd": ".", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(tmp_path.resolve())


def test_shell_runs_in_workspace_relative_subdirectory(tmp_path: Path) -> None:
    subdir = tmp_path / "src"
    subdir.mkdir()

    result = shell(
        {"command": _print_cwd_command(), "cwd": "src", "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(subdir.resolve())


def test_shell_accepts_absolute_cwd_inside_workspace_root(tmp_path: Path) -> None:
    subdir = tmp_path / "src"
    subdir.mkdir()

    result = shell(
        {"command": _print_cwd_command(), "cwd": str(subdir.resolve()), "timeout_seconds": 5},
        tmp_path,
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(subdir.resolve())


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
    assert result["error"]["type"] == "approval_required"
    assert "用户确认" in result["error"]["message"]


def test_shell_rejects_non_positive_timeout_with_argument_error(tmp_path: Path) -> None:
    result = shell(
        {"command": _print_cwd_command(), "timeout_seconds": 0},
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "category": "argument",
        "message": "timeout_seconds must be positive",
        "retryable": False,
    }
    assert result["recovery"]["action"] == "correct_arguments"


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


def test_auto_approve_path_policy_skips_edit_diff_for_multiple_writes(tmp_path: Path) -> None:
    """auto_approve：已授权路径上多次 file_write 不触发 edit_diff handler。"""
    from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver

    writer = make_writer(tmp_path)
    policy = PathPolicy(project_root=tmp_path, permission_mode="auto_approve")
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        path_policy=policy,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    handler_calls: list[object] = []

    def interaction_handler(request):
        handler_calls.append(request)
        return HumanInteractionResponse(approved=False, answer="deny")

    # 模拟 orchestrator bridge：mode 自动跳过时不调用用户 handler
    resolver = HumanInteractionResolver(permission_mode="auto_approve")

    def bridge(request):
        if resolution := resolver.resolve(request):
            return resolution.to_response()
        handler_calls.append(request)
        return interaction_handler(request)

    r1 = router.dispatch(
        "file_write",
        {"path": "a.txt", "content": "one\n", "mode": "create"},
        interaction_handler=bridge,
    )
    r2 = router.dispatch(
        "file_write",
        {"path": "b.txt", "content": "two\n", "mode": "create"},
        interaction_handler=bridge,
    )

    assert r1["status"] == "success"
    assert r2["status"] == "success"
    assert handler_calls == []
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "two\n"


def test_auto_approve_still_denies_unauthorized_external_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    writer = make_writer(project)
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=project,
        path_policy=PathPolicy(project_root=project, permission_mode="auto_approve"),
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    handler_calls: list[object] = []

    def interaction_handler(request):
        handler_calls.append(request)
        return HumanInteractionResponse(approved=True, answer="once")

    result = router.dispatch(
        "file_write",
        {"path": str(external / "x.txt"), "content": "no\n", "mode": "create"},
        interaction_handler=interaction_handler,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "path_policy_denied"
    assert not (external / "x.txt").exists()
    assert handler_calls == []


def test_full_access_skips_edit_diff_and_allows_external_write(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    writer = make_writer(project)
    policy = PathPolicy(project_root=project, permission_mode="full_access")
    router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=project,
        path_policy=policy,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver

    resolver = HumanInteractionResolver(permission_mode="full_access")
    handler_calls: list[object] = []

    def bridge(request):
        if resolution := resolver.resolve(request):
            return resolution.to_response()
        handler_calls.append(request)
        return HumanInteractionResponse(approved=False, answer="deny")

    result = router.dispatch(
        "file_write",
        {"path": str(external / "out.txt"), "content": "ok\n", "mode": "create"},
        interaction_handler=bridge,
    )

    assert result["status"] == "success"
    assert (external / "out.txt").read_text(encoding="utf-8") == "ok\n"
    assert handler_calls == []


def test_request_approval_always_skips_later_edit_diffs_not_shell(tmp_path: Path) -> None:
    """always 只免除后续 edit_diff；shell 仍需 approval。"""
    from haagent.runtime.execution.human_interaction_resolver import HumanInteractionResolver

    writer = make_writer(tmp_path)
    resolver = HumanInteractionResolver(permission_mode="request_approval")
    edit_prompts: list[str] = []
    approval_prompts: list[str] = []

    def bridge(request):
        if resolution := resolver.resolve(request):
            return resolution.to_response()
        if request.interaction_type == "edit_diff":
            edit_prompts.append(request.args_summary.get("path", ""))
            response = HumanInteractionResponse(approved=True, answer="always" if len(edit_prompts) == 1 else "once")
        else:
            approval_prompts.append(request.tool_name)
            response = HumanInteractionResponse(approved=False, answer="no")
        resolver.record(request, response, turn=len(edit_prompts) + len(approval_prompts))
        return response

    file_router = ToolRouter(
        allowed_tools=["file_write"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["file_write"],
        approved_tools=["file_write"],
    )
    assert (
        file_router.dispatch(
            "file_write",
            {"path": "a.txt", "content": "a\n", "mode": "create"},
            interaction_handler=bridge,
        )["status"]
        == "success"
    )
    assert (
        file_router.dispatch(
            "file_write",
            {"path": "b.txt", "content": "b\n", "mode": "create"},
            interaction_handler=bridge,
        )["status"]
        == "success"
    )
    assert edit_prompts == ["a.txt"]

    shell_router = ToolRouter(
        allowed_tools=["shell"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell"],
    )
    shell_result = shell_router.dispatch(
        "shell",
        {"command": "echo hi", "cwd": "."},
        interaction_handler=bridge,
    )
    assert shell_result["status"] == "error"
    assert shell_result["error"]["type"] in {"policy_denied", "approval_denied"}
    assert approval_prompts == ["shell"]


def _read_single_tool_call(writer: EpisodeWriter) -> dict[str, object]:
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    return json.loads(trace)


def _print_cwd_command() -> str:
    return "python -c \"from pathlib import Path; print(Path.cwd().resolve())\""
