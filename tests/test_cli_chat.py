"""
tests/test_cli_chat.py - chat 入口与 AgentSession 测试

验证自然语言单次模式、REPL 命令和有预算的会话摘要。
"""

import json
from pathlib import Path

from haagent import cli
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.chat_session import AgentSession
from haagent.runtime.human_interaction import HumanInteractionResponse
from haagent.runtime.task_contract import load_task


class RecordingGateway:
    provider_name = "recording"

    def __init__(self) -> None:
        self.model_inputs: list[str] = []

    def generate(self, task, model_input, tool_schemas, observations):
        self.model_inputs.append(model_input)
        if task.goal == "first" and not observations:
            return ModelResponse("listing", [ToolCall("file_list", {})])
        return ModelResponse(f"done: {task.goal}", [])


class WriteThenDoneGateway:
    provider_name = "write-then-done"

    def generate(self, task, model_input, tool_schemas, observations):
        if not observations:
            return ModelResponse(
                "writing",
                [
                    ToolCall(
                        "file_write",
                        {
                            "path": "notes.txt",
                            "content": "SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT",
                            "mode": "create",
                        },
                    ),
                ],
            )
        return ModelResponse("done writing", [])


class BadToolGateway:
    provider_name = "bad-tool"

    def generate(self, task, model_input, tool_schemas, observations):
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class CodeRunThenDoneGateway:
    provider_name = "code-run-then-done"

    def generate(self, task, model_input, tool_schemas, observations):
        if not observations:
            return ModelResponse(
                "running code",
                [
                    ToolCall(
                        "code_run",
                        {
                            "code": "print('SECRET_STDOUT_SHOULD_NOT_PRINT' * 200)",
                            "timeout_seconds": 5,
                        },
                    ),
                ],
            )
        return ModelResponse("done code", [])


class ClarifyThenDoneGateway:
    provider_name = "clarify-then-done"

    def __init__(self) -> None:
        self.model_inputs: list[str] = []

    def generate(self, task, model_input, tool_schemas, observations):
        self.model_inputs.append(model_input)
        if not observations:
            return ModelResponse(
                "need target",
                [
                    ToolCall(
                        "request_user_input",
                        {
                            "question": "Which file should I inspect?",
                            "reason": "Need target file",
                        },
                    ),
                ],
            )
        return ModelResponse("done with answer", [])


class FakeProfileGateway:
    provider_name = "openai-chat"

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

    def generate(self, task, model_input, tool_schemas, observations):
        return ModelResponse("profile done", [])


def test_cli_chat_parser_accepts_optional_request() -> None:
    parser = cli.build_parser()

    repl_args = parser.parse_args(["chat"])
    single_args = parser.parse_args(["chat", "List this project"])

    assert repl_args.command == "chat"
    assert repl_args.request is None
    assert repl_args.workspace_root is None
    assert repl_args.provider == "fake"
    assert single_args.request == "List this project"


def test_cli_chat_without_prompt_enters_repl_and_quits(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    prompts = []
    inputs = iter([":quit"])

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(inputs)

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert prompts == ["haagent> "]
    assert "session_id=" in output
    assert "bye" in output


def test_cli_chat_repl_runs_one_prompt_then_quits(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = iter(["List project", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    task = load_task(episode_path / "task.yaml")
    assert exit_code == 0
    assert "status=completed" in output
    assert "provider=fake" in output
    assert "verification=not_run" in output
    assert task.goal == "List project"
    assert task.workspace_root == str(tmp_path.resolve())


def test_cli_chat_default_tools_include_real_task_tool_pack(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(["chat", "Check tools", "--workspace-root", str(tmp_path), "--provider", "fake"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    task = load_task(episode_path / "task.yaml")
    assert exit_code == 0
    assert "file_write" in task.allowed_tools
    assert "code_run" in task.allowed_tools
    assert "request_user_input" in task.allowed_tools
    assert "file_write" in task.policy["approval_allowed_tools"]
    assert "code_run" in task.policy["approval_allowed_tools"]
    assert task.policy["approved_tools"] == []


def test_cli_chat_single_prompt_accepts_explicit_workspace_root(
    tmp_path: Path,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    exit_code = cli.main(["chat", "Work here", "--workspace-root", str(workspace), "--provider", "fake"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    task = load_task(episode_path / "task.yaml")
    assert exit_code == 0
    assert task.workspace_root == str(workspace.resolve())


def test_cli_chat_profile_does_not_leak_secret(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    config_dir = tmp_path / ".haagent"
    config_dir.mkdir()
    (config_dir / "providers.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "local",
                        "provider": "openai-chat",
                        "base_url": "https://api.example/v1",
                        "model": "chat-test",
                        "api_key_env": "CHAT_SECRET",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHAT_SECRET", "sk-secret-value-that-must-not-leak")
    monkeypatch.setattr(cli, "OpenAIChatCompletionsGateway", FakeProfileGateway)

    exit_code = cli.main(["chat", "Use profile", "--profile", "local"])

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    episode_text = (episode_path / "episode.json").read_text(encoding="utf-8")
    task_text = (episode_path / "task.yaml").read_text(encoding="utf-8")
    assert exit_code == 0
    assert "provider=openai-chat" in output
    assert "sk-secret-value-that-must-not-leak" not in output
    assert "sk-secret-value-that-must-not-leak" not in episode_text
    assert "sk-secret-value-that-must-not-leak" not in task_text
    assert "CHAT_SECRET" not in output


def test_cli_chat_repl_status_reports_session_state(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = iter([":status", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"workspace_root={tmp_path.resolve()}" in output
    assert "provider=fake" in output
    assert "turn_count=0" in output
    assert "session_id=" in output


def test_cli_chat_repl_new_resets_turn_count_and_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = iter(["First task", ":status", ":new", ":status", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "turn_count=1" in output
    assert "session reset" in output
    assert "turn_count=0" in output


def test_cli_chat_repl_empty_input_does_not_run_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    inputs = iter(["", "   ", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    assert exit_code == 0
    assert not (tmp_path / ".runs").exists()


def test_cli_chat_single_prompt_still_runs_once(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(["chat", "Single request", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status=completed" in output
    assert "verification=not_run" in output
    assert "session_id=" not in output


def test_agent_session_events_show_single_turn_order_and_tool_success(tmp_path: Path) -> None:
    events = []
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=WriteThenDoneGateway(),
        max_turns=20,
    )

    result = session.run_prompt_events(
        "write notes",
        event_sink=events.append,
        include_session_events=True,
        interaction_handler=lambda request: HumanInteractionResponse(approved=True, answer="yes"),
    )

    assert result.status == "completed"
    assert [event.event_type for event in events] == [
        "session_started",
        "turn_started",
        "tool_started",
        "approval_requested",
        "approval_granted",
        "tool_finished",
        "assistant_message",
        "turn_finished",
        "session_finished",
    ]
    assert events[2].payload["tool_name"] == "file_write"
    assert events[2].payload["args_summary"]["content_chars"] == len("SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT")
    assert "content" not in events[2].payload["args_summary"]
    assert events[5].payload["result_summary"]["bytes_written"] == len("SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT")
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT"
    assert (result.episode_path / "tool-calls.jsonl").exists()


def test_agent_session_user_input_request_continues_with_answer(tmp_path: Path) -> None:
    events = []
    gateway = ClarifyThenDoneGateway()
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=20,
    )

    result = session.run_prompt_events(
        "inspect target",
        event_sink=events.append,
        interaction_handler=lambda request: HumanInteractionResponse(
            approved=True,
            answer="Use README.md",
        ),
    )

    assert result.status == "completed"
    assert [event.event_type for event in events] == [
        "turn_started",
        "tool_started",
        "user_input_requested",
        "user_input_received",
        "tool_finished",
        "assistant_message",
        "turn_finished",
    ]
    assert events[2].payload["question"] == "Which file should I inspect?"
    assert events[3].payload["answer_chars"] == len("Use README.md")
    assert "Use README.md" in gateway.model_inputs[1]
    assert "tool-calls.jsonl" not in gateway.model_inputs[1]


def test_agent_session_denied_approval_fails_without_running_tool(tmp_path: Path) -> None:
    events = []
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=WriteThenDoneGateway(),
        max_turns=20,
    )

    result = session.run_prompt_events(
        "write notes",
        event_sink=events.append,
        interaction_handler=lambda request: HumanInteractionResponse(approved=False, answer="no"),
    )

    assert result.status == "failed"
    assert not (tmp_path / "notes.txt").exists()
    assert [event.event_type for event in events] == [
        "turn_started",
        "tool_started",
        "approval_requested",
        "approval_denied",
        "tool_failed",
        "turn_finished",
    ]
    assert result.failure_category == "User Denied Failure"
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event") == "approval_denied" for record in transcript)


def test_agent_session_events_emit_tool_failed_on_real_tool_error(tmp_path: Path) -> None:
    events = []
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=BadToolGateway(),
        max_turns=20,
    )

    result = session.run_prompt_events("fail", event_sink=events.append)

    assert result.status == "failed"
    assert [event.event_type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_failed",
        "turn_finished",
    ]
    assert events[2].payload["tool_name"] == "file_read"
    assert events[2].payload["error_type"] == "tool_argument_invalid"
    assert "missing required argument: path" in events[2].payload["message"]


def test_cli_chat_single_prompt_prints_progress_events_without_secret_content(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_build_run_model_gateway", lambda args: WriteThenDoneGateway())
    inputs = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "Write notes", "--workspace-root", str(tmp_path), "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "event=session_started" in output
    assert "event=turn_started" in output
    assert "event=tool_started tool=file_write" in output
    assert "event=approval_requested tool=file_write" in output
    assert "event=approval_granted tool=file_write" in output
    assert "content_chars=37" in output
    assert "event=tool_finished tool=file_write status=success" in output
    assert "event=assistant_message" in output
    assert "event=session_finished" in output
    assert "SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT" not in output
    assert "status=completed" in output


def test_cli_chat_repl_prints_events_and_still_accepts_quit(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_build_run_model_gateway", lambda args: WriteThenDoneGateway())
    inputs = iter(["Write notes", "y", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "event=session_started" in output
    assert "event=tool_started tool=file_write" in output
    assert "event=approval_requested tool=file_write" in output
    assert "event=approval_granted tool=file_write" in output
    assert "event=session_finished" in output
    assert "bye" in output
    assert "SECRET_WRITE_CONTENT_SHOULD_NOT_PRINT" not in output


def test_cli_chat_event_output_hides_full_code_and_long_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_build_run_model_gateway", lambda args: CodeRunThenDoneGateway())
    inputs = iter(["y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "Run code", "--workspace-root", str(tmp_path), "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "event=tool_started tool=code_run" in output
    assert "event=approval_requested tool=code_run" in output
    assert "code_chars=" in output
    assert "event=tool_finished tool=code_run status=success" in output
    assert "SECRET_STDOUT_SHOULD_NOT_PRINT" not in output
    assert "print(" not in output


def test_cli_chat_repl_answers_user_input_request(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_build_run_model_gateway", lambda args: ClarifyThenDoneGateway())
    inputs = iter(["Inspect", "README.md", ":quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    exit_code = cli.main(["chat", "--provider", "fake"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "event=user_input_requested" in output
    assert "question=\"Which file should I inspect?\"" in output
    assert "event=user_input_received" in output
    assert "answer_chars=9" in output
    assert "status=completed" in output


def test_user_input_answer_is_bounded_in_next_model_input(tmp_path: Path) -> None:
    long_answer = "A" * 1200
    gateway = ClarifyThenDoneGateway()
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=20,
    )

    result = session.run_prompt_events(
        "inspect target",
        interaction_handler=lambda request: HumanInteractionResponse(
            approved=True,
            answer=long_answer,
        ),
    )

    assert result.status == "completed"
    assert "A" * 240 in gateway.model_inputs[1]
    assert "A" * 400 not in gateway.model_inputs[1]
    assert "tool-calls.jsonl" not in gateway.model_inputs[1]


def test_agent_session_summary_is_bounded_and_not_episode_trace(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    gateway = RecordingGateway()
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=20,
    )

    first = session.run_prompt("first")
    gateway.model_inputs.clear()
    second = session.run_prompt("second")

    second_model_input = gateway.model_inputs[0]
    context_manifest = json.loads(
        (second.episode_path / "contexts" / "0001.json").read_text(encoding="utf-8"),
    )
    session_sources = [
        source
        for source in context_manifest["sources"]
        if source["source_type"] == "session_summary"
    ]
    assert first.status == "completed"
    assert second.status == "completed"
    assert "Session Summary:" in second_model_input
    assert "first" in second_model_input
    assert str(first.episode_path) in second_model_input
    assert "tool-calls.jsonl" not in second_model_input
    assert '"tool_name"' not in second_model_input
    assert '"event": "model_call"' not in second_model_input
    assert len(session_sources) == 1
    assert session_sources[0]["budget"]["model_input_char_count"] <= 1000


def test_agent_session_new_clears_summary(tmp_path: Path) -> None:
    gateway = RecordingGateway()
    session = AgentSession(
        workspace_root=tmp_path,
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=20,
    )
    session.run_prompt("first")
    session.new()
    gateway.model_inputs.clear()

    session.run_prompt("second")

    assert session.turn_count == 1
    assert "first" not in gateway.model_inputs[0]
