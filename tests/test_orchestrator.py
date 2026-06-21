"""
tests/test_orchestrator.py - RunOrchestrator 状态流转测试

验证成功路径、工具失败和模型失败会写入正确 run 状态。
"""

import json
from pathlib import Path

from haagent.models.gateway import ModelCallError
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.models.gateway import OpenAIChatCompletionsGateway
from haagent.models.gateway import OpenAIResponsesGateway
from haagent.runtime.orchestrator import RunOrchestrator
from haagent.runtime.state import RunStatus
from haagent.verification.engine import VerificationResult


class FailingGateway:
    provider_name = "failing"

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        raise ModelCallError("model exploded")


class TypeErrorGateway:
    provider_name = "type-error"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        self.call_count += 1
        raise TypeError("internal provider type error mentioning model_input")


class SequenceGateway:
    provider_name = "sequence"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self.observations_seen = []
        self.model_inputs_seen = []
        self.tool_schemas_seen = []

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        self.observations_seen.append(list(observations or []))
        self.model_inputs_seen.append(model_input)
        self.tool_schemas_seen.append(list(tool_schemas or []))
        return self._responses.pop(0)


def write_task(
    path: Path,
    allowed_tools: list[str],
    verification_commands: list[str] | None = None,
    workspace_root: str | None = None,
    policy_block: str = "",
) -> None:
    allowed_tools_yaml = "\n".join(f"  - {tool}" for tool in allowed_tools)
    verification_commands = verification_commands or []
    verification_yaml = "\n".join(f"  - {command}" for command in verification_commands)
    verification_block = f"\n{verification_yaml}" if verification_yaml else " []"
    workspace_root_line = f"workspace_root: {workspace_root}\n" if workspace_root is not None else ""
    path.write_text(
        f"""
goal: Exercise orchestrator
{workspace_root_line}constraints: []
allowed_tools:
{allowed_tools_yaml}
acceptance_criteria:
  - Run reaches terminal state
verification_commands:{verification_block}
{policy_block}
""".strip(),
        encoding="utf-8",
    )


def test_orchestrator_records_successful_state_flow(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.EXECUTING,
        RunStatus.VERIFYING,
        RunStatus.COMPLETED,
    ]
    environment = json.loads((result.episode_path / "environment.json").read_text(encoding="utf-8"))
    assert environment["workspace_root"] == str(tmp_path.resolve())
    sandbox = json.loads((result.episode_path / "sandbox.json").read_text(encoding="utf-8"))
    assert sandbox["workspace_root"] == str(tmp_path.resolve())
    assert sandbox["filesystem_boundary"] == "workspace_root"
    assert sandbox["network_policy"] == "unrestricted"
    assert sandbox["process_policy"] == "local_subprocess"
    assert sandbox["credential_policy"] == "inherit_environment"
    assert isinstance(sandbox["resource_limits"]["command_timeout_seconds"], int | float)
    episode = json.loads((result.episode_path / "episode.json").read_text(encoding="utf-8"))
    assert episode["episode_version"] == "1.0"
    assert episode["status"] == "completed"
    assert episode["provider"] == "fake"
    assert episode["task_path"] == str(task_path)
    assert episode["workspace_root"] == str(tmp_path.resolve())
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))
    assert failure == {"status": "success", "failure": None}
    plan = json.loads((result.episode_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["goal"] == "Exercise orchestrator"
    assert plan["allowed_tools"] == ["fake_tool"]
    assert plan["acceptance_criteria"] == ["Run reaches terminal state"]
    assert plan["verification_commands"] == []
    assert plan["planned_steps"] == [
        "Clarify the task goal and constraints from task.yaml.",
        "Use allowed tools: fake_tool.",
        "Check acceptance criteria: Run reaches terminal state.",
        "Run verification commands if provided.",
    ]
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {
        "event": "planning",
        "plan_path": "plan.json",
        "planned_step_count": 4,
    } in transcript


def test_orchestrator_executes_openai_provider_tool_call_smoke(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    payloads: list[dict[str, object]] = []

    def fake_transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "output_text": "",
                "output": [
                    {
                        "type": "function_call",
                        "name": "fake_tool",
                        "arguments": "{}",
                    },
                ],
            }
        return {"output_text": "provider finished after observing fake_tool"}

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=fake_transport,
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    episode = json.loads((result.episode_path / "episode.json").read_text(encoding="utf-8"))
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    context_manifest = json.loads(
        (result.episode_path / "context-manifest.json").read_text(encoding="utf-8"),
    )
    tool_calls = [
        json.loads(line)
        for line in (
            result.episode_path / "tool-calls.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    model_calls = [record for record in transcript if record["event"] == "model_call"]
    model_responses = [record for record in transcript if record["event"] == "model_response"]
    tool_observations = [record for record in transcript if record["event"] == "tool_observation"]

    assert result.status is RunStatus.COMPLETED
    assert episode["provider"] == "openai"
    assert len(model_calls) >= 2
    assert model_responses[0]["tool_calls"] == [{"name": "fake_tool", "args": {}}]
    assert tool_observations[0]["tool_name"] == "fake_tool"
    assert tool_observations[0]["result"]["status"] == "success"
    assert context_manifest["context_count"] == 2
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "fake_tool"
    assert tool_calls[0]["args"] == {}
    assert tool_calls[0]["status"] == "success"
    assert tool_calls[0]["result"]["status"] == "success"
    assert tool_calls[0]["policy"]["action"] == "allow"
    assert len(payloads) == 2
    assert all("tools" in payload for payload in payloads)
    first_tools = payloads[0]["tools"]
    assert isinstance(first_tools, list)
    assert first_tools[0]["name"] == "fake_tool"


def test_orchestrator_executes_openai_chat_provider_file_read_tool_call_smoke(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    note_text = "workspace note says hello from file_read\n"
    (tmp_path / "notes.txt").write_text(note_text, encoding="utf-8")
    write_task(task_path, ["file_read"])
    payloads: list[dict[str, object]] = []

    def fake_transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "file_read",
                                        "arguments": "{\"path\": \"notes.txt\"}",
                                    },
                                },
                            ],
                        },
                    },
                ],
            }
        return {"choices": [{"message": {"content": "final answer from file note"}}]}

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        model="chat-test",
        transport=fake_transport,
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    episode = json.loads((result.episode_path / "episode.json").read_text(encoding="utf-8"))
    transcript = _read_transcript(result.episode_path)
    context_manifest = json.loads(
        (result.episode_path / "context-manifest.json").read_text(encoding="utf-8"),
    )
    tool_calls = [
        json.loads(line)
        for line in (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tool_observations = [record for record in transcript if record["event"] == "tool_observation"]
    second_message = payloads[1]["messages"][0]

    assert result.status is RunStatus.COMPLETED
    assert episode["provider"] == "openai-chat"
    assert context_manifest["context_count"] == 2
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "file_read"
    assert tool_calls[0]["args"] == {"path": "notes.txt"}
    assert tool_calls[0]["status"] == "success"
    assert tool_calls[0]["result"]["content"] == note_text
    assert tool_observations[0]["tool_name"] == "file_read"
    assert tool_observations[0]["result"]["content"] == note_text
    assert len(payloads) == 2
    assert isinstance(second_message, dict)
    assert note_text.strip() in str(second_message["content"])
    assert '"path": "notes.txt"' in str(second_message["content"])


def test_orchestrator_fails_when_fake_tool_is_not_allowed(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["other_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history[-1] is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Task Spec Failure" in failure_text
    assert "other_tool" in failure_text
    assert (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8") == ""


def test_orchestrator_workspace_root_can_point_to_project_root_and_read_agents_md(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    tasks_dir = project_root / "tasks"
    tasks_dir.mkdir(parents=True)
    (project_root / "AGENTS.md").write_text("Project root instruction.", encoding="utf-8")
    task_path = tasks_dir / "task.yaml"
    write_task(task_path, ["fake_tool"], workspace_root="..")
    gateway = SequenceGateway([ModelResponse("done", [])])

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    first_context = (result.episode_path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    context_manifest = json.loads((result.episode_path / "contexts" / "0001.json").read_text(encoding="utf-8"))
    assert "Project root instruction." in first_context
    assert context_manifest["workspace_root"] == str(project_root.resolve())


def test_orchestrator_file_tool_uses_resolved_workspace_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    tasks_dir = project_root / "tasks"
    tasks_dir.mkdir(parents=True)
    (project_root / "root-note.txt").write_text("from project root\n", encoding="utf-8")
    task_path = tasks_dir / "task.yaml"
    write_task(task_path, ["file_read"], workspace_root="..")
    gateway = SequenceGateway(
        [
            ModelResponse("read root file", [ToolCall("file_read", {"path": "root-note.txt"})]),
            ModelResponse("done", []),
        ],
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    tool_records = [
        json.loads(line)
        for line in (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert tool_records[0]["result"]["content"] == "from project root\n"


def test_orchestrator_invalid_tool_args_are_tool_argument_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["file_read"])
    gateway = SequenceGateway(
        [ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Tool Argument Failure" in failure_text
    assert "missing required argument: path" in failure_text
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure"]["category"] == "Tool Argument Failure"
    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert tool_call["error"]["type"] == "tool_argument_invalid"


def test_orchestrator_policy_denied_tool_is_tool_interface_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["shell"])
    gateway = SequenceGateway(
        [ModelResponse("try shell", [ToolCall("shell", {"command": "echo blocked"})])],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Tool Interface Failure" in failure_text
    assert "policy denies high risk tool shell" in failure_text
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure"]["category"] == "Tool Interface Failure"
    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert tool_call["status"] == "error"
    assert tool_call["policy"]["action"] == "deny"
    assert tool_call["error"]["type"] == "policy_denied"


def test_orchestrator_passes_policy_approval_allowed_tools_to_router(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(
        task_path,
        ["shell"],
        policy_block="""
policy:
  approval_allowed_tools:
    - shell
""".strip(),
    )
    gateway = SequenceGateway(
        [ModelResponse("try shell", [ToolCall("shell", {"command": "echo blocked"})])],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert tool_call["policy"]["action"] == "deny"
    assert tool_call["policy"]["approval"] == {
        "required": True,
        "status": "missing",
        "reason": "approval allowed but missing for high risk tool shell",
    }


def test_orchestrator_passes_policy_approved_tools_to_router(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(
        task_path,
        ["shell"],
        policy_block="""
policy:
  approval_allowed_tools:
    - shell
  approved_tools:
    - shell
""".strip(),
    )
    gateway = SequenceGateway(
        [
            ModelResponse("try approved shell", [ToolCall("shell", {"command": "echo approved"})]),
            ModelResponse("done", []),
        ],
    )

    def approved_shell(args, workspace_root):
        return {"status": "success", "stdout": "approved\n", "stderr": "", "args": args}

    monkeypatch.setattr("haagent.tools.router.shell", approved_shell)

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.COMPLETED
    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8"))
    assert tool_call["status"] == "success"
    assert tool_call["policy"]["action"] == "allow"
    assert tool_call["policy"]["approval"] == {
        "required": True,
        "status": "granted",
        "reason": "approval granted for high risk tool shell",
    }


def test_orchestrator_unknown_runtime_tool_is_tool_interface_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = SequenceGateway(
        [ModelResponse("unknown runtime tool", [ToolCall("mystery_tool", {})])],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Tool Interface Failure" in failure_text
    assert "tool is not allowed: mystery_tool" in failure_text


def test_orchestrator_verification_uses_resolved_workspace_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    tasks_dir = project_root / "tasks"
    tasks_dir.mkdir(parents=True)
    task_path = tasks_dir / "task.yaml"
    write_task(
        task_path,
        ["fake_tool"],
        verification_commands=[
            "python -c \"from pathlib import Path; Path('verified-root.txt').write_text('ok', encoding='utf-8')\"",
        ],
        workspace_root="..",
    )
    gateway = SequenceGateway([ModelResponse("done", [])])

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert (project_root / "verified-root.txt").read_text(encoding="utf-8") == "ok"


def test_orchestrator_redacts_verification_output_in_failure_attribution(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    raw_key = "OPENAI_API_KEY=super-secret-value"
    write_task(
        task_path,
        ["fake_tool"],
        verification_commands=[
            f"python -c \"import sys; print('{raw_key}'); sys.exit(5)\"",
        ],
    )
    gateway = SequenceGateway([ModelResponse("done", [])])

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert result.status is RunStatus.FAILED
    assert raw_key not in failure_text
    assert "OPENAI_API_KEY=[REDACTED]" in failure_text


def test_orchestrator_fails_when_workspace_root_does_not_exist(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path, ["fake_tool"], workspace_root="missing-workspace")

    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Task Spec Failure" in failure_text
    assert "workspace_root does not exist" in failure_text
    episode = json.loads((result.episode_path / "episode.json").read_text(encoding="utf-8"))
    assert episode["status"] == "failed"
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure"]["category"] == "Task Spec Failure"
    assert failure["failure"]["stage"] == "created"
    assert "workspace_root does not exist" in failure["failure"]["evidence"]


def test_orchestrator_fails_when_model_gateway_fails(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=FailingGateway()).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Model Failure" in failure_text
    assert "model exploded" in failure_text
    assert (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8") == ""


def test_orchestrator_does_not_retry_internal_gateway_type_error(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = TypeErrorGateway()

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.FAILED
    assert gateway.call_count == 1
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Model Call Failure" in failure_text
    assert "internal provider type error mentioning model_input" in failure_text
    assert (result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8") == ""


def test_orchestrator_fails_when_verification_command_fails(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(
        task_path,
        ["fake_tool"],
        verification_commands=[
            "python -c \"import sys; print('verify-out'); print('verify-err', file=sys.stderr); sys.exit(5)\"",
        ],
    )

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.EXECUTING,
        RunStatus.VERIFYING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Verification Failure" in failure_text
    assert "exit_code=5" in failure_text
    assert "stdout: verify-out" in failure_text
    assert "stderr: verify-err" in failure_text
    commands_log = result.episode_path / "verification" / "commands.jsonl"
    assert json.loads(commands_log.read_text(encoding="utf-8"))["exit_code"] == 5
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["failure"]["category"] == "Verification Failure"
    assert "stdout: verify-out" in failure["failure"]["evidence"]
    assert "stderr: verify-err" in failure["failure"]["evidence"]


def test_orchestrator_fails_unknown_tool_as_task_spec_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["mystery_tool"])

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.FAILED,
    ]
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Task Spec Failure" in failure_text
    assert "mystery_tool" in failure_text


def test_orchestrator_attributes_agents_md_read_failure_as_context_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    agents_path = tmp_path / "AGENTS.md"
    write_task(task_path, ["fake_tool"])
    agents_path.write_text("blocked", encoding="utf-8")
    original_read_text = Path.read_text

    def read_text_with_failure(path, *args, **kwargs):
        if Path(path) == agents_path:
            raise OSError("cannot read AGENTS.md")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text_with_failure)

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Context Failure" in failure_text
    assert "cannot read AGENTS.md" in failure_text


def test_orchestrator_attributes_context_budget_failure_as_context_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path, ["fake_tool"])
    (tmp_path / "AGENTS.md").write_text("x" * 13000, encoding="utf-8")

    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Context Failure" in failure_text
    assert "context character budget exceeded" in failure_text


def test_orchestrator_failure_attribution_includes_verification_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"], verification_commands=["slow command"])

    class TimeoutVerificationEngine:
        def __init__(self, episode_writer, workspace_root):
            pass

        def run(self, commands):
            return VerificationResult(
                status="failed",
                failed_command=commands[0],
                exit_code=None,
                failure_reason="timeout",
            )

    monkeypatch.setattr(
        "haagent.runtime.orchestrator.VerificationEngine",
        TimeoutVerificationEngine,
    )

    result = RunOrchestrator(runs_root=runs_dir).run(task_path)

    assert result.status is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Verification Failure" in failure_text
    assert "slow command" in failure_text
    assert "timeout" in failure_text


def test_orchestrator_completes_after_two_tool_rounds(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = SequenceGateway(
        [
            ModelResponse("round 1", [ToolCall("fake_tool", {"round": 1})]),
            ModelResponse("round 2", [ToolCall("fake_tool", {"round": 2})]),
            ModelResponse("done", []),
        ],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert len(gateway.observations_seen) == 3
    assert gateway.observations_seen[0] == []
    assert gateway.observations_seen[1][0]["tool_name"] == "fake_tool"
    transcript = _read_transcript(result.episode_path)
    assert [record["context_id"] for record in transcript if record.get("event") == "model_call"] == [
        "0001",
        "0002",
        "0003",
    ]
    assert len([record for record in transcript if record.get("event") == "tool_observation"]) == 2
    first_context = (result.episode_path / "contexts" / "0001.txt").read_text(encoding="utf-8")
    second_context = (result.episode_path / "contexts" / "0002.txt").read_text(encoding="utf-8")
    second_manifest = json.loads(
        (result.episode_path / "contexts" / "0002.json").read_text(encoding="utf-8"),
    )
    run_manifest = json.loads((result.episode_path / "context-manifest.json").read_text(encoding="utf-8"))
    assert "Observations:" in first_context
    assert "- none" in first_context
    assert "Plan:" in first_context
    assert "- Use allowed tools: fake_tool." in first_context
    assert first_context.index("Plan:") < first_context.index("Observations:")
    assert first_context.index("Observations:") < first_context.index("Pending next step:")
    assert "Pending next step:" in first_context
    assert "Pending next step:\n- none" in first_context
    assert "Plan:" in second_context
    assert "- Use allowed tools: fake_tool." in second_context
    assert "fake_tool" in second_context
    assert '"args_keys": ["round"]' in second_context
    assert '"result_keys": ["args", "status"]' in second_context
    assert '"args": {"round": 1}' not in second_context
    assert second_context.index("Plan:") < second_context.index("Observations:")
    assert second_context.index("Observations:") < second_context.index("Pending next step:")
    assert "Pending next step:" in second_context
    assert "Continue from the latest successful tool observation" in second_context
    assert any(
        source["source_type"] == "observation" and source["name"] == "fake_tool"
        for source in second_manifest["sources"]
    )
    assert any(
        source["source_type"] == "plan" and source["name"] == "plan.json"
        for source in second_manifest["sources"]
    )
    assert any(
        source["source_type"] == "pending_next_step" and source["name"] == "pending_next_step"
        for source in second_manifest["sources"]
    )
    assert second_manifest["next_action"]["status"] == "continue"
    assert second_manifest["next_action"]["based_on_observation_index"] == 0
    assert second_manifest["next_action"]["based_on_tool_name"] == "fake_tool"
    assert "latest successful tool observation" in second_manifest["next_action"]["reason"]
    for context_id in ["0001", "0002", "0003"]:
        context_manifest = json.loads(
            (result.episode_path / "contexts" / f"{context_id}.json").read_text(encoding="utf-8"),
        )
        assert all(source["budget"]["included_in_model_input"] is True for source in context_manifest["sources"])
        assert all(isinstance(source["budget"]["char_count"], int) for source in context_manifest["sources"])
        assert all(source["budget"]["inclusion_reason"] for source in context_manifest["sources"])
    assert [context["budget"]["context_id"] for context in run_manifest["contexts"]] == [
        "0001",
        "0002",
        "0003",
    ]
    assert all(context["budget"]["source_count"] > 0 for context in run_manifest["contexts"])


def test_orchestrator_verifies_immediately_when_model_returns_no_tools(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = SequenceGateway([ModelResponse("no tools", [])])

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert result.state_history == [
        RunStatus.CREATED,
        RunStatus.PLANNING,
        RunStatus.VERIFYING,
        RunStatus.COMPLETED,
    ]
    transcript = _read_transcript(result.episode_path)
    assert [record.get("event") for record in transcript].count("model_call") == 1
    assert [record.get("event") for record in transcript].count("tool_observation") == 0


def test_orchestrator_fails_when_loop_exceeds_max_turns(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = SequenceGateway(
        [
            ModelResponse("round 1", [ToolCall("fake_tool", {"round": 1})]),
            ModelResponse("round 2", [ToolCall("fake_tool", {"round": 2})]),
            ModelResponse("round 3", [ToolCall("fake_tool", {"round": 3})]),
        ],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway, max_turns=3).run(task_path)

    assert result.status is RunStatus.FAILED
    assert result.state_history[-1] is RunStatus.FAILED
    failure_text = (result.episode_path / "failure-attribution.md").read_text(encoding="utf-8")
    assert "Loop Limit Failure" in failure_text


def test_orchestrator_model_call_has_context_id_each_turn(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool"])
    gateway = SequenceGateway(
        [
            ModelResponse("round 1", [ToolCall("fake_tool", {})]),
            ModelResponse("done", []),
        ],
    )

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    transcript = _read_transcript(result.episode_path)
    model_calls = [record for record in transcript if record.get("event") == "model_call"]
    assert result.status is RunStatus.COMPLETED
    assert [record["context_id"] for record in model_calls] == ["0001", "0002"]


def test_orchestrator_passes_model_input_and_tool_schemas_to_gateway(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    runs_dir = tmp_path / ".runs"
    write_task(task_path, ["fake_tool", "file_read"])
    gateway = SequenceGateway([ModelResponse("done", [])])

    result = RunOrchestrator(runs_root=runs_dir, model_gateway=gateway).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert gateway.model_inputs_seen[0] is not None
    assert "HaAgent Context v1" in gateway.model_inputs_seen[0]
    assert [schema["name"] for schema in gateway.tool_schemas_seen[0]] == [
        "fake_tool",
        "file_read",
    ]
    assert gateway.tool_schemas_seen[0][0]["parameters"]["type"] == "object"


def _read_transcript(episode_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
