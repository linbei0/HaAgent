"""
tests/test_cli.py - AgentFoundry CLI 测试

验证 run/inspect 子命令参数解析、runs-root 传递和 stdout 输出。
"""

import json
from datetime import datetime
from pathlib import Path

from agentfoundry import cli
from agentfoundry.models.gateway import ModelResponse, ToolCall
from agentfoundry.runtime.episode_validator import EpisodePackageView
from agentfoundry.runtime.orchestrator import RunOrchestrator
from agentfoundry.runtime.state import RunStatus


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


class OneShotGateway:
    provider_name = "one-shot"

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        if observations:
            return ModelResponse("done", [])
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class ShellOnceGateway:
    provider_name = "shell-once"

    def __init__(self) -> None:
        self._called = False

    def generate(self, task, model_input=None, tool_schemas=None, observations=None):
        if self._called or observations:
            return ModelResponse("done", [])
        self._called = True
        return ModelResponse("shell", [ToolCall("shell", {"command": "echo approval"})])


def write_minimal_episode(
    episode_path: Path,
    episode_json: dict[str, object] | None = None,
    failure_json: dict[str, object] | None = None,
) -> None:
    (episode_path / "verification").mkdir(parents=True)
    if episode_json is not None:
        (episode_path / "episode.json").write_text(json.dumps(episode_json), encoding="utf-8")
        (episode_path / "task.yaml").write_text(
            """
goal: Inspect me
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
            encoding="utf-8",
        )
        (episode_path / "environment.json").write_text(
            json.dumps(
                {
                    "python": "3.13",
                    "platform": "test",
                    "created_at": "2026-06-19T00:00:00+00:00",
                    "workspace_root": episode_json.get("workspace_root"),
                },
            ),
            encoding="utf-8",
        )
        (episode_path / "sandbox.json").write_text(
            json.dumps(
                {
                    "workspace_root": episode_json.get("workspace_root"),
                    "filesystem_boundary": "workspace_root",
                    "network_policy": "unrestricted",
                    "process_policy": "local_subprocess",
                    "credential_policy": "inherit_environment",
                    "resource_limits": {"command_timeout_seconds": 60},
                },
            ),
            encoding="utf-8",
        )
        (episode_path / "plan.json").write_text(
            json.dumps(
                {
                    "goal": "Inspect me",
                    "allowed_tools": ["fake_tool"],
                    "acceptance_criteria": [],
                    "verification_commands": [],
                    "planned_steps": ["Use allowed tools: fake_tool."],
                },
            ),
            encoding="utf-8",
        )
    if failure_json is not None:
        (episode_path / "failure.json").write_text(json.dumps(failure_json), encoding="utf-8")
    (episode_path / "context-manifest.json").write_text(
        json.dumps({"summary": {"provider": "fake"}, "context_count": 0, "contexts": []}),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    (episode_path / "verification" / "commands.jsonl").write_text("", encoding="utf-8")
    (episode_path / "failure-attribution.md").write_text("legacy", encoding="utf-8")


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def test_cli_run_uses_default_runs_root_and_prints_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path)])

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}
    output = capsys.readouterr().out
    assert "status=completed" in output
    assert f"episode_path={tmp_path / '.runs' / 'episode-1'}" in output


def test_cli_run_accepts_custom_runs_root(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    custom_runs = tmp_path / "custom-runs"
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(custom_runs / "episode-1")

    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path), "--runs-root", str(custom_runs)])

    assert exit_code == 0
    assert calls == {"runs_root": custom_runs, "task_path": task_path}


def test_cli_run_explicit_fake_provider_keeps_default_gateway_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path), "--provider", "fake"])

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}


def test_cli_run_fake_provider_ignores_base_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "fake",
            "--base-url",
            "https://compatible.example/v1",
        ],
    )

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}


def test_cli_run_openai_provider_passes_gateway_to_orchestrator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIGateway:
        provider_name = "openai"

        def __init__(self, model: str = "gpt-4.1-mini") -> None:
            calls["model"] = model

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, model_gateway=None) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "OpenAIResponsesGateway", FakeOpenAIGateway)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(
        ["run", str(task_path), "--provider", "openai", "--model", "gpt-test"],
    )

    assert exit_code == 0
    assert calls["runs_root"] == Path(".runs")
    assert calls["task_path"] == task_path
    assert calls["model"] == "gpt-test"
    assert isinstance(calls["model_gateway"], FakeOpenAIGateway)


def test_cli_run_openai_provider_passes_base_url_to_gateway(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIGateway:
        provider_name = "openai"

        def __init__(
            self,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            calls["model"] = model
            calls["base_url"] = base_url

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, model_gateway=None) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "OpenAIResponsesGateway", FakeOpenAIGateway)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--base-url",
            "https://compatible.example/v1",
        ],
    )

    assert exit_code == 0
    assert calls["model"] == "gpt-test"
    assert calls["base_url"] == "https://compatible.example/v1"
    assert isinstance(calls["model_gateway"], FakeOpenAIGateway)


def test_cli_run_base_url_argument_takes_priority_over_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIGateway:
        provider_name = "openai"

        def __init__(
            self,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            calls["base_url"] = base_url

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, model_gateway=None) -> None:
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "OpenAIResponsesGateway", FakeOpenAIGateway)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "openai",
            "--base-url",
            "https://cli.example/v1",
        ],
    )

    assert exit_code == 0
    assert calls["base_url"] == "https://cli.example/v1"


def test_cli_run_openai_chat_provider_passes_gateway_to_orchestrator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIChatGateway:
        provider_name = "openai-chat"

        def __init__(
            self,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            calls["model"] = model
            calls["base_url"] = base_url

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, model_gateway=None) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli, "OpenAIChatCompletionsGateway", FakeOpenAIChatGateway)
    monkeypatch.setattr(cli, "RunOrchestrator", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "openai-chat",
            "--model",
            "deepseek-chat",
            "--base-url",
            "https://api.deepseek.com/v1",
        ],
    )

    assert exit_code == 0
    assert calls["runs_root"] == Path(".runs")
    assert calls["task_path"] == task_path
    assert calls["model"] == "deepseek-chat"
    assert calls["base_url"] == "https://api.deepseek.com/v1"
    assert isinstance(calls["model_gateway"], FakeOpenAIChatGateway)


def test_cli_run_openai_missing_api_key_fails_without_network(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Fail before OpenAI network call
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Missing key is explicit
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "openai",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    failure_text = (episode_path / "failure.json").read_text(encoding="utf-8")
    assert exit_code == 1
    assert "status=failed" in output
    assert "OPENAI_API_KEY is required for OpenAIResponsesGateway" in failure_text


def test_cli_inspect_completed_episode_outputs_summary(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
    (episode_path / "contexts").mkdir(parents=True)
    (episode_path / "task.yaml").write_text(
        """
goal: Inspect me
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    (episode_path / "episode.json").write_text(
        json.dumps(
            {
                "episode_version": "1.0",
                "created_at": "2026-06-19T00:00:00+00:00",
                "task_path": "task.yaml",
                "status": "completed",
                "provider": "fake",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "context-manifest.json").write_text(
        json.dumps(
            {
                "version": "1.1",
                "context_count": 2,
                "summary": {"provider": "fake", "goal": "Inspect me"},
                    "contexts": [
                        {
                            "context_id": "0001",
                            "model_input_path": "contexts/0001.txt",
                            "manifest_path": "contexts/0001.json",
                            "budget": {
                                "context_id": "0001",
                                "total_chars": 11,
                                "max_chars": 12000,
                                "status": "within_limit",
                                "source_count": 1,
                                "included_source_count": 1,
                            },
                        },
                        {
                            "context_id": "0002",
                            "model_input_path": "contexts/0002.txt",
                            "manifest_path": "contexts/0002.json",
                            "budget": {
                                "context_id": "0002",
                                "total_chars": 12,
                                "max_chars": 12000,
                                "status": "within_limit",
                                "source_count": 0,
                                "included_source_count": 0,
                            },
                        },
                    ],
                },
            ),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "state_transition", "status": "created"}),
                json.dumps(
                    {
                        "event": "planning",
                        "plan_path": "plan.json",
                        "planned_step_count": 1,
                    },
                ),
                json.dumps({"event": "state_transition", "status": "completed"}),
                json.dumps({"event": "model_call", "provider": "fake", "context_id": "0001"}),
                json.dumps({"event": "model_call", "provider": "fake", "context_id": "0002"}),
                json.dumps(
                    {
                        "event": "model_response",
                        "provider": "fake",
                        "turn": 2,
                        "content": "Final answer from inspect fixture.",
                        "tool_calls": [],
                    },
                ),
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text(
        json.dumps(
            {
                "tool_name": "fake_tool",
                "status": "success",
                "policy": {
                    "tool_name": "fake_tool",
                    "risk_level": "low",
                    "action": "allow",
                    "reason": "policy allows low risk tool fake_tool",
                    "approval": {
                        "required": False,
                        "status": "not_required",
                        "reason": "approval not required for low risk tool fake_tool",
                    },
                },
            },
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps({"command": "uv run pytest", "status": "success", "exit_code": 0}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "failure.json").write_text(
        json.dumps({"status": "success", "failure": None}),
        encoding="utf-8",
    )
    (episode_path / "failure-attribution.md").write_text("# Failure Attribution\n\n未失败。\n", encoding="utf-8")
    (episode_path / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.13",
                "platform": "test",
                "created_at": "2026-06-19T00:00:00+00:00",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "sandbox.json").write_text(
        json.dumps(
            {
                "workspace_root": str(tmp_path),
                "filesystem_boundary": "workspace_root",
                "network_policy": "unrestricted",
                "process_policy": "local_subprocess",
                "credential_policy": "inherit_environment",
                "resource_limits": {"command_timeout_seconds": 60},
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "plan.json").write_text(
        json.dumps(
            {
                "goal": "Inspect me",
                "allowed_tools": ["fake_tool"],
                "acceptance_criteria": [],
                "verification_commands": ["uv run pytest"],
                "planned_steps": ["Use allowed tools: fake_tool."],
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "contexts" / "0001.txt").write_text("model input", encoding="utf-8")
    (episode_path / "contexts" / "0001.json").write_text(
        json.dumps(
            {
                "context_id": "0001",
                "budget": {
                    "character_count": 11,
                    "character_limit": 12000,
                    "status": "within_limit",
                },
                "sources": [
                    {
                        "source_type": "task",
                        "name": "task.yaml",
                        "description": "测试任务事实",
                        "inclusion_reason": "inspect fixture",
                        "budget": {
                            "char_count": 11,
                            "included_in_model_input": True,
                            "inclusion_reason": "inspect fixture",
                        },
                    },
                ],
                "next_action": {
                    "status": "none",
                    "reason": "none",
                    "based_on_observation_index": None,
                    "based_on_tool_name": None,
                },
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "contexts" / "0002.txt").write_text("model input 2", encoding="utf-8")
    (episode_path / "contexts" / "0002.json").write_text(
        json.dumps(
            {
                "context_id": "0002",
                "budget": {
                    "character_count": 12,
                    "character_limit": 12000,
                    "status": "within_limit",
                },
                "sources": [],
                "next_action": {
                    "status": "continue",
                    "reason": "Continue from the latest successful tool observation and judge whether the acceptance criteria are satisfied.",
                    "based_on_observation_index": 0,
                    "based_on_tool_name": "fake_tool",
                },
            },
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Run Summary" in output
    assert "episode_version: 1.0" in output
    assert "status: completed" in output
    assert "State Flow" in output
    assert "created -> completed" in output
    assert "Plan" in output
    assert "Use allowed tools: fake_tool." in output
    assert "Next Actions" in output
    assert "0001: status=none based_on_tool_name=none reason=none" in output
    assert "0002: status=continue based_on_tool_name=fake_tool" in output
    assert "latest successful tool observation" in output
    assert "Contexts" in output
    assert "0001" in output
    assert "Sandbox" in output
    assert "filesystem_boundary: workspace_root" in output
    assert "network_policy: unrestricted" in output
    assert "process_policy: local_subprocess" in output
    assert "credential_policy: inherit_environment" in output
    assert "command_timeout_seconds: 60" in output
    assert "Model Calls" in output
    assert "Final Response" in output
    assert "provider=fake turn=2 tool_call_count=0" in output
    assert "Final answer from inspect fixture." in output
    assert "Tool Calls" in output
    assert "fake_tool: success" in output
    assert "Approval Summary" in output
    assert "fake_tool: action=allow approval.required=false approval.status=not_required" in output
    assert "approval not required for low risk tool fake_tool" in output
    assert "Tool Argument Errors" in output
    assert "- none" in output
    assert "Verification" in output
    assert "uv run pytest: success (exit_code=0)" in output
    assert "Failure Attribution" in output
    assert "未失败" in output


def test_cli_inspect_outputs_verification_evidence(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
    (episode_path / "task.yaml").write_text(
        """
goal: Inspect me
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    (episode_path / "episode.json").write_text(
        json.dumps(
            {
                "episode_version": "1.0",
                "created_at": "2026-06-19T00:00:00+00:00",
                "task_path": "task.yaml",
                "status": "failed",
                "provider": "fake",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "context-manifest.json").write_text(
        json.dumps({"summary": {"provider": "fake"}, "context_count": 0, "contexts": []}),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "failed"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    (episode_path / "verification" / "commands.jsonl").write_text(
        json.dumps(
            {
                "command": "python fail.py",
                "status": "failed",
                "exit_code": 3,
                "timeout": False,
                "stdout_excerpt": "out evidence",
                "stderr_excerpt": "err evidence",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    (episode_path / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure": {
                    "category": "Verification Failure",
                    "stage": "verifying",
                    "evidence": "structured evidence",
                },
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "failure-attribution.md").write_text(
        "# Failure Attribution\n\n- category: Verification Failure\n",
        encoding="utf-8",
    )
    (episode_path / "environment.json").write_text(
        json.dumps(
            {
                "python": "3.13",
                "platform": "test",
                "created_at": "2026-06-19T00:00:00+00:00",
                "workspace_root": str(tmp_path),
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "sandbox.json").write_text(
        json.dumps(
            {
                "workspace_root": str(tmp_path),
                "filesystem_boundary": "workspace_root",
                "network_policy": "unrestricted",
                "process_policy": "local_subprocess",
                "credential_policy": "inherit_environment",
                "resource_limits": {"command_timeout_seconds": 60},
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "plan.json").write_text(
        json.dumps(
            {
                "goal": "Inspect me",
                "allowed_tools": ["fake_tool"],
                "acceptance_criteria": [],
                "verification_commands": [],
                "planned_steps": ["Run verification commands if provided."],
            },
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "python fail.py: failed (exit_code=3)" in output
    assert "stdout: out evidence" in output
    assert "stderr: err evidence" in output
    assert "Structured Failure" in output
    assert "category: Verification Failure" in output
    assert "stage: verifying" in output
    assert "structured evidence" in output


def test_cli_inspect_outputs_tool_argument_errors(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Show tool argument errors
constraints: []
allowed_tools:
  - file_read
acceptance_criteria:
  - Argument error is visible
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=OneShotGateway(),
    ).run(task_path)
    verification_dir = result.episode_path / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "commands.jsonl").write_text("", encoding="utf-8")
    inspect_exit = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert result.status is RunStatus.FAILED
    assert inspect_exit == 0
    assert "Tool Argument Errors" in output
    assert "file_read" in output
    assert "missing required argument: path" in output


def test_cli_inspect_approval_summary_shows_none_without_tool_calls(
    tmp_path: Path,
    capsys,
) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Approval Summary\n- none" in output


def test_cli_inspect_approval_summary_handles_legacy_missing(
    tmp_path: Path,
    capsys,
) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "tool-calls.jsonl").write_text(
        json.dumps({"tool_name": "legacy_tool", "status": "success"}) + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Approval Summary" in output
    assert "legacy_tool: legacy/missing" in output


def test_cli_inspect_approval_summary_shows_high_risk_missing(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Inspect missing approval
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=ShellOnceGateway(),
    ).run(task_path)
    verification_dir = result.episode_path / "verification"
    verification_dir.mkdir(exist_ok=True)
    (verification_dir / "commands.jsonl").write_text("", encoding="utf-8")
    inspect_exit = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert result.status is RunStatus.FAILED
    assert inspect_exit == 0
    assert "Approval Summary" in output
    assert "shell: action=deny approval.required=true approval.status=missing" in output
    assert "approval allowed but missing for high risk tool shell" in output


def test_cli_inspect_approval_summary_shows_high_risk_granted(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Inspect granted approval
constraints: []
allowed_tools:
  - shell
acceptance_criteria: []
verification_commands: []
policy:
  approval_allowed_tools:
    - shell
  approved_tools:
    - shell
""".strip(),
        encoding="utf-8",
    )

    def approved_shell(args, workspace_root):
        return {"status": "success", "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr("agentfoundry.tools.router.shell", approved_shell)
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=ShellOnceGateway(),
    ).run(task_path)
    inspect_exit = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert result.status is RunStatus.COMPLETED
    assert inspect_exit == 0
    assert "Approval Summary" in output
    assert "shell: action=allow approval.required=true approval.status=granted" in output
    assert "approval granted for high risk tool shell" in output


def test_cli_inspect_redacts_verification_secrets_from_real_episode(tmp_path: Path, capsys) -> None:
    raw_key = "OPENAI_API_KEY=super-secret-value"
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""
goal: Redact inspect output
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Secret is not shown
verification_commands:
  - python -c "import sys; print('{raw_key}'); sys.exit(9)"
""".strip(),
        encoding="utf-8",
    )

    run_exit = cli.main(["run", str(task_path), "--runs-root", str(tmp_path / ".runs")])
    run_output = capsys.readouterr().out
    episode_path = Path(next(line.split("=", 1)[1] for line in run_output.splitlines() if line.startswith("episode_path=")))
    inspect_exit = cli.main(["inspect", str(episode_path)])

    output = capsys.readouterr().out
    assert run_exit == 1
    assert inspect_exit == 0
    assert raw_key not in output
    assert "OPENAI_API_KEY=[REDACTED]" in output


def test_cli_inspect_legacy_episode_without_episode_json_warns(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    (episode_path / "verification").mkdir(parents=True)
    (episode_path / "contexts").mkdir(parents=True)
    (episode_path / "context-manifest.json").write_text(
        json.dumps(
            {
                "summary": {"provider": "fake"},
                "context_count": 1,
                "contexts": [
                    {
                        "context_id": "0001",
                        "model_input_path": "contexts/0001.txt",
                        "manifest_path": "contexts/0001.json",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (episode_path / "contexts" / "0001.txt").write_text("legacy input", encoding="utf-8")
    (episode_path / "contexts" / "0001.json").write_text(
        json.dumps({"context_id": "0001", "sources": []}),
        encoding="utf-8",
    )
    (episode_path / "transcript.jsonl").write_text(
        json.dumps({"event": "state_transition", "status": "completed"}) + "\n",
        encoding="utf-8",
    )
    (episode_path / "tool-calls.jsonl").write_text("", encoding="utf-8")
    (episode_path / "verification" / "commands.jsonl").write_text("", encoding="utf-8")
    (episode_path / "failure-attribution.md").write_text("legacy", encoding="utf-8")

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "warning: episode.json missing; inspecting legacy episode" in output
    assert "Next Actions" in output
    assert "0001: legacy/missing" in output
    assert "Final Response" in output
    assert "Final Response\n- none" in output
    assert "Approval Summary\n- none" in output


def test_cli_inspect_final_response_content_is_truncated() -> None:
    lines = cli._format_final_response(
        [
            {
                "event": "model_response",
                "provider": "fake",
                "turn": 1,
                "content": "x" * 600,
                "tool_calls": [],
            },
        ],
    )

    assert lines[0] == "- provider=fake turn=1 tool_call_count=0"
    assert lines[1] == f"- content: {'x' * 500}... [truncated]"


def test_cli_inspect_unknown_episode_version_fails(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()
    (episode_path / "episode.json").write_text(
        json.dumps({"episode_version": "9.9"}),
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "unsupported episode_version: 9.9" in output


def test_cli_inspect_fails_when_episode_json_missing_field(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_json = valid_episode_json(tmp_path)
    del episode_json["workspace_root"]
    write_minimal_episode(episode_path, episode_json=episode_json)

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: episode.json missing required field: workspace_root" in output


def test_cli_inspect_fails_when_episode_json_status_is_invalid(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(episode_path, episode_json=valid_episode_json(tmp_path, status="done-ish"))

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: episode.json status is invalid: done-ish" in output


def test_cli_inspect_fails_when_failure_json_category_is_unknown(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path, status="failed"),
        failure_json={
            "status": "failed",
            "failure": {
                "category": "Surprise Failure",
                "stage": "verifying",
                "evidence": "bad",
            },
        },
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: failure.json category is invalid: Surprise Failure" in output


def test_cli_inspect_fails_when_success_failure_json_has_failure(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": {"category": "Runtime Failure"}},
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: failure.json success record must have failure=null" in output


def test_cli_inspect_legacy_episode_without_failure_json_still_works(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(episode_path)

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "legacy episode without failure.json" in output


def test_cli_inspect_new_episode_missing_required_file_uses_validator(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "environment.json").unlink()

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "episode package missing required file: environment.json" in output


def test_cli_export_eval_outputs_valid_json_for_episode(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export CLI eval case
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Eval JSON is emitted
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    assert exit_code == 0
    eval_case = json.loads(capsys.readouterr().out)
    assert eval_case["eval_case_version"] == "1.0"
    assert eval_case["task"]["goal"] == "Export CLI eval case"
    assert "sandbox_summary" in eval_case
    assert eval_case["sandbox_summary"]["command_timeout_seconds"] == 60
    assert eval_case["approval_summary"] == [
        {
            "tool_name": "fake_tool",
            "action": "allow",
            "approval_required": False,
            "approval_status": "not_required",
            "approval_reason": "approval not required for low risk tool fake_tool",
        },
    ]


def test_cli_export_eval_uses_pretty_utf8_json(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: 导出 eval case
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - 输出 UTF-8 JSON
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "\n  \"eval_case_version\"" in output
    assert "导出 eval case" in output
    assert "\\u5bfc\\u51fa" not in output


def test_cli_export_eval_writes_output_file(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case to file
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - JSON file is emitted
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    stdout = capsys.readouterr().out
    eval_case = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout.strip() == f"exported_eval_case={output_path}"
    assert "\"eval_case_version\"" not in stdout
    assert eval_case["eval_case_version"] == "1.0"
    assert "sandbox_summary" in eval_case
    assert "approval_summary" in eval_case


def test_cli_export_eval_output_file_parent_must_exist(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case to missing directory
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "missing-parent" / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "output parent directory does not exist" in output
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_cli_export_eval_invalid_episode_with_output_does_not_write_file(
    tmp_path: Path,
    capsys,
) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "sandbox.json").unlink()
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(episode_path), "--output", str(output_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "episode package missing required file: sandbox.json" in output
    assert not output_path.exists()


def test_cli_export_eval_batch_writes_one_file_per_episode(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_dir = tmp_path / "eval-batch"
    output_dir.mkdir()

    exit_code = cli.main(
        [
            "export-eval",
            str(first.episode_path),
            str(second.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    stdout = capsys.readouterr().out
    first_output = output_dir / f"{first.episode_path.name}.json"
    second_output = output_dir / f"{second.episode_path.name}.json"
    first_case = json.loads(first_output.read_text(encoding="utf-8"))
    second_case = json.loads(second_output.read_text(encoding="utf-8"))
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert f"exported_eval_case={first_output}" in stdout
    assert f"exported_eval_case={second_output}" in stdout
    assert first_case["eval_case_version"] == "1.0"
    assert second_case["eval_case_version"] == "1.0"
    assert "sandbox_summary" in first_case
    assert "approval_summary" in second_case
    assert manifest["manifest_version"] == "1.0"
    assert datetime.fromisoformat(manifest["generated_at"])
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["total_count"] == 2
    assert manifest["success_count"] == 2
    assert manifest["failure_count"] == 0
    assert manifest["records"] == [
        {
            "episode_path": str(first.episode_path),
            "status": "success",
            "output_file": str(first_output),
            "error": None,
        },
        {
            "episode_path": str(second.episode_path),
            "status": "success",
            "output_file": str(second_output),
            "error": None,
        },
    ]


def test_cli_export_eval_batch_continues_after_invalid_episode(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export valid episode in mixed batch
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    valid = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    invalid = tmp_path / "bad-episode"
    write_minimal_episode(
        invalid,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (invalid / "sandbox.json").unlink()
    output_dir = tmp_path / "eval-batch"
    output_dir.mkdir()

    exit_code = cli.main(
        [
            "export-eval",
            str(invalid),
            str(valid.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    stdout = capsys.readouterr().out
    valid_output = output_dir / f"{valid.episode_path.name}.json"
    invalid_output = output_dir / "bad-episode.json"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 1
    assert f"error={invalid}: episode package missing required file: sandbox.json" in stdout
    assert f"exported_eval_case={valid_output}" in stdout
    assert valid_output.exists()
    assert not invalid_output.exists()
    assert manifest["total_count"] == 2
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 1
    assert manifest["records"] == [
        {
            "episode_path": str(invalid),
            "status": "error",
            "output_file": None,
            "error": "episode package missing required file: sandbox.json",
        },
        {
            "episode_path": str(valid.episode_path),
            "status": "success",
            "output_file": str(valid_output),
            "error": None,
        },
    ]


def test_cli_export_eval_single_stdout_does_not_write_manifest(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case without manifest
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(result.episode_path)])

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert not (result.episode_path / "manifest.json").exists()


def test_cli_export_eval_single_output_does_not_write_manifest(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case file without manifest
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_path = tmp_path / "eval-case.json"

    exit_code = cli.main(["export-eval", str(result.episode_path), "--output", str(output_path)])

    assert exit_code == 0
    assert "exported_eval_case=" in capsys.readouterr().out
    assert output_path.exists()
    assert not (tmp_path / "manifest.json").exists()


def test_cli_export_eval_batch_output_dir_must_exist(tmp_path: Path, capsys) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch to missing directory
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    output_dir = tmp_path / "missing-output-dir"

    exit_code = cli.main(
        [
            "export-eval",
            str(first.episode_path),
            str(second.episode_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "output directory does not exist" in output
    assert not output_dir.exists()


def test_cli_export_eval_multiple_episodes_requires_output_dir(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Export eval case batch without output dir
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    first = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    second = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    exit_code = cli.main(["export-eval", str(first.episode_path), str(second.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error:" in output
    assert "multiple episode paths require --output-dir" in output


def test_cli_export_eval_invalid_episode_returns_error(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "sandbox.json").unlink()

    exit_code = cli.main(["export-eval", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "error:" in output
    assert "episode package missing required file: sandbox.json" in output


def test_cli_inspect_new_episode_invalid_jsonl_uses_validator(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path),
        failure_json={"status": "success", "failure": None},
    )
    (episode_path / "transcript.jsonl").write_text("{not json}\n", encoding="utf-8")

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "transcript.jsonl line 1 is not valid JSON" in output


def test_cli_inspect_new_episode_uses_package_view(tmp_path: Path, monkeypatch, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()
    (episode_path / "episode.json").write_text(json.dumps(valid_episode_json(tmp_path)), encoding="utf-8")
    (episode_path / "failure-attribution.md").write_text("view attribution", encoding="utf-8")
    package_view = EpisodePackageView(
        episode_metadata=valid_episode_json(tmp_path),
        failure_record={"status": "success", "failure": None},
        context_manifest={"summary": {"provider": "fake"}, "context_count": 0, "contexts": []},
        transcript=[
            {"event": "state_transition", "status": "created"},
            {"event": "state_transition", "status": "completed"},
        ],
        tool_calls=[{"tool_name": "fake_tool", "status": "success"}],
        verification_commands=[],
    )

    def fail_if_jsonl_is_read(path: Path):
        raise AssertionError(f"unexpected JSONL read: {path}")

    monkeypatch.setattr(cli, "load_validated_episode_package", lambda path: package_view)
    monkeypatch.setattr(cli, "_read_jsonl", fail_if_jsonl_is_read)

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "created -> completed" in output
    assert "fake_tool: success" in output
    assert "view attribution" in output


def test_cli_inspect_fails_when_required_file_is_missing(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_path.mkdir()

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "missing required episode file" in output
    assert "context-manifest.json" in output
