"""
tests/integration/cli/test_cli_run.py - HaAgent run 子命令测试

验证 run 子命令参数解析、provider 组装和最小 stdout 摘要。
"""

import json
from pathlib import Path

import inspect

import pytest

from haagent import cli
from haagent.models.types import ModelResponse, ToolCall
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.contracts.task import load_task


def _cli_gateway_from_profile(profile, gateway_cls):
    """测试用：按 Gateway 构造函数签名转发临时 profile 字段。"""
    params = inspect.signature(gateway_cls.__init__).parameters
    kwargs = {}
    if "model" in params:
        kwargs["model"] = profile.ref.model
    if "base_url" in params:
        kwargs["base_url"] = profile.base_url or None
    if "api_key" in params:
        kwargs["api_key"] = profile.credential.api_key or None
    return gateway_cls(**kwargs)


def _set_home(monkeypatch, home: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: home)


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


class OneShotGateway:
    provider_name = "one-shot"

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        if any(m.get("role") == "tool" for m in messages):
            return ModelResponse("done", [])
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class ShellOnceGateway:
    provider_name = "shell-once"

    def __init__(self) -> None:
        self._called = False

    def generate(self, invocation, **kwargs):
        messages = invocation.messages
        tool_schemas = invocation.tool_schemas
        if self._called or any(m.get("role") == "tool" for m in messages):
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
    (episode_path / "failure-attribution.md").write_text("current attribution", encoding="utf-8")


def valid_episode_json(tmp_path: Path, status: str = "completed") -> dict[str, object]:
    return {
        "episode_version": "1.0",
        "created_at": "2026-06-19T00:00:00+00:00",
        "task_path": "task.yaml",
        "status": status,
        "provider": "fake",
        "workspace_root": str(tmp_path),
    }


def valid_policy(tool_name: str = "fake_tool") -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "risk_level": "low",
        "action": "allow",
        "reason": f"policy allows low risk tool {tool_name}",
        "approval": {
            "required": False,
            "status": "not_required",
            "reason": f"approval not required for low risk tool {tool_name}",
        },
    }


def valid_verification_command(**updates: object) -> dict[str, object]:
    record = {
        "command": "uv run pytest",
        "status": "success",
        "exit_code": 0,
        "timeout": False,
        "stdout_excerpt": "",
        "stderr_excerpt": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_original_length": 0,
        "stderr_original_length": 0,
        "redacted": False,
    }
    record.update(updates)
    return record

def test_cli_run_uses_default_runs_root_and_prints_result(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path)])

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}
    output = capsys.readouterr().out
    assert "status=completed" in output
    assert f"episode_path={tmp_path / '.runs' / 'episode-1'}" in output


def test_cli_run_success_outputs_provider_and_final_response_excerpt(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Run fake summary smoke
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria:
  - Fake model finishes.
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "fake",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status=completed" in output
    assert "episode_path=" in output
    assert "provider=fake" in output
    assert "final_response=Fake model observed tool results." in output


def test_cli_run_goal_entry_starts_fake_provider_task(
    tmp_path: Path,
    capsys,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    exit_code = cli.main(
        [
            "run",
            "--goal",
            "Run the generated task smoke.",
            "--workspace-root",
            str(workspace_root),
            "--verify",
            "python -c \"print('ok')\"",
            "--provider",
            "fake",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in output.splitlines() if line.startswith("episode_path=")),
    )
    task_text = (episode_path / "task.yaml").read_text(encoding="utf-8")
    first_model_input = (episode_path / "contexts" / "0001.json").read_text(encoding="utf-8")

    assert exit_code == 0
    assert "status=completed" in output
    assert "provider=fake" in output
    assert "goal: Run the generated task smoke." in task_text
    assert f"workspace_root: {workspace_root}" in task_text
    assert "- file_list" in task_text
    assert "- grep" in task_text
    assert "- file_read" in task_text
    assert "- apply_patch" in task_text
    assert "- shell" in task_text
    assert "&id" not in task_text
    assert "Task Authoring Ergonomics" not in first_model_input
    assert "This task was generated" not in first_model_input


def test_cli_run_goal_entry_generates_current_task_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root
            calls["max_turns"] = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            task = load_task(received_task_path)
            calls["task_path_name"] = received_task_path.name
            calls["task"] = task
            calls["task_text"] = received_task_path.read_text(encoding="utf-8")
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            "--goal",
            "Fix a small bug.",
            "--workspace-root",
            str(workspace_root),
            "--verify",
            "uv run pytest",
            "--max-turns",
            "9",
        ],
    )

    task = calls["task"]
    assert exit_code == 0
    assert calls["runs_root"] == Path(".runs")
    assert calls["max_turns"] == 9
    assert calls["task_path_name"] == "task.yaml"
    assert task.goal == "Fix a small bug."
    assert task.workspace_root == str(workspace_root)
    assert task.allowed_tools == ["file_list", "grep", "file_read", "apply_patch", "shell"]
    assert task.verification_commands == ["uv run pytest"]
    assert task.policy == {
        "approval_allowed_tools": ["apply_patch", "shell"],
        "approved_tools": ["apply_patch", "shell"],
    }


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["run", "--workspace-root", ".", "--verify", "uv run pytest"],
            "error: --goal is required when task_yaml is omitted",
        ),
        (
            ["run", "--goal", "Fix it", "--verify", "uv run pytest"],
            "error: --workspace-root is required when task_yaml is omitted",
        ),
        (
            ["run", "--goal", "Fix it", "--workspace-root", "."],
            "error: --verify is required when task_yaml is omitted",
        ),
    ],
)
def test_cli_run_goal_entry_reports_missing_required_arguments(
    argv: list[str],
    expected: str,
    capsys,
) -> None:
    exit_code = cli.main(argv)

    output = capsys.readouterr().out
    assert exit_code == 2
    assert expected in output


def test_cli_run_task_yaml_entry_still_ignores_goal_entry_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--goal",
            "ignored",
            "--workspace-root",
            str(tmp_path),
            "--verify",
            "ignored",
        ],
    )

    assert exit_code == 0
    assert calls == {"runs_root": Path(".runs"), "task_path": task_path}


def test_cli_run_uses_default_max_turns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root
            calls["max_turns"] = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path)])

    assert exit_code == 0
    assert calls == {
        "runs_root": Path(".runs"),
        "max_turns": 3,
        "task_path": task_path,
    }


def test_cli_run_passes_custom_max_turns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root
            calls["max_turns"] = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(["run", str(task_path), "--max-turns", "12"])

    assert exit_code == 0
    assert calls["max_turns"] == 12


def test_cli_run_accepts_custom_runs_root(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    custom_runs = tmp_path / "custom-runs"
    calls = {}

    class FakeOrchestrator:
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(custom_runs / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(self, runs_root: Path, max_turns: int = 3) -> None:
            calls["runs_root"] = runs_root

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIChatGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

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


def test_cli_run_openai_chat_provider_passes_custom_max_turns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIChatGateway:
        provider_name = "openai-chat"

        def __init__(self, model: str = "gpt-4.1-mini") -> None:
            calls["model"] = model

    class FakeOrchestrator:
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway
            calls["max_turns"] = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIChatGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--provider",
            "openai-chat",
            "--model",
            "deepseek-chat",
            "--max-turns",
            "12",
        ],
    )

    assert exit_code == 0
    assert calls["model"] == "deepseek-chat"
    assert calls["max_turns"] == 12
    assert isinstance(calls["model_gateway"], FakeOpenAIChatGateway)


def test_cli_run_connection_creates_gateway_from_local_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "connection-secret")
    (tmp_path / "home" / ".haagent").mkdir(parents=True)
    (tmp_path / "home" / ".haagent" / "providers.json").write_text(
        json.dumps(
            {
                "version": 4,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "home" / ".haagent" / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": "deepseek", "model": "deepseek-v4-pro"}}),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")
    calls = {}

    class FakeOpenAIChatGateway:
        provider_name = "openai-chat"

        def __init__(
            self,
            api_key: str | None = None,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            calls["api_key"] = api_key
            calls["model"] = model
            calls["base_url"] = base_url

    class FakeOrchestrator:
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            calls["runs_root"] = runs_root
            calls["model_gateway"] = model_gateway
            calls["max_turns"] = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            calls["task_path"] = received_task_path
            return FakeResult(tmp_path / ".runs" / "episode-1")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FakeOpenAIChatGateway: _cli_gateway_from_profile(profile, _cls))
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(
        [
            "run",
            str(task_path),
            "--profile",
            "deepseek",
            "--max-turns",
            "12",
        ],
    )

    assert exit_code == 0
    assert calls["api_key"] == "connection-secret"
    assert calls["model"] == "deepseek-v4-pro"
    assert calls["base_url"] == "https://api.deepseek.com"
    assert calls["max_turns"] == 12
    assert isinstance(calls["model_gateway"], FakeOpenAIChatGateway)


def test_cli_run_connection_missing_name_fails_explicitly(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "home" / ".haagent").mkdir(parents=True)
    (tmp_path / "home" / ".haagent" / "providers.json").write_text(
        json.dumps({"version": 4, "connections": []}),
        encoding="utf-8",
    )
    (tmp_path / "home" / ".haagent" / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": "deepseek", "model": "deepseek-v4-pro"}}),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")

    exit_code = cli.main(["run", str(task_path), "--profile", "deepseek"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error: provider connection not found: deepseek" in output
    assert not (tmp_path / ".runs").exists()


def test_cli_run_connection_missing_api_key_env_fails_explicitly(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _set_home(monkeypatch, tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / "home" / ".haagent").mkdir(parents=True)
    (tmp_path / "home" / ".haagent" / "providers.json").write_text(
        json.dumps(
            {
                "version": 4,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "home" / ".haagent" / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": "deepseek", "model": "deepseek-v4-pro"}}),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: x\n", encoding="utf-8")

    exit_code = cli.main(["run", str(task_path), "--profile", "deepseek"])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "error: API key is not available for connection deepseek" in output
    assert "api_key_env=DEEPSEEK_API_KEY" in output
    assert not (tmp_path / ".runs").exists()


def test_cli_run_connection_secret_stays_out_of_episode_inspect_eval_and_model_input(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    secret = "connection-secret-value"
    _set_home(monkeypatch, tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    (tmp_path / "home" / ".haagent").mkdir(parents=True)
    (tmp_path / "home" / ".haagent" / "providers.json").write_text(
        json.dumps(
            {
                "version": 4,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "home" / ".haagent" / "settings.json").write_text(
        json.dumps({"active_model": {"connection_id": "deepseek", "model": "deepseek-v4-pro"}}),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Finish without tools
constraints: []
allowed_tools: []
acceptance_criteria:
  - Final response is recorded.
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    class FinalGateway:
        provider_name = "openai-chat"

        def __init__(
            self,
            api_key: str | None = None,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            assert api_key == secret
            assert model == "deepseek-v4-pro"
            assert base_url == "https://api.deepseek.com"

        def generate(self, invocation, **kwargs):
            messages = invocation.messages
            model_input = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
            assert secret not in model_input
            return ModelResponse("done", [])

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "gateway_factory", lambda profile, _cls=FinalGateway: _cli_gateway_from_profile(profile, _cls))

    run_exit = cli.main(
        [
            "run",
            str(task_path),
            "--profile",
            "deepseek",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )
    run_output = capsys.readouterr().out
    episode_path = Path(
        next(line.split("=", 1)[1] for line in run_output.splitlines() if line.startswith("episode_path=")),
    )
    episode_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in episode_path.rglob("*")
        if path.is_file()
    )

    inspect_exit = cli.main(["inspect", str(episode_path)])
    inspect_output = capsys.readouterr().out
    export_exit = cli.main(["export-eval", str(episode_path)])
    export_output = capsys.readouterr().out

    assert run_exit == 0
    assert inspect_exit == 0
    assert export_exit == 0
    assert secret not in run_output
    assert secret not in episode_text
    assert secret not in inspect_output
    assert secret not in export_output


def test_cli_run_rejects_non_positive_max_turns(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "task.yaml", "--max-turns", "0"])

    output = capsys.readouterr().err
    assert error.value.code == 2
    assert "--max-turns must be a positive integer" in output


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
    assert "provider=openai" in output
    assert "failed_stage=planning" in output
    assert "failure_category=Model Failure" in output
    assert "reason=OPENAI_API_KEY is required for OpenAIResponsesGateway" in output
    assert "OPENAI_API_KEY is required for OpenAIResponsesGateway" in failure_text


