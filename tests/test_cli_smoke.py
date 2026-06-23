"""
tests/test_cli_smoke.py - HaAgent smoke 子命令测试

验证 smoke 子命令的 fake/real profile 编排和失败摘要输出。
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from haagent import cli
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.episode_validator import EpisodePackageView
from haagent.runtime.orchestrator import RunOrchestrator
from haagent.runtime.state import RunStatus


class FakeResult:
    status = RunStatus.COMPLETED

    def __init__(self, episode_path: Path) -> None:
        self.episode_path = episode_path


class OneShotGateway:
    provider_name = "one-shot"

    def generate(self, task, model_input, tool_schemas, observations):
        if observations:
            return ModelResponse("done", [])
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class ShellOnceGateway:
    provider_name = "shell-once"

    def __init__(self) -> None:
        self._called = False

    def generate(self, task, model_input, tool_schemas, observations):
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

def test_cli_smoke_runs_fake_hello_by_default(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(["smoke", "--runs-root", str(tmp_path / ".runs")])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "smoke=hello" in output
    assert "status=completed" in output
    assert "episode_path=" in output
    assert "real_file_read" not in output
    assert "real_edit_verify" not in output


def test_cli_smoke_with_profile_runs_fake_and_real_tasks(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "profile-secret")
    (tmp_path / ".haagent").mkdir()
    (tmp_path / ".haagent" / "providers.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    calls = []

    class FakeOpenAIChatGateway:
        provider_name = "openai-chat"

        def __init__(
            self,
            api_key: str | None = None,
            model: str = "gpt-4.1-mini",
            base_url: str | None = None,
        ) -> None:
            assert api_key == "profile-secret"
            assert model == "deepseek-v4-pro"
            assert base_url == "https://api.deepseek.com"

    class FakeOrchestrator:
        def __init__(
            self,
            runs_root: Path,
            model_gateway=None,
            max_turns: int = 3,
        ) -> None:
            self._runs_root = runs_root
            self._model_gateway = model_gateway
            self._max_turns = max_turns

        def run(self, received_task_path: Path) -> FakeResult:
            calls.append(
                (received_task_path.as_posix(), self._model_gateway, self._max_turns),
            )
            return FakeResult(self._runs_root / f"episode-{len(calls)}")

    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "chat_gateway_cls", FakeOpenAIChatGateway)
    monkeypatch.setattr(cli.DEFAULT_RUNTIME, "orchestrator_cls", FakeOrchestrator)

    exit_code = cli.main(
        [
            "smoke",
            "--profile",
            "deepseek",
            "--runs-root",
            str(tmp_path / ".runs"),
            "--max-turns",
            "9",
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "smoke=hello" in output
    assert "smoke=real_file_read" in output
    assert "smoke=real_edit_verify" in output
    assert output.count("status=completed") == 3
    assert calls[0][0].endswith("examples/tasks/hello.yaml")
    assert calls[0][1] is None
    assert calls[1][0].endswith("examples/tasks/openai_chat_file_read_smoke.yaml")
    assert isinstance(calls[1][1], FakeOpenAIChatGateway)
    assert calls[2][0].endswith("examples/tasks/openai_chat_edit_smoke.yaml")
    assert isinstance(calls[2][1], FakeOpenAIChatGateway)
    assert [call[2] for call in calls] == [9, 9, 9]


def test_cli_smoke_missing_profile_reports_real_failures_after_fake(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        [
            "smoke",
            "--profile",
            "deepseek",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "smoke=hello" in output
    assert "status=completed" in output
    assert "smoke=real_file_read" in output
    assert "smoke=real_edit_verify" in output
    assert output.count("status=failed") == 2
    assert "episode_path=none" in output
    assert "failed_stage=configuration" in output
    assert "failure_category=Provider Profile Error" in output
    assert "reason=provider profile not found: deepseek; searched:" in output
    assert ".haagent\\providers.json" in output


def test_cli_smoke_missing_profile_api_key_env_reports_real_failures_after_fake(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    profile_dir = tmp_path / ".haagent"
    profile_dir.mkdir()
    (profile_dir / "providers.json").write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "smoke",
            "--profile",
            "deepseek",
            "--runs-root",
            str(tmp_path / ".runs"),
        ],
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "smoke=hello" in output
    assert "status=completed" in output
    assert "smoke=real_file_read" in output
    assert "smoke=real_edit_verify" in output
    assert output.count("status=failed") == 2
    assert "failed_stage=configuration" in output
    assert "failure_category=Provider Profile Error" in output
    assert "reason=api key environment variable is not set: DEEPSEEK_API_KEY" in output


