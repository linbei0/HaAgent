"""
tests/test_cli_inspect.py - HaAgent inspect 子命令测试

验证 episode inspect 摘要渲染和损坏 package 错误输出。
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from haagent import cli, cli_inspect
from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.episode_validator import EpisodePackageView
from haagent.runtime.human_interaction import HumanInteractionResponse
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


class NoToolGateway:
    provider_name = "no-tool"

    def generate(self, task, model_input, tool_schemas, observations):
        return ModelResponse("done", [])


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
        json.dumps(valid_verification_command()) + "\n",
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
                            "raw_char_count": 11,
                            "model_input_char_count": 11,
                            "included_in_model_input": True,
                            "truncated": False,
                            "inclusion_reason": "inspect fixture",
                            "exclusion_reason": None,
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
            valid_verification_command(
                command="python fail.py",
                status="failed",
                exit_code=3,
                timeout=False,
                stdout_excerpt="out evidence",
                stderr_excerpt="err evidence",
                stdout_original_length=len("out evidence"),
                stderr_original_length=len("err evidence"),
            ),
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


def test_cli_inspect_outputs_workspace_preflight(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""
goal: Inspect workspace preflight
workspace_root: {workspace}
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=NoToolGateway(),
    ).run(task_path)

    exit_code = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Workspace Preflight" in output
    assert f"workspace_root: {workspace.resolve()}" in output
    assert "exists: true" in output
    assert "git_status: not_git_repo" in output
    assert "modifies_original_workspace: true" in output


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


def test_cli_inspect_rejects_tool_call_missing_policy(
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
        json.dumps({"tool_name": "fake_tool", "status": "success"}) + "\n",
        encoding="utf-8",
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "tool-calls.jsonl line 1 missing required field: policy" in output


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

    monkeypatch.setattr("haagent.tools.router.shell", approved_shell)
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


def test_cli_inspect_shows_human_interaction_events_for_denied_approval(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Inspect denied approval
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
        interaction_handler=lambda request: HumanInteractionResponse(approved=False, answer="no"),
    ).run(task_path)
    inspect_exit = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert result.status is RunStatus.FAILED
    assert inspect_exit == 0
    assert "Human Interactions" in output
    assert "- approval_requested: tool=shell" in output
    assert "- approval_denied: tool=shell" in output
    assert "User Denied Failure" in output


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


def test_cli_inspect_missing_episode_json_fails(tmp_path: Path, capsys) -> None:
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
    (episode_path / "contexts" / "0001.txt").write_text("input", encoding="utf-8")
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
    (episode_path / "failure-attribution.md").write_text("current attribution", encoding="utf-8")

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "episode package missing required file: episode.json" in output


def test_cli_inspect_final_response_content_is_truncated() -> None:
    lines = cli_inspect._format_final_response(
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
    episode_json = valid_episode_json(tmp_path)
    episode_json["episode_version"] = "9.9"
    write_minimal_episode(
        episode_path,
        episode_json=episode_json,
        failure_json={"status": "success", "failure": None},
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "unsupported episode_version: 9.9" in output


def test_cli_inspect_fails_when_episode_json_missing_field(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    episode_json = valid_episode_json(tmp_path)
    del episode_json["workspace_root"]
    write_minimal_episode(
        episode_path,
        episode_json=episode_json,
        failure_json={"status": "success", "failure": None},
    )

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "corrupt episode: episode.json missing required field: workspace_root" in output


def test_cli_inspect_fails_when_episode_json_status_is_invalid(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(
        episode_path,
        episode_json=valid_episode_json(tmp_path, status="done-ish"),
        failure_json={"status": "success", "failure": None},
    )

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


def test_cli_inspect_missing_failure_json_fails(tmp_path: Path, capsys) -> None:
    episode_path = tmp_path / "episode-1"
    write_minimal_episode(episode_path, episode_json=valid_episode_json(tmp_path))

    exit_code = cli.main(["inspect", str(episode_path)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "episode package missing required file: failure.json" in output


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


def test_cli_inspect_failed_episode_before_verifying_allows_missing_verification_commands(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Fail before verification
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands:
  - python -c "print('not reached')"
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=OneShotGateway(),
    ).run(task_path)

    commands_path = result.episode_path / "verification" / "commands.jsonl"
    assert result.status is RunStatus.FAILED
    assert not commands_path.exists()

    exit_code = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Verification" in output
    assert "- not reached" in output
    assert "Tool Calls" in output
    assert "file_read: error" in output
    assert "Final Response" in output
    assert "provider=one-shot turn=1 tool_call_count=1" in output
    assert "Failure Attribution" in output
    assert "missing required argument: path" in output


def test_cli_inspect_task_loading_failure_outputs_failure_summary_without_later_files(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: []\n", encoding="utf-8")
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    assert result.status is RunStatus.FAILED
    assert not (result.episode_path / "plan.json").exists()
    assert not (result.episode_path / "context-manifest.json").exists()
    assert not (result.episode_path / "verification" / "commands.jsonl").exists()

    exit_code = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"episode_path: {result.episode_path}" in output
    assert "status: failed" in output
    assert "provider: unknown" in output
    assert "created -> failed" in output
    assert "Verification" in output
    assert "- not reached" in output
    assert "Structured Failure" in output
    assert "category: Task Spec Failure" in output
    assert "stage: created" in output
    assert "goal must be a string" in output


def test_cli_inspect_failed_episode_after_verifying_missing_verification_commands_fails(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Verification reached package must stay strict
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands:
  - python -c "import sys; sys.exit(7)"
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").unlink()

    exit_code = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert result.status is RunStatus.FAILED
    assert exit_code == 1
    assert "episode package missing required file: verification/commands.jsonl" in output


def test_cli_inspect_completed_episode_missing_verification_commands_still_fails(
    tmp_path: Path,
    capsys,
) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Completed package must stay strict
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)
    (result.episode_path / "verification" / "commands.jsonl").unlink()

    exit_code = cli.main(["inspect", str(result.episode_path)])

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "episode package missing required file: verification/commands.jsonl" in output


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
        tool_calls=[{"tool_name": "fake_tool", "status": "success", "policy": valid_policy()}],
        verification_commands=[],
        sandbox={
            "workspace_root": str(tmp_path),
            "filesystem_boundary": "workspace_root",
            "network_policy": "unrestricted",
            "process_policy": "local_subprocess",
            "credential_policy": "inherit_environment",
            "resource_limits": {"command_timeout_seconds": 60},
        },
    )

    monkeypatch.setattr(cli_inspect, "load_inspect_episode_package", lambda path: package_view)

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
    assert "episode package missing required file: episode.json" in output
