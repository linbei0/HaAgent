"""
tests/integration/runtime/test_execution_boundary.py - 本地执行边界测试

覆盖 shell/code_run 的 cwd、timeout、输出摘要、脱敏和临时脚本边界。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from haagent.context.builder import ContextBuilder
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.human_interaction import HumanInteractionResponse
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.path_policy import ExternalRoot, PathPolicy
from haagent.runtime.contracts.plan import build_plan
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from haagent.tools.code_run import code_run
from haagent.tools.base import ToolExecutionContext
from haagent.tools.router import ToolRouter
from haagent.tools.shell import shell


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Test execution boundary
constraints: []
allowed_tools:
  - shell
  - code_run
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    writer = EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
    writer.write_plan(build_plan(_task()))
    return writer


def test_shell_defaults_cwd_to_workspace_root_and_supports_dot(tmp_path: Path) -> None:
    command = f"{sys.executable} -c \"from pathlib import Path; print(Path.cwd().resolve())\""

    missing = shell({"command": command, "timeout_seconds": 5}, tmp_path)
    dot = shell({"command": command, "cwd": ".", "timeout_seconds": 5}, tmp_path)

    assert missing["status"] == "success"
    assert missing["stdout_excerpt"].strip() == str(tmp_path.resolve())
    assert missing["truncated"] is False
    assert dot["status"] == "success"
    assert dot["stdout_excerpt"].strip() == str(tmp_path.resolve())


def test_code_run_terminates_python_process_when_cancelled(tmp_path: Path) -> None:
    token = CancellationToken()

    def cancel_soon() -> None:
        time.sleep(0.05)
        token.cancel()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    result = code_run(
        {"code": "import time\ntime.sleep(5)", "timeout_seconds": 10},
        tmp_path,
        cancellation_token=token,
    )
    thread.join(timeout=1)

    assert result["status"] == "error"
    assert result["error"]["type"] == "cancelled"
    assert result["exit_code"] is None
    assert result["timeout"] is False
    assert result["truncated"] is False


@pytest.mark.parametrize("use_local_sandbox", [False, True])
def test_code_run_defaults_file_io_and_stdio_to_utf8(tmp_path: Path, use_local_sandbox: bool) -> None:
    source = tmp_path / "utf8-source.txt"
    target = tmp_path / "utf8-output.txt"
    source.write_text("压缩预算 € →", encoding="utf-8")
    backend = (
        LocalSubprocessSandboxBackend(
            workspace_root=tmp_path,
            command_timeout_seconds=5,
        )
        if use_local_sandbox
        else None
    )
    result = code_run(
        {
            "code": (
                "from pathlib import Path\n"
                "import sys\n"
                "text = Path('utf8-source.txt').read_text()\n"
                "Path('utf8-output.txt').write_text(text)\n"
                "print(sys.flags.utf8_mode)\n"
                "print(text)\n"
            ),
            "timeout_seconds": 5,
        },
        tmp_path,
        sandbox_backend=backend,
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].splitlines() == ["1", "压缩预算 € →"]
    assert target.read_bytes() == "压缩预算 € →".encode("utf-8")


def test_shell_rejects_workspace_escape_before_execution(tmp_path: Path) -> None:
    outside = tmp_path.parent / "shell_escape_marker.txt"

    result = shell(
        {
            "command": f"{sys.executable} -c \"from pathlib import Path; Path(r'{outside}').write_text('bad')\"",
            "cwd": "..",
            "timeout_seconds": 5,
        },
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_required"
    assert "用户确认" in result["error"]["message"]
    assert not outside.exists()


def test_code_run_rejects_workspace_escape_before_script_creation(tmp_path: Path) -> None:
    outside = tmp_path.parent / "code_run_escape_marker.txt"

    result = code_run(
        {
            "code": f"from pathlib import Path\nPath(r'{outside}').write_text('bad')",
            "cwd": "..",
            "timeout_seconds": 5,
        },
        tmp_path,
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_required"
    assert not outside.exists()
    assert not (tmp_path / ".haagent-tmp").exists()


def test_timeout_above_execution_boundary_is_rejected(tmp_path: Path) -> None:
    result = shell({"command": f"{sys.executable} -c \"print('no run')\"", "timeout_seconds": 9999}, tmp_path)

    assert result["status"] == "error"
    assert result["error"] == {
        "type": "tool_argument_invalid",
        "category": "argument",
        "message": "timeout_seconds must be <= 120",
        "retryable": False,
    }
    assert result["recovery"]["action"] == "correct_arguments"


def test_shell_timeout_result_is_structured(tmp_path: Path) -> None:
    result = shell({"command": f"{sys.executable} -c \"import time; time.sleep(2)\"", "timeout_seconds": 0.1}, tmp_path)

    assert result["status"] == "error"
    assert result["exit_code"] is None
    assert result["timeout"] is True
    assert result["error"]["type"] == "timeout"
    assert "stdout_excerpt" in result
    assert "stderr_excerpt" in result
    assert "truncated" in result


def test_shell_non_zero_exit_is_recoverable_observation(tmp_path: Path) -> None:
    result = shell({"command": f"{sys.executable} -c \"import sys; sys.exit(7)\""}, tmp_path)

    assert result["status"] == "error"
    assert result["exit_code"] == 7
    assert result["execution_state"] == "completed"
    assert result["error"]["type"] == "command_failed"
    assert result["recovery"]["action"] == "correct_arguments"


def test_shell_long_stdout_and_stderr_are_excerpted(tmp_path: Path) -> None:
    command = (
        f"{sys.executable} -c "
        "\"import sys; print('o' * 5000); print('e' * 5000, file=sys.stderr)\""
    )

    result = shell({"command": command, "timeout_seconds": 5}, tmp_path)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["stdout_excerpt"]) < 5000
    assert len(result["stderr_excerpt"]) < 5000
    assert "stdout" not in result
    assert "stderr" not in result


def test_secret_like_output_is_redacted_from_tool_result_and_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    secret = "super-secret-value-12345"
    monkeypatch.setenv("HAAGENT_TEST_SECRET_TOKEN", secret)
    result = shell(
        {"command": f"{sys.executable} -c \"import os; print(os.environ['HAAGENT_TEST_SECRET_TOKEN'])\"", "timeout_seconds": 5},
        tmp_path,
    )
    writer = make_writer(tmp_path)
    context = ContextBuilder(
        task=_task(),
        workspace_root=tmp_path,
        provider_name="fake",
        episode_writer=writer,
        observations=[{"tool_name": "shell", "args": {"command": "print secret"}, "result": result}],
    ).build()

    assert result["status"] == "success"
    assert secret not in json.dumps(result, ensure_ascii=False)
    assert "[REDACTED_SECRET]" in result["stdout_excerpt"]
    assert secret not in context.model_input


def test_code_run_script_path_stays_inside_workspace(tmp_path: Path) -> None:
    result = code_run({"code": "print('ok')", "timeout_seconds": 5}, tmp_path)

    assert result["status"] == "success"
    assert result["script_path"].startswith(".haagent-tmp/")
    assert (tmp_path / result["script_path"]).resolve().is_file()
    assert tmp_path.resolve() in (tmp_path / result["script_path"]).resolve().parents


def test_guardrail_blocks_shell_and_code_run_before_real_execution(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(
        allowed_tools=["shell", "code_run"],
        episode_writer=writer,
        workspace_root=tmp_path,
        approval_allowed_tools=["shell", "code_run"],
        approved_tools=["shell", "code_run"],
    )
    shell_marker = tmp_path / "shell_marker.txt"
    code_marker = tmp_path / "code_marker.txt"

    shell_result = router.dispatch(
        "shell",
        {
            "command": (
                f"{sys.executable} -c \"from pathlib import Path; "
                f"Path(r'{shell_marker}').write_text('bad'); print('api_key')\""
            ),
            "timeout_seconds": 5,
        },
    )
    code_result = router.dispatch(
        "code_run",
        {
            "code": f"from pathlib import Path\nPath(r'{code_marker}').write_text('bad')\nprint('api_key')",
            "timeout_seconds": 5,
        },
    )

    assert shell_result["error"]["type"] == "guardrail_denied"
    assert code_result["error"]["type"] == "guardrail_denied"
    assert not shell_marker.exists()
    assert not code_marker.exists()
    assert not (tmp_path / ".haagent-tmp").exists()
    records = [
        json.loads(line)
        for line in (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["guardrail"]["rule_id"] for record in records] == [
        "shell_secret_exfiltration",
        "code_run_secret_access",
    ]


def test_external_read_root_cannot_be_execution_cwd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    result = shell(
        {"command": f"{sys.executable} -c \"print('should not run')\"", "cwd": str(external), "timeout_seconds": 5},
        project,
        path_policy=PathPolicy(
            project_root=project,
            external_roots=[ExternalRoot(path=external, access="read", source="user", created_at="now")],
        ),
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_required"
    assert "用户确认" in result["error"]["message"]


def test_external_full_root_can_be_execution_cwd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    result = shell(
        {"command": f"{sys.executable} -c \"from pathlib import Path; print(Path.cwd().resolve())\"", "cwd": str(external), "timeout_seconds": 5},
        project,
        path_policy=PathPolicy(
            project_root=project,
            external_roots=[ExternalRoot(path=external, access="full", source="user", created_at="now")],
        ),
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(external.resolve())


def test_auto_approve_path_policy_does_not_allow_external_execution_cwd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    result = shell(
        {"command": f"{sys.executable} -c \"print('should not run')\"", "cwd": str(external), "timeout_seconds": 5},
        project,
        path_policy=PathPolicy(project_root=project, permission_mode="auto_approve"),
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "path_policy_denied"
    assert "目录未授权" in result["error"]["message"]


def test_full_access_path_policy_allows_external_execution_cwd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    result = shell(
        {"command": f"{sys.executable} -c \"from pathlib import Path; print(Path.cwd().resolve())\"", "cwd": str(external), "timeout_seconds": 5},
        project,
        path_policy=PathPolicy(project_root=project, permission_mode="full_access"),
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(external.resolve())


def test_request_approval_allows_external_execution_cwd_after_confirmation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=True)

    result = shell(
        {"command": f"{sys.executable} -c \"from pathlib import Path; print(Path.cwd().resolve())\"", "cwd": str(external), "timeout_seconds": 5},
        project,
        path_policy=PathPolicy(project_root=project, permission_mode="request_approval"),
        execution_context=ToolExecutionContext(interaction_handler=interaction),
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == str(external.resolve())
    assert len(requests) == 1
    assert requests[0].tool_name == "external_directory"
    assert requests[0].args_summary["access"] == "full"


def test_shell_requests_external_directory_from_powershell_command_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    external = tmp_path / "profile"
    project.mkdir()
    external.mkdir()
    target = external / "providers.json"
    target.write_text('{"version": 4}', encoding="utf-8")
    monkeypatch.setenv("HAAGENT_TEST_PROFILE", str(external))
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=False, answer="deny")

    result = shell(
        {
            "command": r"Get-Content $env:HAAGENT_TEST_PROFILE\providers.json",
            "timeout_seconds": 5,
        },
        project,
        path_policy=PathPolicy(project_root=project, permission_mode="request_approval"),
        execution_context=ToolExecutionContext(interaction_handler=interaction),
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert [request.tool_name for request in requests] == ["external_directory"]
    assert requests[0].args_summary["directories"] == [str(external.resolve())]


def test_code_run_declared_external_directory_requests_permission(tmp_path: Path) -> None:
    project = tmp_path / "project"
    external = tmp_path / "external"
    project.mkdir()
    external.mkdir()
    requests = []

    def interaction(request):
        requests.append(request)
        return HumanInteractionResponse(approved=False, answer="deny")

    result = code_run(
        {
            "code": "print('must not run')",
            "external_directories": [str(external)],
            "timeout_seconds": 5,
        },
        project,
        path_policy=PathPolicy(project_root=project, permission_mode="request_approval"),
        execution_context=ToolExecutionContext(interaction_handler=interaction),
    )

    assert result["status"] == "error"
    assert result["error"]["type"] == "approval_denied"
    assert [request.tool_name for request in requests] == ["external_directory"]
    assert not (project / ".haagent-tmp").exists()


def test_low_privilege_allows_local_shell_in_workspace(tmp_path: Path) -> None:
    result = shell(
        {"command": "echo should-not-run", "timeout_seconds": 5},
        tmp_path,
        path_policy=PathPolicy(project_root=tmp_path, permission_mode="auto_approve"),
        sandbox_backend=LocalSubprocessSandboxBackend(
            workspace_root=tmp_path,
            command_timeout_seconds=5,
        ),
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == "should-not-run"


def test_low_privilege_allows_local_code_run_in_workspace(tmp_path: Path) -> None:
    result = code_run(
        {"code": "print('should-not-run')", "timeout_seconds": 5},
        tmp_path,
        path_policy=PathPolicy(project_root=tmp_path, permission_mode="request_approval"),
        sandbox_backend=LocalSubprocessSandboxBackend(
            workspace_root=tmp_path,
            command_timeout_seconds=5,
        ),
    )

    assert result["status"] == "success"
    assert result["stdout_excerpt"].strip() == "should-not-run"


def _task() -> TaskSpec:
    return TaskSpec(
        goal="Test execution boundary",
        constraints=[],
        allowed_tools=["shell", "code_run"],
        acceptance_criteria=[],
        verification_commands=[],
    )
