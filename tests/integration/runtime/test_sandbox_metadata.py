"""
tests/integration/runtime/test_sandbox_metadata.py - 沙箱 metadata 集成测试

验证本机沙箱后端 metadata 会以 expanded sandbox.json schema 写入 episode。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.sandbox.base import SandboxAvailability, SandboxMetadata
from haagent.runtime.execution.command import ShellContract
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from tests.support.episode_packages import read_json, write_task


class FakeLifecycleSandboxBackend:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.resolve()
        self.closed = False

    def metadata(self) -> SandboxMetadata:
        return SandboxMetadata(
            workspace_root=str(self.workspace_root),
            filesystem_boundary="workspace_root",
            backend="fake_sandbox",
            process_policy="fake_process",
            network_policy="none",
            credential_policy="minimal_env",
            resource_limits={"command_timeout_seconds": 60},
            isolation={
                "no_new_privileges": True,
                "cap_drop": ["ALL"],
                "read_only_rootfs": True,
                "user": "fake",
                "privileged": False,
            },
            availability=SandboxAvailability(
                available=True,
                degraded=False,
                reason="",
            ),
        )

    def shell_contract(self) -> ShellContract:
        return ShellContract("posix", "fake-sh", "fake")

    def run_shell(self, command):
        raise AssertionError("tool execution is not needed in this test")

    def run_python(self, script_path, command):
        raise AssertionError("tool execution is not needed in this test")

    def close(self) -> None:
        self.closed = True


def test_episode_writer_writes_expanded_sandbox_metadata(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text("goal: sandbox metadata\n", encoding="utf-8")
    writer = EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)
    backend = LocalSubprocessSandboxBackend(
        workspace_root=tmp_path,
        command_timeout_seconds=60,
    )

    writer.write_sandbox_metadata(backend.metadata())

    sandbox = read_json(writer.path / "sandbox.json")
    assert sandbox["workspace_root"] == str(tmp_path.resolve())
    assert sandbox["filesystem_boundary"] == "workspace_root"
    assert sandbox["backend"] == "local_subprocess"
    assert sandbox["process_policy"] == "local_subprocess"
    assert sandbox["network_policy"] == "unrestricted"
    assert sandbox["credential_policy"] == "inherit_environment"
    assert sandbox["resource_limits"] == {"command_timeout_seconds": 60}
    assert sandbox["isolation"] == {
        "no_new_privileges": False,
        "cap_drop": [],
        "read_only_rootfs": False,
        "user": "host",
        "privileged": False,
    }
    assert sandbox["availability"] == {
        "available": False,
        "degraded": True,
        "reason": "docker sandbox disabled",
    }


def test_orchestrator_writes_sandbox_metadata_and_closes_backend(tmp_path: Path, monkeypatch) -> None:
    task_path = tmp_path / "task.yaml"
    write_task(task_path)
    created_backends: list[FakeLifecycleSandboxBackend] = []

    def fake_create_sandbox_backend(*, settings, workspace_root, session_id, command_timeout_seconds):
        backend = FakeLifecycleSandboxBackend(workspace_root)
        created_backends.append(backend)
        return backend

    monkeypatch.setattr(
        "haagent.runtime.orchestration.orchestrator.create_sandbox_backend",
        fake_create_sandbox_backend,
    )

    result = RunOrchestrator(runs_root=tmp_path / ".runs").run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert created_backends[0].closed is True
    sandbox = read_json(result.episode_path / "sandbox.json")
    assert sandbox["backend"] == "fake_sandbox"
    assert sandbox["process_policy"] == "fake_process"
