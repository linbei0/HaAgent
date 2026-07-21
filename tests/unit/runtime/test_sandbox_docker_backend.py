"""
tests/unit/runtime/test_sandbox_docker_backend.py - Docker 沙箱后端测试

验证 Docker 可用性、容器 hardening argv、metadata 和降级策略。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from haagent.runtime.sandbox.docker_backend import (
    DockerSandboxBackend,
    DockerSandboxUnavailable,
    create_docker_or_fallback,
    get_docker_availability,
)
from haagent.runtime.execution.command import CommandResult
from haagent.runtime.sandbox.base import SandboxCommand
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from haagent.runtime.sandbox.manager import create_sandbox_backend
from haagent.runtime.sandbox.settings import DockerSandboxSettings, SandboxSettings


def _settings(
    *,
    fail_if_unavailable: bool = False,
    extra_env_names: list[str] | None = None,
) -> SandboxSettings:
    return SandboxSettings(
        enabled=True,
        backend="docker",
        fail_if_unavailable=fail_if_unavailable,
        docker=DockerSandboxSettings(
            image="haagent-sandbox:py311",
            auto_build_image=False,
            cpu_limit=1.0,
            memory_limit="1g",
            pids_limit=128,
            network="none",
            read_only_rootfs=True,
            tmpfs=["/tmp:rw,noexec,nosuid,size=256m"],
            extra_readonly_mounts=[],
            extra_env_names=extra_env_names or [],
        ),
    )


def test_docker_availability_reports_missing_cli(monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: None)

    availability = get_docker_availability(_settings())

    assert availability.available is False
    assert availability.degraded is True
    assert "docker CLI not found" in availability.reason


def test_docker_availability_reports_daemon_failure(monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="daemon unavailable")

    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.subprocess.run", fake_run)

    availability = get_docker_availability(_settings())

    assert availability.available is False
    assert availability.degraded is True
    assert "daemon unavailable" in availability.reason


def test_docker_availability_reports_available(monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.subprocess.run", fake_run)

    availability = get_docker_availability(_settings())

    assert availability.available is True
    assert availability.degraded is False
    assert availability.reason == ""


def test_docker_run_argv_is_hardened(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    argv = backend.build_run_argv()

    assert argv[:3] == ["docker", "run", "-d"]
    assert "--rm" in argv
    assert "--name" in argv
    assert "haagent-sandbox-abc123" in argv
    assert "--network" in argv
    assert "none" in argv
    assert "--cpus" in argv
    assert "1.0" in argv
    assert "--memory" in argv
    assert "1g" in argv
    assert "--pids-limit" in argv
    assert "128" in argv
    assert "--security-opt" in argv
    assert "no-new-privileges" in argv
    assert "--cap-drop" in argv
    assert "ALL" in argv
    assert "--read-only" in argv
    assert "--tmpfs" in argv
    assert "/tmp:rw,noexec,nosuid,size=256m" in argv
    assert "--mount" in argv
    assert f"type=bind,source={tmp_path.resolve()},target=/workspace" in argv
    assert "-w" in argv
    assert "/workspace" in argv
    assert "--privileged" not in argv


def test_docker_run_argv_only_injects_allowed_env_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/uv")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    backend = DockerSandboxBackend(
        settings=_settings(extra_env_names=["UV_CACHE_DIR"]),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    argv = backend.build_run_argv()

    assert "UV_CACHE_DIR=/tmp/uv" in argv
    assert all("OPENAI_API_KEY" not in item for item in argv)
    assert all("secret" not in item for item in argv)


def test_docker_exec_argv_keeps_workdir_and_env_names(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    argv = backend.build_exec_argv(["python", "script.py"], cwd=tmp_path, env={"UV_CACHE_DIR": "/tmp/uv"})

    assert argv[:2] == ["docker", "exec"]
    assert "-w" in argv
    assert "/workspace" in argv
    assert "-e" in argv
    assert "UV_CACHE_DIR=/tmp/uv" in argv
    assert "haagent-sandbox-abc123" in argv
    assert argv[-2:] == ["python", "script.py"]


def test_docker_run_python_forces_utf8_mode_and_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )
    script_path = tmp_path / "script.py"
    script_path.write_text("print('中文')\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run_process(**kwargs):
        captured.update(kwargs)
        return CommandResult(
            command="python script.py",
            status="success",
            exit_code=0,
            stdout="",
            stderr="",
            stdout_excerpt="",
            stderr_excerpt="",
            stdout_truncated=False,
            stderr_truncated=False,
            truncated=False,
            timeout=False,
            redacted=False,
            duration_seconds=0.01,
            timeout_seconds=60,
        )

    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.run_process", fake_run_process)

    backend.run_python(
        script_path,
        SandboxCommand(
            command="python script.py",
            cwd=tmp_path,
            timeout_seconds=60,
            env={"CUSTOM_ENV": "1"},
        ),
    )

    argv = captured["popen_args"]
    assert isinstance(argv, list)
    assert "CUSTOM_ENV=1" in argv
    assert "PYTHONUTF8=1" in argv
    assert "PYTHONIOENCODING=utf-8" in argv
    assert argv[-4:] == ["python", "-X", "utf8", "/workspace/script.py"]


def test_docker_run_python_copies_host_temp_script_into_container(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )
    host_script = Path(tempfile.gettempdir()) / "haagent-code-run-outside.py"
    host_script.write_text("print('hi')\n", encoding="utf-8")
    captured: dict[str, object] = {}
    docker_calls: list[list[str]] = []

    def fake_run_process(**kwargs):
        captured.update(kwargs)
        return CommandResult(
            command="python script.py",
            status="success",
            exit_code=0,
            stdout="",
            stderr="",
            stdout_excerpt="",
            stderr_excerpt="",
            stdout_truncated=False,
            stderr_truncated=False,
            truncated=False,
            timeout=False,
            redacted=False,
            duration_seconds=0.01,
            timeout_seconds=60,
        )

    def fake_subprocess_run(argv, **kwargs):
        docker_calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.run_process", fake_run_process)
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.subprocess.run", fake_subprocess_run)

    try:
        backend.run_python(
            host_script,
            SandboxCommand(
                command="python script.py",
                cwd=tmp_path,
                timeout_seconds=60,
            ),
        )
    finally:
        host_script.unlink(missing_ok=True)

    assert any(call[:2] == ["docker", "cp"] for call in docker_calls)
    argv = captured["popen_args"]
    assert isinstance(argv, list)
    container_script = argv[-1]
    assert isinstance(container_script, str)
    assert container_script.startswith("/tmp/haagent-code-run/")
    assert argv[-4:-1] == ["python", "-X", "utf8"]


def test_docker_exec_argv_maps_workspace_child_cwd_to_container_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    child = tmp_path / "pkg"
    child.mkdir()
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    argv = backend.build_exec_argv(["pwd"], cwd=child)

    assert argv[argv.index("-w") + 1] == "/workspace/pkg"


def test_docker_backend_maps_container_stdout_paths_back_to_host(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )
    container_stdout = "/workspace\n/workspace/pkg\n"
    host_stdout = f"{tmp_path.resolve()}\n{tmp_path.resolve() / 'pkg'}\n"
    raw = CommandResult(
        command="pwd",
        status="success",
        exit_code=0,
        stdout=container_stdout,
        stderr="",
        stdout_excerpt=container_stdout,
        stderr_excerpt="",
        stdout_truncated=False,
        stderr_truncated=False,
        truncated=False,
        timeout=False,
        redacted=False,
        duration_seconds=0.01,
        timeout_seconds=60,
    )

    mapped = backend._host_visible_result(raw)

    assert mapped.stdout == host_stdout
    assert mapped.stdout_excerpt == host_stdout


def test_docker_metadata_records_limits_and_isolation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: "docker")
    backend = DockerSandboxBackend(
        settings=_settings(),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    metadata = backend.metadata()

    assert metadata.backend == "docker"
    assert metadata.process_policy == "docker_exec_non_root"
    assert metadata.network_policy == "none"
    assert metadata.credential_policy == "minimal_env"
    assert metadata.resource_limits["cpu_limit"] == 1.0
    assert metadata.resource_limits["memory_limit"] == "1g"
    assert metadata.resource_limits["pids_limit"] == 128
    assert metadata.isolation["no_new_privileges"] is True
    assert metadata.isolation["cap_drop"] == ["ALL"]
    assert metadata.isolation["read_only_rootfs"] is True
    assert metadata.isolation["user"] == "haagent"
    assert metadata.availability.available is True
    assert metadata.availability.degraded is False


def test_unavailable_docker_falls_back_when_allowed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: None)

    backend = create_docker_or_fallback(
        settings=_settings(fail_if_unavailable=False),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    assert isinstance(backend, LocalSubprocessSandboxBackend)
    assert backend.metadata().availability.degraded is True
    assert "docker CLI not found" in backend.metadata().availability.reason


def test_unavailable_docker_raises_when_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: None)

    with pytest.raises(DockerSandboxUnavailable, match="docker CLI not found"):
        create_docker_or_fallback(
            settings=_settings(fail_if_unavailable=True),
            workspace_root=tmp_path,
            session_id="abc123",
            command_timeout_seconds=60,
        )


def test_manager_uses_docker_fallback_for_docker_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("haagent.runtime.sandbox.docker_backend.shutil.which", lambda name: None)

    backend = create_sandbox_backend(
        settings=_settings(fail_if_unavailable=False),
        workspace_root=tmp_path,
        session_id="abc123",
        command_timeout_seconds=60,
    )

    assert isinstance(backend, LocalSubprocessSandboxBackend)
    assert "docker CLI not found" in backend.metadata().availability.reason
