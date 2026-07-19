"""
src/haagent/runtime/sandbox/docker_backend.py - Docker 沙箱后端

管理 session 级 Docker 容器，并通过 docker exec 执行命令。
"""

from __future__ import annotations

import os
import posixpath
import shutil
import subprocess
from pathlib import Path

from haagent.runtime.execution.command import (
    CommandResult,
    ShellContract,
    build_python_utf8_environment,
    run_process,
)
from haagent.runtime.sandbox.base import SandboxAvailability, SandboxCommand, SandboxMetadata
from haagent.runtime.sandbox.docker_image import build_default_image, image_exists
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from haagent.runtime.sandbox.settings import SandboxSettings


class DockerSandboxUnavailable(RuntimeError):
    """Docker 沙箱不可用且配置要求失败时抛出。"""


def get_docker_availability(settings: SandboxSettings) -> SandboxAvailability:
    if not settings.enabled or settings.backend != "docker":
        return SandboxAvailability(available=False, degraded=True, reason="docker sandbox disabled")
    docker = shutil.which("docker")
    if docker is None:
        return SandboxAvailability(available=False, degraded=True, reason="docker CLI not found")
    result = subprocess.run(
        [docker, "info"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "docker daemon unavailable"
        return SandboxAvailability(available=False, degraded=True, reason=reason)
    return SandboxAvailability(available=True, degraded=False, reason="")


def create_docker_or_fallback(
    *,
    settings: SandboxSettings,
    workspace_root: Path,
    session_id: str,
    command_timeout_seconds: int | float,
):
    availability = get_docker_availability(settings)
    if not availability.available:
        if settings.fail_if_unavailable:
            raise DockerSandboxUnavailable(availability.reason)
        return LocalSubprocessSandboxBackend(
            workspace_root=workspace_root,
            command_timeout_seconds=command_timeout_seconds,
            degraded_reason=availability.reason,
        )
    backend = DockerSandboxBackend(
        settings=settings,
        workspace_root=workspace_root,
        session_id=session_id,
        command_timeout_seconds=command_timeout_seconds,
    )
    backend.start()
    return backend


class DockerSandboxBackend:
    def __init__(
        self,
        *,
        settings: SandboxSettings,
        workspace_root: Path,
        session_id: str,
        command_timeout_seconds: int | float,
    ) -> None:
        self._settings = settings
        self._workspace_root = workspace_root.resolve()
        self._container_workspace_root = "/workspace"
        self._session_id = session_id
        self._command_timeout_seconds = command_timeout_seconds
        self._container_name = f"haagent-sandbox-{session_id}"
        self._running = False

    def metadata(self) -> SandboxMetadata:
        docker = self._settings.docker
        return SandboxMetadata(
            workspace_root=str(self._workspace_root),
            filesystem_boundary="workspace_root",
            backend="docker",
            process_policy="docker_exec_non_root",
            network_policy=docker.network,
            credential_policy="minimal_env",
            resource_limits={
                "command_timeout_seconds": self._command_timeout_seconds,
                "cpu_limit": docker.cpu_limit,
                "memory_limit": docker.memory_limit,
                "pids_limit": docker.pids_limit,
                "tmpfs": docker.tmpfs,
            },
            isolation={
                "no_new_privileges": True,
                "cap_drop": ["ALL"],
                "read_only_rootfs": docker.read_only_rootfs,
                "user": "haagent",
                "privileged": False,
            },
            availability=SandboxAvailability(available=True, degraded=False, reason=""),
        )

    def build_run_argv(self) -> list[str]:
        docker = shutil.which("docker") or "docker"
        cfg = self._settings.docker
        workspace = str(self._workspace_root)
        container_workspace = self._container_workspace_root
        argv = [
            docker,
            "run",
            "-d",
            "--rm",
            "--name",
            self._container_name,
            "--network",
            "none",
            "--cpus",
            str(cfg.cpu_limit),
            "--memory",
            cfg.memory_limit,
            "--pids-limit",
            str(cfg.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
        ]
        if cfg.read_only_rootfs:
            argv.append("--read-only")
        for tmpfs in cfg.tmpfs:
            argv.extend(["--tmpfs", tmpfs])
        argv.extend(["--mount", f"type=bind,source={workspace},target={container_workspace}", "-w", container_workspace])
        for mount in cfg.extra_readonly_mounts:
            resolved = str(Path(mount).expanduser().resolve())
            argv.extend(["--mount", f"type=bind,source={resolved},target={resolved},readonly"])
        for name in cfg.extra_env_names:
            if name in os.environ:
                argv.extend(["-e", f"{name}={os.environ[name]}"])
        argv.extend([cfg.image, "tail", "-f", "/dev/null"])
        return argv

    def build_exec_argv(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> list[str]:
        docker = shutil.which("docker") or "docker"
        command = [docker, "exec", "-w", self._container_path(cwd)]
        for key, value in (env or {}).items():
            command.extend(["-e", f"{key}={value}"])
        command.append(self._container_name)
        command.extend(argv)
        return command

    def start(self) -> None:
        cfg = self._settings.docker
        if not image_exists(cfg.image):
            if not cfg.auto_build_image or not build_default_image(cfg.image):
                raise DockerSandboxUnavailable(f"docker image unavailable: {cfg.image}")
        result = subprocess.run(
            self.build_run_argv(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            reason = result.stderr.strip() or result.stdout.strip() or "failed to start docker sandbox"
            raise DockerSandboxUnavailable(reason)
        self._running = True

    def run_shell(self, command: SandboxCommand) -> CommandResult:
        result = run_process(
            command=command.command,
            popen_args=self.build_exec_argv(
                ["bash", "-lc", command.command],
                cwd=command.cwd,
                env=command.env,
            ),
            shell=False,
            cwd=self._workspace_root,
            timeout_seconds=command.timeout_seconds,
            cancellation_token=command.cancellation_token,
        )
        return self._host_visible_result(result)

    def shell_contract(self) -> ShellContract:
        """Docker shell 固定由容器内 bash 解释。"""
        return ShellContract("posix", "bash", "linux")

    def run_python(self, script_path: Path, command: SandboxCommand) -> CommandResult:
        container_script_path = self._container_path(script_path)
        python_env = build_python_utf8_environment(command.env, inherit=False)
        result = run_process(
            command=f"python -X utf8 {script_path}",
            popen_args=self.build_exec_argv(
                ["python", "-X", "utf8", container_script_path],
                cwd=command.cwd,
                env=python_env,
            ),
            shell=False,
            cwd=self._workspace_root,
            timeout_seconds=command.timeout_seconds,
            cancellation_token=command.cancellation_token,
        )
        return self._host_visible_result(result)

    def _container_path(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self._workspace_root)
        except ValueError:
            return str(resolved)
        if str(relative) == ".":
            return self._container_workspace_root
        return posixpath.join(self._container_workspace_root, *relative.parts)

    def _host_visible_result(self, result: CommandResult) -> CommandResult:
        return CommandResult(
            command=result.command,
            status=result.status,
            exit_code=result.exit_code,
            stdout=self._host_visible_text(result.stdout),
            stderr=self._host_visible_text(result.stderr),
            stdout_excerpt=self._host_visible_text(result.stdout_excerpt),
            stderr_excerpt=self._host_visible_text(result.stderr_excerpt),
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            truncated=result.truncated,
            timeout=result.timeout,
            redacted=result.redacted,
            duration_seconds=result.duration_seconds,
            timeout_seconds=result.timeout_seconds,
        )

    def _host_visible_text(self, text: str) -> str:
        host_root = str(self._workspace_root)
        pieces = text.split(self._container_workspace_root)
        if len(pieces) == 1:
            return text
        mapped = [pieces[0]]
        for piece in pieces[1:]:
            if piece.startswith("/"):
                relative, separator, rest = piece[1:].partition("\n")
                mapped.append(str(self._workspace_root / Path(*relative.split("/"))))
                mapped.append(separator)
                mapped.append(rest)
            else:
                mapped.append(host_root)
                mapped.append(piece)
        return "".join(mapped)

    def close(self) -> None:
        if not self._running:
            return
        subprocess.run(
            [shutil.which("docker") or "docker", "stop", self._container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self._running = False
