"""
src/haagent/runtime/sandbox/status.py - 沙箱用户状态与诊断

提供面向 CLI/TUI 的沙箱状态、Docker 诊断，以及显式开启/关闭配置写入。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from haagent.models.provider_profile import user_settings_path
from haagent.runtime.sandbox.docker_image import image_exists
from haagent.runtime.sandbox.settings import DockerSandboxSettings, SandboxSettings
from haagent.runtime.settings import RuntimeSettingsError, load_runtime_settings


@dataclass(frozen=True)
class SandboxUserStatus:
    backend: str
    isolation_level: str
    network_policy: str
    credential_policy: str
    degraded: bool
    reason: str
    recommendation: str
    config_path: Path


@dataclass(frozen=True)
class SandboxDoctorReport:
    backend: str
    ready: bool
    docker_cli: str
    docker_daemon: str
    image: str
    auto_build_image: bool
    reason: str
    next_action: str


def sandbox_user_status(*, config_path: Path | None = None) -> SandboxUserStatus:
    path = config_path or user_settings_path()
    try:
        settings = load_runtime_settings(config_path=path).sandbox
    except RuntimeSettingsError as error:
        return SandboxUserStatus(
            backend="unknown",
            isolation_level="unknown",
            network_policy="unknown",
            credential_policy="unknown",
            degraded=True,
            reason=str(error),
            recommendation=f"Fix runtime settings: {path}",
            config_path=path,
        )
    if settings.enabled and settings.backend == "docker":
        return SandboxUserStatus(
            backend="docker",
            isolation_level="stronger",
            network_policy=settings.docker.network,
            credential_policy="minimal_env",
            degraded=False,
            reason="",
            recommendation="Run `haagent sandbox doctor` to verify Docker readiness.",
            config_path=path,
        )
    return SandboxUserStatus(
        backend="local_subprocess",
        isolation_level="weak",
        network_policy="unrestricted",
        credential_policy="inherit_environment",
        degraded=True,
        reason="docker sandbox disabled",
        recommendation="Run `haagent sandbox enable docker` for stronger isolation.",
        config_path=path,
    )


def sandbox_doctor_report(
    *,
    config_path: Path | None = None,
    check_disabled: bool = False,
) -> SandboxDoctorReport:
    path = config_path or user_settings_path()
    try:
        settings = load_runtime_settings(config_path=path).sandbox
    except RuntimeSettingsError as error:
        return SandboxDoctorReport(
            backend="unknown",
            ready=False,
            docker_cli="not_checked",
            docker_daemon="not_checked",
            image="not_checked",
            auto_build_image=True,
            reason=str(error),
            next_action=f"Fix runtime settings: {path}",
        )
    if (not settings.enabled or settings.backend != "docker") and not check_disabled:
        return SandboxDoctorReport(
            backend="local_subprocess",
            ready=False,
            docker_cli="not_checked",
            docker_daemon="not_checked",
            image="not_checked",
            auto_build_image=settings.docker.auto_build_image,
            reason="docker sandbox disabled",
            next_action="Run `haagent sandbox enable docker` to enable Docker isolation.",
        )
    backend = "docker" if settings.enabled and settings.backend == "docker" else "local_subprocess"
    docker = shutil.which("docker")
    if docker is None:
        return SandboxDoctorReport(
            backend=backend,
            ready=False,
            docker_cli="missing",
            docker_daemon="not_checked",
            image="not_checked",
            auto_build_image=settings.docker.auto_build_image,
            reason="docker CLI not found",
            next_action="Install Docker Desktop or Docker Engine, then run `haagent sandbox doctor` again.",
        )
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
        return SandboxDoctorReport(
            backend=backend,
            ready=False,
            docker_cli="found",
            docker_daemon="unavailable",
            image="not_checked",
            auto_build_image=settings.docker.auto_build_image,
            reason=reason,
            next_action="Start Docker Desktop or Docker daemon, then run `haagent sandbox doctor` again.",
        )
    image_status = "present" if image_exists(settings.docker.image) else "missing"
    if image_status == "missing" and not settings.docker.auto_build_image:
        return SandboxDoctorReport(
            backend=backend,
            ready=False,
            docker_cli="found",
            docker_daemon="running",
            image=image_status,
            auto_build_image=False,
            reason=f"docker image missing: {settings.docker.image}",
            next_action=f"Build image `{settings.docker.image}` or enable sandbox.docker.auto_build_image.",
        )
    if image_status == "missing":
        return SandboxDoctorReport(
            backend=backend,
            ready=True,
            docker_cli="found",
            docker_daemon="running",
            image=image_status,
            auto_build_image=True,
            reason="",
            next_action="Docker is reachable; first sandbox run will build the missing image.",
        )
    return SandboxDoctorReport(
        backend=backend,
        ready=True,
        docker_cli="found",
        docker_daemon="running",
        image=image_status,
        auto_build_image=settings.docker.auto_build_image,
        reason="",
        next_action="Docker sandbox is ready.",
    )


def enable_docker_sandbox(
    *,
    config_path: Path | None = None,
    fail_if_unavailable: bool = True,
) -> SandboxUserStatus:
    path = config_path or user_settings_path()
    current = load_runtime_settings(config_path=path)
    sandbox = SandboxSettings(
        enabled=True,
        backend="docker",
        fail_if_unavailable=fail_if_unavailable,
        docker=current.sandbox.docker,
    )
    _write_sandbox_settings(path, sandbox)
    return sandbox_user_status(config_path=path)


def disable_sandbox(*, config_path: Path | None = None) -> SandboxUserStatus:
    path = config_path or user_settings_path()
    current = load_runtime_settings(config_path=path)
    sandbox = SandboxSettings(
        enabled=False,
        backend="local_subprocess",
        fail_if_unavailable=False,
        docker=current.sandbox.docker,
    )
    _write_sandbox_settings(path, sandbox)
    return sandbox_user_status(config_path=path)


def _write_sandbox_settings(path: Path, sandbox: SandboxSettings) -> None:
    raw = _read_settings_raw(path)
    raw["sandbox"] = _sandbox_to_json(sandbox)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_settings_raw(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeSettingsError(f"settings config is invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise RuntimeSettingsError("settings config must be a JSON object")
    return raw


def _sandbox_to_json(settings: SandboxSettings) -> dict[str, object]:
    docker = _docker_to_json(settings.docker)
    return {
        "enabled": settings.enabled,
        "backend": settings.backend,
        "fail_if_unavailable": settings.fail_if_unavailable,
        "docker": docker,
    }


def _docker_to_json(settings: DockerSandboxSettings) -> dict[str, object]:
    return {
        "image": settings.image,
        "auto_build_image": settings.auto_build_image,
        "cpu_limit": settings.cpu_limit,
        "memory_limit": settings.memory_limit,
        "pids_limit": settings.pids_limit,
        "network": settings.network,
        "read_only_rootfs": settings.read_only_rootfs,
        "tmpfs": list(settings.tmpfs),
        "extra_readonly_mounts": list(settings.extra_readonly_mounts),
        "extra_env_names": list(settings.extra_env_names),
    }
