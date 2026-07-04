"""
src/haagent/runtime/sandbox/settings.py - 沙箱配置解析

定义 HaAgent 命令执行沙箱的用户级配置与校验规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SandboxBackendName = Literal["local_subprocess", "docker"]


class SandboxSettingsError(ValueError):
    """sandbox settings 损坏或字段非法时抛出。"""


@dataclass(frozen=True)
class DockerSandboxSettings:
    image: str = "haagent-sandbox:py311"
    auto_build_image: bool = True
    cpu_limit: float = 1.0
    memory_limit: str = "1g"
    pids_limit: int = 128
    network: str = "none"
    read_only_rootfs: bool = True
    tmpfs: list[str] = field(default_factory=lambda: ["/tmp:rw,noexec,nosuid,size=256m"])
    extra_readonly_mounts: list[str] = field(default_factory=list)
    extra_env_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SandboxSettings:
    enabled: bool = False
    backend: SandboxBackendName = "local_subprocess"
    fail_if_unavailable: bool = False
    docker: DockerSandboxSettings = field(default_factory=DockerSandboxSettings)


def load_sandbox_settings(raw: object | None) -> SandboxSettings:
    if raw is None:
        return SandboxSettings()
    if not isinstance(raw, dict):
        raise SandboxSettingsError("sandbox must be an object")

    enabled = _bool(raw.get("enabled", False), "sandbox.enabled")
    backend = raw.get("backend", "local_subprocess")
    if backend not in ("local_subprocess", "docker"):
        raise SandboxSettingsError("sandbox.backend must be local_subprocess or docker")
    fail_if_unavailable = _bool(
        raw.get("fail_if_unavailable", False),
        "sandbox.fail_if_unavailable",
    )
    return SandboxSettings(
        enabled=enabled,
        backend=backend,
        fail_if_unavailable=fail_if_unavailable,
        docker=_load_docker_settings(raw.get("docker")),
    )


def _load_docker_settings(raw: object | None) -> DockerSandboxSettings:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SandboxSettingsError("sandbox.docker must be an object")

    image = _non_empty_str(raw.get("image", "haagent-sandbox:py311"), "sandbox.docker.image")
    auto_build_image = _bool(raw.get("auto_build_image", True), "sandbox.docker.auto_build_image")
    cpu_limit = _positive_float(raw.get("cpu_limit", 1.0), "sandbox.docker.cpu_limit")
    memory_limit = _non_empty_str(raw.get("memory_limit", "1g"), "sandbox.docker.memory_limit")
    pids_limit = _positive_int(raw.get("pids_limit", 128), "sandbox.docker.pids_limit")
    network = _non_empty_str(raw.get("network", "none"), "sandbox.docker.network")
    if network != "none":
        raise SandboxSettingsError("sandbox.docker.network only supports none")
    read_only_rootfs = _bool(raw.get("read_only_rootfs", True), "sandbox.docker.read_only_rootfs")
    tmpfs = _string_list(raw.get("tmpfs", ["/tmp:rw,noexec,nosuid,size=256m"]), "sandbox.docker.tmpfs")
    extra_readonly_mounts = _string_list(
        raw.get("extra_readonly_mounts", []),
        "sandbox.docker.extra_readonly_mounts",
    )
    for mount in extra_readonly_mounts:
        _validate_readonly_mount(mount)
    extra_env_names = _string_list(raw.get("extra_env_names", []), "sandbox.docker.extra_env_names")
    for name in extra_env_names:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise SandboxSettingsError("sandbox.docker.extra_env_names must contain variable names")

    return DockerSandboxSettings(
        image=image,
        auto_build_image=auto_build_image,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        pids_limit=pids_limit,
        network=network,
        read_only_rootfs=read_only_rootfs,
        tmpfs=tmpfs,
        extra_readonly_mounts=extra_readonly_mounts,
        extra_env_names=extra_env_names,
    )


def _validate_readonly_mount(value: str) -> None:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise SandboxSettingsError("sandbox.docker.extra_readonly_mounts must contain absolute paths")
    lowered = value.replace("\\", "/").lower()
    forbidden_tokens = [
        "/var/run/docker.sock",
        "/.ssh",
        "/.aws",
        "/.azure",
        "/.config/gcloud",
        "/appdata/local/google/chrome",
        "/appdata/roaming/mozilla",
    ]
    if any(token in lowered for token in forbidden_tokens):
        raise SandboxSettingsError("sandbox.docker.extra_readonly_mounts contains forbidden path")


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SandboxSettingsError(f"{field_name} must be a boolean")
    return value


def _non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SandboxSettingsError(f"{field_name} must be non-empty")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise SandboxSettingsError(f"{field_name} must be positive")
    return float(value)


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SandboxSettingsError(f"{field_name} must be positive")
    return value


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SandboxSettingsError(f"{field_name} must be a list of strings")
    return list(value)
