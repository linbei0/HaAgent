"""
src/haagent/runtime/sandbox/__init__.py - 沙箱 runtime 公共入口

集中导出沙箱配置、metadata 契约和后端创建函数。
"""

from haagent.runtime.sandbox.base import (
    SandboxAvailability,
    SandboxBackend,
    SandboxCommand,
    SandboxMetadata,
)
from haagent.runtime.sandbox.docker_backend import (
    DockerSandboxBackend,
    DockerSandboxUnavailable,
    create_docker_or_fallback,
    get_docker_availability,
)
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from haagent.runtime.sandbox.manager import create_sandbox_backend
from haagent.runtime.sandbox.settings import (
    DockerSandboxSettings,
    SandboxSettings,
    SandboxSettingsError,
    load_sandbox_settings,
)

__all__ = [
    "DockerSandboxSettings",
    "DockerSandboxBackend",
    "DockerSandboxUnavailable",
    "LocalSubprocessSandboxBackend",
    "SandboxAvailability",
    "SandboxBackend",
    "SandboxCommand",
    "SandboxMetadata",
    "SandboxSettings",
    "SandboxSettingsError",
    "create_docker_or_fallback",
    "create_sandbox_backend",
    "get_docker_availability",
    "load_sandbox_settings",
]
