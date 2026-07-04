"""
src/haagent/runtime/sandbox/manager.py - 沙箱后端创建器

根据 runtime sandbox 配置创建当前可用后端，并处理 Docker 不可用降级。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.sandbox.base import SandboxBackend
from haagent.runtime.sandbox.docker_backend import create_docker_or_fallback
from haagent.runtime.sandbox.local import LocalSubprocessSandboxBackend
from haagent.runtime.sandbox.settings import SandboxSettings


def create_sandbox_backend(
    *,
    settings: SandboxSettings,
    workspace_root: Path,
    session_id: str,
    command_timeout_seconds: int | float,
) -> SandboxBackend:
    if settings.enabled and settings.backend == "docker":
        return create_docker_or_fallback(
            settings=settings,
            workspace_root=workspace_root,
            session_id=session_id,
            command_timeout_seconds=command_timeout_seconds,
        )
    return LocalSubprocessSandboxBackend(
        workspace_root=workspace_root,
        command_timeout_seconds=command_timeout_seconds,
    )
