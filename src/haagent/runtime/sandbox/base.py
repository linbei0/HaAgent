"""
src/haagent/runtime/sandbox/base.py - 沙箱后端接口

定义命令执行沙箱的基础契约，以及写入 episode 的审计 metadata。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.execution.command import CommandResult


@dataclass(frozen=True)
class SandboxAvailability:
    available: bool
    degraded: bool
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxMetadata:
    workspace_root: str
    filesystem_boundary: str
    backend: str
    process_policy: str
    network_policy: str
    credential_policy: str
    resource_limits: dict[str, object]
    isolation: dict[str, object]
    availability: SandboxAvailability

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SandboxCommand:
    command: str
    cwd: Path
    timeout_seconds: float
    cancellation_token: CancellationToken | None = None
    env: dict[str, str] = field(default_factory=dict)


class SandboxBackend(Protocol):
    def metadata(self) -> SandboxMetadata:
        """返回真实执行边界，用于写入 sandbox.json。"""

    def run_shell(self, command: SandboxCommand) -> CommandResult:
        """执行 shell 命令。"""

    def run_python(self, script_path: Path, command: SandboxCommand) -> CommandResult:
        """执行 workspace 内 Python 脚本。"""

    def close(self) -> None:
        """释放后端资源。"""
