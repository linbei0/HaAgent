"""
src/haagent/runtime/sandbox/local.py - 本机命令执行后端

封装当前 subprocess 行为，作为默认兼容后端与 Task 1 metadata 来源。
"""

from __future__ import annotations

import sys
from pathlib import Path

from haagent.runtime.execution.command import CommandResult, run_command, run_process
from haagent.runtime.sandbox.base import SandboxAvailability, SandboxCommand, SandboxMetadata


class LocalSubprocessSandboxBackend:
    def __init__(
        self,
        *,
        workspace_root: Path,
        command_timeout_seconds: int | float,
        degraded_reason: str = "docker sandbox disabled",
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._command_timeout_seconds = command_timeout_seconds
        self._degraded_reason = degraded_reason

    def metadata(self) -> SandboxMetadata:
        return SandboxMetadata(
            workspace_root=str(self._workspace_root),
            filesystem_boundary="workspace_root",
            backend="local_subprocess",
            process_policy="local_subprocess",
            network_policy="unrestricted",
            credential_policy="inherit_environment",
            resource_limits={"command_timeout_seconds": self._command_timeout_seconds},
            isolation={
                "no_new_privileges": False,
                "cap_drop": [],
                "read_only_rootfs": False,
                "user": "host",
                "privileged": False,
            },
            availability=SandboxAvailability(
                available=False,
                degraded=True,
                reason=self._degraded_reason,
            ),
        )

    def run_shell(self, command: SandboxCommand) -> CommandResult:
        return run_command(
            command.command,
            command.cwd,
            command.timeout_seconds,
            cancellation_token=command.cancellation_token,
        )

    def run_python(self, script_path: Path, command: SandboxCommand) -> CommandResult:
        return run_process(
            command=f"{sys.executable} {script_path}",
            popen_args=[sys.executable, str(script_path)],
            shell=False,
            cwd=command.cwd,
            timeout_seconds=command.timeout_seconds,
            cancellation_token=command.cancellation_token,
        )

    def close(self) -> None:
        return None
