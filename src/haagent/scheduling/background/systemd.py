"""
haagent/scheduling/background/systemd.py - systemd user service adapter

写入 ~/.config/systemd/user unit，daemon-reload 后 enable；临时文件 + os.replace。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.scheduling.background.base import (
    BackgroundServiceError,
    bounded_detail,
    worker_command_args,
)

SERVICE_NAME = "haagent-scheduler.service"


class SystemdBackgroundAdapter:
    """Linux systemd --user 服务。"""

    def __init__(self) -> None:
        self._unit_dir = Path.home() / ".config" / "systemd" / "user"
        self._unit_path = self._unit_dir / SERVICE_NAME

    def status(self) -> BackgroundServiceStatus:
        if not self._unit_path.exists():
            return BackgroundServiceStatus(
                state="not_installed",
                host_type="systemd_user",
                detail="unit missing",
                executable=sys.executable,
            )
        result = self._systemctl(["--user", "is-active", SERVICE_NAME])
        active = (result.stdout or "").strip()
        if result.returncode != 0 and active not in {
            "inactive",
            "failed",
            "dead",
            "unknown",
        }:
            raise BackgroundServiceError(
                bounded_detail(result.stderr or result.stdout or "systemctl status failed")
            )
        if active == "active":
            state = "running"
        elif active in {"inactive", "failed"}:
            state = "stopped"
        else:
            state = "installed"
        return BackgroundServiceStatus(
            state=state,
            host_type="systemd_user",
            detail=bounded_detail(active or "installed"),
            executable=sys.executable,
        )

    def install(self) -> BackgroundServiceStatus:
        self._unit_dir.mkdir(parents=True, exist_ok=True)
        content = self._unit_content()
        # 临时文件 + 原子替换
        tmp = self._unit_path.with_suffix(self._unit_path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, self._unit_path)

        reload = self._systemctl(["--user", "daemon-reload"])
        if reload.returncode != 0:
            raise BackgroundServiceError(
                bounded_detail(reload.stderr or reload.stdout or "daemon-reload failed")
            )
        enable = self._systemctl(["--user", "enable", "--now", SERVICE_NAME])
        if enable.returncode != 0:
            # 部分环境无 session bus：仍保留 unit 文件
            enable2 = self._systemctl(["--user", "enable", SERVICE_NAME])
            if enable2.returncode != 0:
                raise BackgroundServiceError(
                    bounded_detail(enable.stderr or enable.stdout or "enable failed")
                )
        return self.status()

    def uninstall(self) -> BackgroundServiceStatus:
        # 停止证明：仅 disable --now 成功，或后续 is-active != active。
        # 普通 disable 成功不代表正在运行的服务已停止。
        stop = self._systemctl(["--user", "disable", "--now", SERVICE_NAME])
        if stop.returncode != 0:
            active = self._systemctl(["--user", "is-active", SERVICE_NAME])
            active_text = (active.stdout or "").strip().lower()
            if active_text == "active":
                raise BackgroundServiceError(
                    bounded_detail(
                        stop.stderr
                        or stop.stdout
                        or active.stderr
                        or "systemctl uninstall failed: service still active"
                    )
                )
            # is-active 明确 inactive/failed/dead/unknown 才可继续；空输出或权限错误 fail-fast
            if active_text not in {"inactive", "failed", "dead", "unknown"}:
                raise BackgroundServiceError(
                    bounded_detail(
                        stop.stderr
                        or active.stderr
                        or stop.stdout
                        or active.stdout
                        or "systemctl uninstall failed: cannot confirm stopped"
                    )
                )
        # disable --now 已成功时不再重复 plain disable
        if stop.returncode != 0:
            disable = self._systemctl(["--user", "disable", SERVICE_NAME])
            if disable.returncode != 0:
                raise BackgroundServiceError(
                    bounded_detail(
                        disable.stderr or disable.stdout or "systemctl disable failed"
                    )
                )
        if self._unit_path.exists():
            try:
                self._unit_path.unlink()
            except OSError as error:
                raise BackgroundServiceError(
                    bounded_detail(f"无法删除 unit: {error}")
                ) from error
        # unit 删除后 daemon-reload 失败也必须暴露，禁止静默 not_installed
        reload = self._systemctl(["--user", "daemon-reload"])
        if reload.returncode != 0:
            raise BackgroundServiceError(
                bounded_detail(reload.stderr or reload.stdout or "daemon-reload failed")
            )
        return BackgroundServiceStatus(
            state="not_installed",
            host_type="systemd_user",
            detail="uninstalled",
            executable=sys.executable,
        )

    def _unit_content(self) -> str:
        args = worker_command_args()
        # systemd ExecStart：可执行文件 + 参数；路径含空格时 quote
        exec_parts = [self._quote(a) for a in args]
        exec_start = " ".join(exec_parts)
        return (
            "[Unit]\n"
            "Description=HaAgent schedule worker\n"
            "After=default.target\n"
            "\n"
            "[Service]\n"
            "Type=simple\n"
            f"ExecStart={exec_start}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n"
            "\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )

    def _quote(self, value: str) -> str:
        # 仅在含空格或引号时加引号；不把路径反斜杠双重转义（便于跨平台单测）。
        if " " not in value and '"' not in value:
            return value
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'

    def _systemctl(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = ["systemctl", *args]
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as error:
            raise BackgroundServiceError("systemctl 不可用") from error
