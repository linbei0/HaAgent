"""
haagent/scheduling/background/launchd.py - macOS launchd agent adapter

写入 ~/Library/LaunchAgents plist；RunAtLoad + KeepAlive(异常退出恢复)。
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.scheduling.background.base import (
    BackgroundServiceError,
    bounded_detail,
    worker_command_args,
)

LABEL = "io.haagent.scheduler"
PLIST_NAME = "io.haagent.scheduler.plist"


class LaunchdBackgroundAdapter:
    """macOS 用户 LaunchAgent。"""

    def __init__(self, *, agents_dir: Path | None = None) -> None:
        if agents_dir is None:
            agents_dir = Path.home() / "Library" / "LaunchAgents"
        self._agents_dir = Path(agents_dir)
        self._plist_path = self._agents_dir / PLIST_NAME

    def status(self) -> BackgroundServiceStatus:
        if not self._plist_path.exists():
            return BackgroundServiceStatus(
                state="not_installed",
                host_type="launchd",
                detail="plist missing",
                executable=sys.executable,
            )
        # 用 launchctl print/list 区分 running/stopped/error
        domain = self._gui_domain()
        printed = self._run(["launchctl", "print", f"{domain}/{LABEL}"])
        if printed.returncode == 0:
            out = printed.stdout or ""
            state = "running" if "state = running" in out.lower() or "pid =" in out.lower() else "stopped"
            return BackgroundServiceStatus(
                state=state,
                host_type="launchd",
                detail=bounded_detail(out or str(self._plist_path)),
                executable=sys.executable,
            )
        listed = self._run(["launchctl", "list", LABEL])
        if listed.returncode == 0:
            return BackgroundServiceStatus(
                state="stopped",
                host_type="launchd",
                detail=bounded_detail(listed.stdout or "loaded"),
                executable=sys.executable,
            )
        return BackgroundServiceStatus(
            state="installed",
            host_type="launchd",
            detail=bounded_detail(str(self._plist_path)),
            executable=sys.executable,
        )

    def install(self) -> BackgroundServiceStatus:
        self._agents_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "Label": LABEL,
            "ProgramArguments": worker_command_args(),
            "RunAtLoad": True,
            # 仅在异常退出时恢复
            "KeepAlive": {"SuccessfulExit": False},
        }
        tmp = self._plist_path.with_suffix(self._plist_path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            plistlib.dump(payload, fh)
        os.replace(tmp, self._plist_path)

        # 优先 bootstrap，失败再 load（兼容旧 launchctl）
        domain = self._gui_domain()
        load = self._run(["launchctl", "bootstrap", domain, str(self._plist_path)])
        if load.returncode != 0:
            load = self._run(["launchctl", "load", str(self._plist_path)])
            if load.returncode != 0:
                detail = bounded_detail(load.stderr or load.stdout or "launchctl load failed")
                # 失败边界：禁止把 launchctl 不可用/无权限伪造成 installed
                raise BackgroundServiceError(detail)
        return self.status()

    def uninstall(self) -> BackgroundServiceStatus:
        # bootout 优先；失败再 unload 兼容旧 launchctl；均失败时再探测是否未加载
        domain = self._gui_domain()
        bootout = self._run(["launchctl", "bootout", domain, str(self._plist_path)])
        if bootout.returncode != 0:
            unload = self._run(["launchctl", "unload", str(self._plist_path)])
            if unload.returncode != 0:
                printed = self._run(["launchctl", "print", f"{domain}/{LABEL}"])
                if printed.returncode == 0:
                    raise BackgroundServiceError(
                        bounded_detail(
                            bootout.stderr
                            or unload.stderr
                            or bootout.stdout
                            or unload.stdout
                            or "launchctl uninstall failed: service still loaded"
                        )
                    )
                # print 非零：仅明确 not-found 可继续；access denied 必须 fail-fast
                if not self._is_launchctl_not_loaded(printed):
                    raise BackgroundServiceError(
                        bounded_detail(
                            printed.stderr
                            or printed.stdout
                            or bootout.stderr
                            or unload.stderr
                            or "launchctl uninstall failed: cannot confirm service stopped"
                        )
                    )
        if self._plist_path.exists():
            try:
                self._plist_path.unlink()
            except OSError as error:
                raise BackgroundServiceError(
                    bounded_detail(f"无法删除 plist: {error}")
                ) from error
        return BackgroundServiceStatus(
            state="not_installed",
            host_type="launchd",
            detail="uninstalled",
            executable=sys.executable,
        )

    @staticmethod
    def _is_launchctl_not_loaded(result: subprocess.CompletedProcess[str]) -> bool:
        """print/list 失败时，仅当输出明确表示 service not found 才视为已卸载。"""
        text = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
        # 窄匹配：禁止宽泛 "not found" 吞掉权限错误文案
        not_found_markers = (
            "could not find service",
            "service not found",
            "no such service",
            "could not find domain",
        )
        error_markers = (
            "access denied",
            "permission denied",
            "operation not permitted",
            "not privileged",
            "connection refused",
            "could not connect",
            "bootstrap failed",
            "input/output error",
            "i/o error",
        )
        if any(m in text for m in error_markers):
            return False
        if any(m in text for m in not_found_markers):
            return True
        return False

    def _gui_domain(self) -> str:
        # os.getuid 仅 POSIX；Windows 上跑单元测试时回退 0
        getuid = getattr(os, "getuid", None)
        uid = int(getuid()) if callable(getuid) else 0
        return f"gui/{uid}"

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as error:
            # 失败边界：launchctl 不存在不得伪装成功
            raise BackgroundServiceError("launchctl 不可用") from error
