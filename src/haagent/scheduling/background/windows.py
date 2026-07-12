"""
haagent/scheduling/background/windows.py - Windows Task Scheduler 后台 adapter

通过 schtasks.exe 参数数组安装登录触发的用户级 schedule-worker。
"""

from __future__ import annotations

import getpass
import subprocess
import sys

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.scheduling.background.base import (
    BackgroundServiceError,
    bounded_detail,
    worker_command_args,
)

TASK_NAME = "HaAgentScheduler"


class WindowsBackgroundAdapter:
    """Windows 任务计划程序：ONLOGON、当前用户、幂等安装。"""

    def __init__(self, *, task_name: str = TASK_NAME) -> None:
        self._task_name = task_name

    def status(self) -> BackgroundServiceStatus:
        result = self._run(["schtasks.exe", "/Query", "/TN", self._task_name, "/FO", "LIST"])
        if result.returncode != 0:
            combined = f"{result.stderr or ''} {result.stdout or ''}".lower()
            # 权限/访问拒绝不得伪装成 not_installed
            access_denied = any(
                token in combined
                for token in (
                    "access is denied",
                    "access denied",
                    "拒绝访问",
                    "denied",
                )
            )
            not_found = any(
                token in combined
                for token in (
                    "cannot find",
                    "not found",
                    "系统找不到",
                    "does not exist",
                    "找不到",
                )
            )
            if access_denied and not not_found:
                return BackgroundServiceStatus(
                    state="error",
                    host_type="windows_task_scheduler",
                    detail=bounded_detail(result.stderr or result.stdout or "access denied"),
                    executable=sys.executable,
                )
            return BackgroundServiceStatus(
                state="not_installed",
                host_type="windows_task_scheduler",
                detail=bounded_detail(result.stderr or result.stdout or "not found"),
                executable=sys.executable,
            )
        detail = bounded_detail(result.stdout or "installed")
        # 中英文状态列：Running / 正在运行
        stdout = result.stdout or ""
        running_markers = ("Running", "正在运行", "running")
        state = (
            "running"
            if any(marker in stdout for marker in running_markers)
            else "stopped"
        )
        return BackgroundServiceStatus(
            state=state if state in {"running", "stopped"} else "installed",
            host_type="windows_task_scheduler",
            detail=detail,
            executable=sys.executable,
        )

    def install(self) -> BackgroundServiceStatus:
        user = getpass.getuser()
        tr = self._build_tr()
        # 幂等：已存在则先删除再创建
        existing = self._run(["schtasks.exe", "/Query", "/TN", self._task_name])
        if existing.returncode == 0:
            self._run(["schtasks.exe", "/Delete", "/TN", self._task_name, "/F"])
        create = self._run(
            [
                "schtasks.exe",
                "/Create",
                "/TN",
                self._task_name,
                "/SC",
                "ONLOGON",
                "/RU",
                user,
                "/TR",
                tr,
                "/F",
            ]
        )
        if create.returncode != 0:
            raise BackgroundServiceError(
                bounded_detail(create.stderr or create.stdout or "schtasks create failed")
            )
        # 安装后必须能查询到；查询失败/权限错误不得伪装 installed
        st = self.status()
        if st.state == "not_installed":
            raise BackgroundServiceError(
                bounded_detail(st.detail or "安装后查询不到任务")
            )
        if st.state == "error":
            raise BackgroundServiceError(bounded_detail(st.detail or "安装后状态查询失败"))
        if st.state in {"running", "stopped"}:
            return BackgroundServiceStatus(
                state="installed" if st.state == "stopped" else st.state,
                host_type="windows_task_scheduler",
                detail=st.detail,
                executable=sys.executable,
            )
        return st

    def uninstall(self) -> BackgroundServiceStatus:
        result = self._run(["schtasks.exe", "/Delete", "/TN", self._task_name, "/F"])
        if result.returncode != 0:
            combined = f"{result.stderr or ''} {result.stdout or ''}".lower()
            # 已不存在可视为成功；其它错误必须 fail-fast
            not_found = any(
                token in combined
                for token in (
                    "cannot find",
                    "not found",
                    "系统找不到",
                    "找不到指定的文件",
                    "does not exist",
                )
            )
            if not not_found:
                raise BackgroundServiceError(
                    bounded_detail(result.stderr or result.stdout or "schtasks delete failed")
                )
        return BackgroundServiceStatus(
            state="not_installed",
            host_type="windows_task_scheduler",
            detail="uninstalled",
            executable=sys.executable,
        )

    def _build_tr(self) -> str:
        # Task Scheduler /TR 需要一条命令行；可执行文件加引号，参数数组再 join
        args = worker_command_args()
        exe = args[0]
        quoted_exe = f'"{exe}"'
        rest = " ".join(args[1:])
        return f"{quoted_exe} {rest}"

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
            raise BackgroundServiceError("schtasks.exe 不可用") from error
