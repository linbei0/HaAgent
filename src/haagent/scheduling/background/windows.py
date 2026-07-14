"""
haagent/scheduling/background/windows.py - Windows Task Scheduler 后台 adapter

通过 schtasks.exe 参数数组安装登录触发的用户级 schedule-worker。
"""

from __future__ import annotations

import getpass
import locale
import re
import subprocess
import sys

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.scheduling.background.base import (
    BackgroundServiceError,
    bounded_detail,
    worker_command_args,
)

TASK_NAME = "HaAgentScheduler"

# 未安装时不向 UI 透传 schtasks 原始错误（中文系统常为 GBK，误用 UTF-8 会乱码）
_DETAIL_NOT_INSTALLED = f"尚未安装计划任务 {TASK_NAME}（登录后自动运行 schedule-worker）"
_DETAIL_INSTALLED = f"已安装计划任务 {TASK_NAME}，登录后保持 worker 可用"
_DETAIL_RUNNING = f"计划任务 {TASK_NAME} 正在运行"
_DETAIL_ACCESS_DENIED = "查询任务计划失败：权限不足（拒绝访问）"


def _console_encoding() -> str:
    """schtasks 输出跟随系统 ANSI/OEM 代码页，不能强制 utf-8。"""
    if sys.platform == "win32":
        preferred = locale.getpreferredencoding(False) or ""
        if preferred and preferred.lower() not in {"utf-8", "utf8", "ascii"}:
            return preferred
        return "gbk"
    return "utf-8"


def _normalize_console_text(text: str) -> str:
    """去掉控制符与异常替换字符，避免 UI 显示乱码菱形。"""
    cleaned = (text or "").replace("\x00", " ")
    # UTF-8 误解码残留的 U+FFFD
    cleaned = cleaned.replace("\ufffd", "")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", cleaned)
    return " ".join(cleaned.split())


class WindowsBackgroundAdapter:
    """Windows 任务计划程序：ONLOGON、当前用户、幂等安装。"""

    def status(self) -> BackgroundServiceStatus:
        result = self._run(["schtasks.exe", "/Query", "/TN", TASK_NAME, "/FO", "LIST"])
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
            if access_denied:
                raw = _normalize_console_text(result.stderr or result.stdout or "")
                return BackgroundServiceStatus(
                    state="error",
                    host_type="windows_task_scheduler",
                    detail=bounded_detail(raw or _DETAIL_ACCESS_DENIED),
                    executable=sys.executable,
                )
            if not_found:
                return BackgroundServiceStatus(
                    state="not_installed",
                    host_type="windows_task_scheduler",
                    detail=_DETAIL_NOT_INSTALLED,
                    executable=sys.executable,
                )
            raw = _normalize_console_text(result.stderr or result.stdout or "")
            return BackgroundServiceStatus(
                state="error",
                host_type="windows_task_scheduler",
                detail=bounded_detail(raw or "查询计划任务失败"),
                executable=sys.executable,
            )
        # 中英文状态列：Running / 正在运行
        stdout = result.stdout or ""
        running_markers = ("Running", "正在运行", "running")
        is_running = any(marker in stdout for marker in running_markers)
        state = "running" if is_running else "stopped"
        return BackgroundServiceStatus(
            state=state,
            host_type="windows_task_scheduler",
            detail=_DETAIL_RUNNING if is_running else _DETAIL_INSTALLED,
            executable=sys.executable,
        )

    def install(self) -> BackgroundServiceStatus:
        user = getpass.getuser()
        tr = self._build_tr()
        # 幂等：已存在则先删除再创建
        existing = self._run(["schtasks.exe", "/Query", "/TN", TASK_NAME])
        if existing.returncode == 0:
            self._run(["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"])
        create = self._run(
            [
                "schtasks.exe",
                "/Create",
                "/TN",
                TASK_NAME,
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
            raw = _normalize_console_text(create.stderr or create.stdout or "")
            raise BackgroundServiceError(bounded_detail(raw or "创建计划任务失败"))
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
                detail=st.detail or _DETAIL_INSTALLED,
                executable=sys.executable,
            )
        return st

    def uninstall(self) -> BackgroundServiceStatus:
        result = self._run(["schtasks.exe", "/Delete", "/TN", TASK_NAME, "/F"])
        if result.returncode != 0:
            combined = f"{result.stderr or ''} {result.stdout or ''}".lower()
            # 已不存在可视为成功；其它错误必须 fail-fast
            access_denied = any(
                token in combined
                for token in ("access is denied", "access denied", "拒绝访问", "denied")
            )
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
            if access_denied or not not_found:
                raw = _normalize_console_text(result.stderr or result.stdout or "")
                raise BackgroundServiceError(
                    bounded_detail(raw or "删除计划任务失败")
                )
        return BackgroundServiceStatus(
            state="not_installed",
            host_type="windows_task_scheduler",
            detail=_DETAIL_NOT_INSTALLED,
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
            # 中文 Windows 下 schtasks 默认系统代码页；强制 utf-8 会把“错误:”等变成乱码
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding=_console_encoding(),
                errors="replace",
                check=False,
            )
            return subprocess.CompletedProcess(
                args=list(args),
                returncode=result.returncode,
                stdout=_normalize_console_text(result.stdout or ""),
                stderr=_normalize_console_text(result.stderr or ""),
            )
        except FileNotFoundError as error:
            raise BackgroundServiceError("schtasks.exe 不可用") from error
