"""
haagent/scheduling/background/base.py - 后台服务 adapter 协议与状态

定义 install/uninstall/status 契约与不支持平台显式错误。
"""

from __future__ import annotations

from typing import Protocol

from haagent.app.assistant_types import BackgroundServiceStatus


class BackgroundServiceUnsupported(RuntimeError):
    """当前平台不支持用户级后台 worker 安装。"""


class BackgroundServiceError(RuntimeError):
    """后台服务安装/卸载/查询失败；消息有界、无 secret。"""


class BackgroundServiceAdapter(Protocol):
    def status(self) -> BackgroundServiceStatus: ...

    def install(self) -> BackgroundServiceStatus: ...

    def uninstall(self) -> BackgroundServiceStatus: ...


def worker_command_args() -> list[str]:
    """系统 host 启动参数数组；禁止拼 shell 字符串。"""
    import sys

    return [sys.executable, "-m", "haagent.cli", "schedule-worker"]


def bounded_detail(text: str, *, limit: int = 400) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 15] + "... [truncated]"
