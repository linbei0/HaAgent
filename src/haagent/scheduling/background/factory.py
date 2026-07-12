"""
haagent/scheduling/background/factory.py - 按平台选择后台服务 adapter
"""

from __future__ import annotations

import sys

from haagent.app.assistant_types import BackgroundServiceStatus
from haagent.scheduling.background.base import BackgroundServiceUnsupported


class UnsupportedBackgroundAdapter:
    """不支持的平台：status 明确 unsupported，install/uninstall 抛错。"""

    def status(self) -> BackgroundServiceStatus:
        return BackgroundServiceStatus(
            state="unsupported",
            host_type="none",
            detail=f"当前平台不支持后台服务: {sys.platform}",
        )

    def install(self) -> BackgroundServiceStatus:
        raise BackgroundServiceUnsupported(f"当前平台不支持后台服务安装: {sys.platform}")

    def uninstall(self) -> BackgroundServiceStatus:
        raise BackgroundServiceUnsupported(f"当前平台不支持后台服务卸载: {sys.platform}")


def create_background_adapter():
    """返回当前平台的 BackgroundServiceAdapter 实现。"""
    platform = sys.platform
    if platform == "win32":
        from haagent.scheduling.background.windows import WindowsBackgroundAdapter

        return WindowsBackgroundAdapter()
    if platform.startswith("linux"):
        from haagent.scheduling.background.systemd import SystemdBackgroundAdapter

        return SystemdBackgroundAdapter()
    if platform == "darwin":
        from haagent.scheduling.background.launchd import LaunchdBackgroundAdapter

        return LaunchdBackgroundAdapter()
    return UnsupportedBackgroundAdapter()
