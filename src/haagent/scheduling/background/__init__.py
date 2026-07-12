"""
haagent/scheduling/background/__init__.py - 后台服务 adapter 稳定导出
"""

from __future__ import annotations

from haagent.scheduling.background.base import (
    BackgroundServiceAdapter,
    BackgroundServiceError,
    BackgroundServiceUnsupported,
    worker_command_args,
)
from haagent.scheduling.background.factory import (
    UnsupportedBackgroundAdapter,
    create_background_adapter,
)

__all__ = [
    "BackgroundServiceAdapter",
    "BackgroundServiceError",
    "BackgroundServiceUnsupported",
    "UnsupportedBackgroundAdapter",
    "create_background_adapter",
    "worker_command_args",
]
