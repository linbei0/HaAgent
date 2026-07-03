"""
src/haagent/runtime/execution/cancellation.py - 运行取消协议

提供 AgentSession 与 RunOrchestrator 共享的显式取消信号。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class RunCancelled(RuntimeError):
    """当前 run 收到用户取消信号时抛出。"""


@dataclass
class CancellationToken:
    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RunCancelled("user cancelled current run")
