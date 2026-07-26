"""
src/haagent/runtime/execution/steering.py - 运行中引导通道

提供 AgentSession 与 RunOrchestrator 共享的线程安全引导消息通道：
用户在任务运行期间提交的引导文本，由 turn loop 在下一个安全边界
（模型调用前）取出并注入对话上下文。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class SteeringChannel:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: list[str] = field(default_factory=list)

    def post(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        with self._lock:
            self._pending.append(normalized)

    def drain(self) -> list[str]:
        with self._lock:
            drained = list(self._pending)
            self._pending.clear()
        return drained

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)
