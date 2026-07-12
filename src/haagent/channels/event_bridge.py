"""
haagent/channels/event_bridge.py - worker 线程到 asyncio 的事件桥

使用 call_soon_threadsafe 保证 RuntimeUiEvent 顺序进入 asyncio queue。
"""

from __future__ import annotations

import asyncio
from typing import Any


class ChannelEventBridge:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._closed = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._closed = False

    async def stop(self) -> None:
        self._closed = True
        if self._queue is not None:
            await self._queue.put(None)

    def emit_from_thread(self, item: Any) -> None:
        # 禁止在 callback 线程直接跑 Agent；只把事件安全投递回 loop。
        if self._closed or self._loop is None or self._queue is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self, *, timeout: float | None = None) -> Any:
        if self._queue is None:
            raise RuntimeError("ChannelEventBridge is not started")
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is None and self._closed:
            raise asyncio.QueueEmpty
        return item
