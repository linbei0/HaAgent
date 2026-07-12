"""
tests/unit/channels/test_event_bridge.py - 线程到 asyncio 事件桥测试
"""

from __future__ import annotations

import asyncio
import threading


def test_worker_events_preserve_order() -> None:
    from haagent.channels.event_bridge import ChannelEventBridge

    async def run() -> None:
        bridge = ChannelEventBridge()
        await bridge.start()
        received: list[int] = []

        def worker() -> None:
            for i in range(20):
                bridge.emit_from_thread(i)

        thread = threading.Thread(target=worker)
        thread.start()
        while len(received) < 20:
            item = await bridge.get(timeout=2.0)
            received.append(item)
        thread.join(timeout=2)
        await bridge.stop()
        assert received == list(range(20))

    asyncio.run(run())
