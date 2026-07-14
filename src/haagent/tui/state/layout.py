"""
haagent/tui/state/layout.py - TUI 状态模型

集中定义交互等待状态和响应式布局判断，让 App 只负责协调状态变化。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse


MIN_WIDTH = 80
MIN_HEIGHT = 24


@dataclass(frozen=True)
class ResponsiveLayout:
    too_small: bool


@dataclass
class PendingInteraction:
    request: HumanInteractionRequest
    done: threading.Event = field(default_factory=threading.Event)
    response: HumanInteractionResponse | None = None


def layout_for_size(width: int, height: int) -> ResponsiveLayout:
    too_small = width < MIN_WIDTH or height < MIN_HEIGHT
    return ResponsiveLayout(too_small=too_small)
