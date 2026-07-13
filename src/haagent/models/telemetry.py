"""
src/haagent/models/telemetry.py - 模型 transport 遥测事件

独立 DTO，避免 models 与 runtime.performance 循环依赖；只承载非敏感数字与枚举。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ModelTransportEventKind = Literal[
    "attempt_started",
    "request_prepared",
    "headers_received",
    "first_sse",
    "first_text",
    "attempt_finished",
    "attempt_failed",
]


@dataclass(frozen=True)
class ModelTransportEvent:
    """单次模型 attempt 的 transport 边界事件；禁止携带 prompt/响应正文/凭据。"""

    kind: ModelTransportEventKind
    attempt: int
    elapsed_ms: float
    request_payload_bytes: int | None = None
