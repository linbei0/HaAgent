"""
haagent/tui/design/utils.py - TUI 纯工具函数

提供时间、截断、敏感信息摘要和短标签格式化，避免组件层散落字符串处理逻辑。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from haagent.runtime.execution.command import redact_secret_like_text


def _to_system_local(value: datetime) -> datetime:
    return value.astimezone()


def format_local_datetime(
    value: datetime | None,
    *,
    timezone_name: str | None = None,
    include_seconds: bool = False,
) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        # 调度层承诺返回 aware UTC；这里显式失败，避免把损坏数据静默当成本地时间。
        raise ValueError("TUI 展示时间必须是 timezone-aware datetime")
    localized = (
        value.astimezone(ZoneInfo(timezone_name))
        if timezone_name is not None
        else _to_system_local(value)
    )
    pattern = "%Y-%m-%d %H:%M:%S" if include_seconds else "%Y-%m-%d %H:%M"
    return localized.strftime(pattern)


def safe_summary(value: str, limit: int) -> str:
    redacted, _ = redact_secret_like_text(value)
    normalized = " ".join(redacted.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "... [truncated]"


def workspace_label(path: Path, limit: int) -> str:
    name = path.name or str(path)
    return truncate_end(name, limit)


def short_session(session_id: str, limit: int) -> str:
    return truncate_end(session_id, limit)


def truncate_end(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def truncate_status_line(value: str, width: int) -> str:
    if width <= 0 or len(value) <= width:
        return value
    return truncate_end(value, width)
