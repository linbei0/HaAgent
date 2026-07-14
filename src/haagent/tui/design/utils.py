"""
haagent/tui/design/utils.py - TUI 纯工具函数

提供截断、敏感信息摘要和短标签格式化，避免组件层散落字符串处理逻辑。
"""

from __future__ import annotations

from pathlib import Path

from haagent.runtime.execution.command import redact_secret_like_text


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
