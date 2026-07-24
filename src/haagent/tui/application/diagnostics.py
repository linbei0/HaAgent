"""
haagent/tui/application/diagnostics.py - TUI 生命周期的非敏感诊断记录

只记录可与终端或 Windows 事件日志关联的环境摘要与计数，禁止写入聊天、工具和凭据内容。
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import textual


class TuiDiagnostics:
    """把 TUI 生命周期摘要写入用户级轮转 JSONL 文件。"""

    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        max_bytes: int = 256 * 1024,
        backup_count: int = 3,
    ) -> None:
        root = config_dir if config_dir is not None else Path.home() / ".haagent"
        self.log_path = root / "logs" / "tui-diagnostics.log"
        self._max_bytes = max_bytes
        self._backup_count = backup_count

    def record_started(self) -> None:
        self._write(
            "tui_started",
            python_version=platform.python_version(),
            textual_version=textual.__version__,
            terminal_program=_environment_name("TERM_PROGRAM"),
            terminal_session=_environment_name("WT_SESSION"),
        )

    def record_stopped(self, *, active_markdown_writers: int, pending_markdown_fragments: int) -> None:
        self._write(
            "tui_stopped",
            active_markdown_writers=max(0, active_markdown_writers),
            pending_markdown_fragments=max(0, pending_markdown_fragments),
        )

    def record_unhandled_exception(self, error: BaseException) -> None:
        # 异常 message / traceback 可能拼入 prompt、路径或 provider payload，仅记录类别。
        self._write(
            "unhandled_exception",
            exception_type=type(error).__name__,
            python_version=platform.python_version(),
            textual_version=textual.__version__,
        )

    def _write(self, event: str, **fields: int | str | None) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **{key: value for key, value in fields.items() if value is not None},
        }
        handler = RotatingFileHandler(
            self.log_path,
            maxBytes=self._max_bytes,
            backupCount=self._backup_count,
            encoding="utf-8",
        )
        try:
            handler.emit(_diagnostic_record(json.dumps(record, ensure_ascii=True, separators=(",", ":"))))
        finally:
            handler.close()


def _diagnostic_record(message: str):
    """构造 RotatingFileHandler 所需的最小 LogRecord，避免全局 logging 配置。"""

    import logging

    return logging.LogRecord("haagent.tui", logging.INFO, __file__, 0, message, (), None)


def _environment_name(name: str) -> str | None:
    """环境变量只标识终端，不记录可能含路径、命令或用户内容的值。"""

    return name if os.environ.get(name) else None
