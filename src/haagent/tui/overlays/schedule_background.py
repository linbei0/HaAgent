"""
haagent/tui/overlays/schedule_background.py - 后台服务状态页

展示安装状态、host 类型、心跳与安装/卸载/诊断动作说明。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haagent.tui.design.utils import safe_summary

_STATE_LABELS = {
    "not_installed": "未安装",
    "stopped": "已停止",
    "running": "运行中",
    "error": "状态异常",
    "unsupported": "不支持",
}


@dataclass(frozen=True)
class ScheduleBackgroundState:
    status: Any | None = None
    host: Any | None = None

    def render(self) -> str:
        lines = [
            "后台服务",
            "系统后台 worker 在 TUI 关闭后仍可触发计划；安装会写入 OS 任务。",
            "",
        ]
        # TUI 内嵌 host：死亡/致命错误必须可见，禁止静默
        host = self.host
        if host is not None:
            running = bool(getattr(host, "running", False))
            fatal = bool(getattr(host, "fatal", False))
            last_error = str(getattr(host, "last_error", "") or "")
            owner = str(getattr(host, "owner_id", "") or "")
            host_label = "运行中" if running else "已停止"
            if fatal:
                host_label = "异常退出"
            lines.append(f"TUI Host: {host_label}")
            if owner:
                lines.append(f"Host Owner: {safe_summary(owner, 40)}")
            if last_error:
                lines.append(f"Host 错误: {safe_summary(last_error, 90)}")
            lines.append("")
        status = self.status
        if status is None:
            lines.append("（正在加载系统后台状态…）")
        else:
            state = str(getattr(status, "state", "unknown"))
            label = _STATE_LABELS.get(state, state)
            host_type = str(getattr(status, "host_type", "-"))
            detail = str(getattr(status, "detail", "") or "")
            executable = getattr(status, "executable", None)
            heartbeat = getattr(status, "last_heartbeat_utc", None)
            lines.append(f"系统后台: {label} ({state})")
            lines.append(f"Host 类型: {host_type}")
            if detail:
                lines.append(f"说明: {safe_summary(detail, 80)}")
            if executable:
                lines.append(f"可执行文件: {safe_summary(str(executable), 70)}")
            if heartbeat is not None:
                lines.append(f"最近心跳: {str(heartbeat)[:19]}")
            if state == "unsupported":
                lines.append("")
                lines.append("当前平台不支持自动安装后台服务。")
        lines.extend(
            [
                "",
                "i 安装/修复  u 卸载  d 诊断刷新  Esc 返回",
            ]
        )
        return "\n".join(lines)
