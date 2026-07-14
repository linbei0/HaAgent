"""
src/haagent/tui/design/theme.py - TUI 语义视觉 token 与主题选择

定义状态 token、符号、中文标签和内置主题，保证颜色不是唯一状态表达。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from textual.theme import Theme

THEME_ENV_VAR = "HAAGENT_TUI_THEME"


class SemanticToken(str, Enum):
    DEFAULT = "default"
    MUTED = "muted"
    EMPHASIS = "emphasis"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    SELECTION = "selection"
    FOCUS = "focus"
    RUNNING = "running"
    CANCELLED = "cancelled"
    PENDING = "pending"
    DANGER = "danger"


@dataclass(frozen=True)
class SemanticStatus:
    token: SemanticToken
    symbol: str
    label: str

    @property
    def css_class(self) -> str:
        return f"status-{self.token.value}"


class TuiThemeMode(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    MONOCHROME = "monochrome"


@dataclass(frozen=True)
class TuiThemeChoice:
    mode: TuiThemeMode
    textual_theme: str
    css_class: str
    monochrome: bool = False


_THEME_CYCLE = (TuiThemeMode.DARK, TuiThemeMode.LIGHT, TuiThemeMode.MONOCHROME)
_THEME_LABELS = {
    TuiThemeMode.DARK: "暗色",
    TuiThemeMode.LIGHT: "浅色",
    TuiThemeMode.MONOCHROME: "单色",
}

_STATUS_MAP = {
    "default": SemanticStatus(SemanticToken.DEFAULT, "-", "普通"),
    "idle": SemanticStatus(SemanticToken.DEFAULT, "-", "空闲"),
    "muted": SemanticStatus(SemanticToken.MUTED, "-", "次要"),
    "emphasis": SemanticStatus(SemanticToken.EMPHASIS, "*", "重点"),
    "success": SemanticStatus(SemanticToken.SUCCESS, "ok", "成功"),
    "done": SemanticStatus(SemanticToken.SUCCESS, "ok", "成功"),
    "completed": SemanticStatus(SemanticToken.SUCCESS, "ok", "完成"),
    "warning": SemanticStatus(SemanticToken.WARNING, "!", "警告"),
    "waiting approval": SemanticStatus(SemanticToken.WARNING, "?", "待审批"),
    "error": SemanticStatus(SemanticToken.ERROR, "!", "错误"),
    "failed": SemanticStatus(SemanticToken.ERROR, "!", "失败"),
    "info": SemanticStatus(SemanticToken.INFO, "i", "提示"),
    "selection": SemanticStatus(SemanticToken.SELECTION, ">", "选中"),
    "focus": SemanticStatus(SemanticToken.FOCUS, "*", "焦点"),
    "running": SemanticStatus(SemanticToken.RUNNING, "...", "运行中"),
    "started": SemanticStatus(SemanticToken.RUNNING, "...", "运行中"),
    "cancelled": SemanticStatus(SemanticToken.CANCELLED, "x", "已取消"),
    "pending": SemanticStatus(SemanticToken.PENDING, "?", "待处理"),
    "pending approval": SemanticStatus(SemanticToken.WARNING, "?", "待审批"),
    "waiting input": SemanticStatus(SemanticToken.PENDING, "?", "待补充"),
    "approved": SemanticStatus(SemanticToken.SUCCESS, "ok", "已允许"),
    "denied": SemanticStatus(SemanticToken.DANGER, "!", "已拒绝"),
    "danger": SemanticStatus(SemanticToken.DANGER, "!", "危险"),
}


def semantic_tokens() -> set[SemanticToken]:
    return set(SemanticToken)


def status_semantic(status: str | None) -> SemanticStatus:
    key = (status or "default").strip().casefold()
    return _STATUS_MAP.get(key, SemanticStatus(SemanticToken.INFO, "i", status or "未知"))


def status_badge(status: str | None) -> str:
    semantic = status_semantic(status)
    return f"{semantic.symbol} {semantic.label}"


def no_color_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return "NO_COLOR" in source


def select_theme(value: str | None = None, env: Mapping[str, str] | None = None) -> TuiThemeChoice:
    source = os.environ if env is None else env
    if no_color_enabled(source):
        return TuiThemeChoice(TuiThemeMode.MONOCHROME, "haagent-monochrome", "theme-monochrome", True)
    raw = (value or source.get(THEME_ENV_VAR) or TuiThemeMode.DARK.value).strip().casefold()
    if raw in {"light", "haagent-light"}:
        return TuiThemeChoice(TuiThemeMode.LIGHT, "haagent-light", "theme-light")
    if raw in {"mono", "monochrome", "no-color", "no_color", "haagent-monochrome"}:
        return TuiThemeChoice(TuiThemeMode.MONOCHROME, "haagent-monochrome", "theme-monochrome", True)
    return TuiThemeChoice(TuiThemeMode.DARK, "haagent-dark", "theme-dark")


def next_theme(choice: TuiThemeChoice, env: Mapping[str, str] | None = None) -> TuiThemeChoice:
    if no_color_enabled(env):
        return select_theme("monochrome", env)
    index = _THEME_CYCLE.index(choice.mode) if choice.mode in _THEME_CYCLE else 0
    return select_theme(_THEME_CYCLE[(index + 1) % len(_THEME_CYCLE)].value, env)


def theme_label(choice: TuiThemeChoice) -> str:
    return _THEME_LABELS.get(choice.mode, choice.mode.value)


def textual_themes() -> tuple[Theme, Theme, Theme]:
    return (
        Theme(
            name="haagent-dark",
            primary="#7aa2f7",
            secondary="#7dcfff",
            warning="#e0af68",
            error="#f7768e",
            success="#9ece6a",
            accent="#bb9af7",
            foreground="#c0caf5",
            background="#16161e",
            surface="#24283b",
            panel="#1f2335",
            dark=True,
        ),
        Theme(
            name="haagent-light",
            primary="#2459a6",
            secondary="#0f766e",
            warning="#9a5b00",
            error="#b42318",
            success="#287233",
            accent="#7a3db8",
            foreground="#202124",
            background="#f7f7f2",
            surface="#ffffff",
            panel="#eef2f6",
            dark=False,
        ),
        Theme(
            name="haagent-monochrome",
            primary="white",
            secondary="white",
            warning="white",
            error="white",
            success="white",
            accent="white",
            foreground="white",
            background="black",
            surface="black",
            panel="black",
            dark=True,
        ),
    )
