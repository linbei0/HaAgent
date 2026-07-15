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


def textual_themes() -> tuple[Theme, Theme, Theme]:
    return (
        Theme(
            name="haagent-dark",
            primary="#6f93a6",
            secondary="#78a89e",
            warning="#c49a55",
            error="#c86f6f",
            success="#78a17d",
            accent="#9184aa",
            foreground="#d7dce2",
            background="#111315",
            surface="#181b1f",
            panel="#20242a",
            dark=True,
        ),
        Theme(
            name="haagent-light",
            primary="#456a7d",
            secondary="#4f7d73",
            warning="#8a6428",
            error="#9e4d4d",
            success="#52785a",
            accent="#695f7c",
            foreground="#25292f",
            background="#f5f6f7",
            surface="#ffffff",
            panel="#eaedf0",
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
