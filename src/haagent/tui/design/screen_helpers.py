"""
haagent/tui/design/screen_helpers.py - Modal dismiss 与列表窗口共享辅助

统一 safe_dismiss（防 ScreenStackError）与可见窗口起点计算，
避免各 overlay 在方向键路径上全量重建或重复 pop 崩溃。
"""

from __future__ import annotations

from typing import Any, TypeVar

from textual.app import ScreenStackError
from textual.screen import Screen
from textual.widgets import OptionList

T = TypeVar("T")


def safe_dismiss(screen: Screen[Any], result: T | None = None) -> None:
    """仅在 screen 仍在栈顶时 dismiss；已关闭或栈错乱时静默返回。"""
    try:
        if screen.app.screen is not screen:
            return
        screen.dismiss(result)
    except ScreenStackError:
        return


def visible_window_start(*, total: int, selected: int, page_size: int) -> int:
    """计算让 selected 落在可见页内的滚动起点（与 ModelSwitch 窗口一致）。"""
    if total <= 0 or page_size <= 0:
        return 0
    selected = min(max(selected, 0), total - 1)
    if total <= page_size:
        return 0
    # 尽量让选中项靠近页底，减少频繁滚动；与 models.MODEL_SWITCH_PAGE_SIZE 算法对齐。
    return max(0, min(selected - page_size + 1, total - page_size))


def set_option_list_highlight(option_list: OptionList, index: int | None) -> None:
    """只改高亮，不调用 set_options。"""
    option_list.highlighted = index
