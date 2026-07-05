"""
haagent/tui/typography/wrap.py - Textual/Rich Unicode 换行适配

基于 Unicode UAX #14 候选断点和终端 cell 宽度，为 HaAgent TUI 安装更适合 CJK 的换行算法。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from rich.cells import cell_len as rich_cell_len
from rich.cells import get_character_cell_size
from uniseg.graphemecluster import grapheme_cluster_boundaries
from uniseg.linebreak import line_break_boundaries


_INSTALLED = False

_CJK_CHAR = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_RE = re.compile(f"[{_CJK_CHAR}]")
_CJK_NUMERIC_PHRASE_RE = re.compile(
    rf"([{_CJK_CHAR}]\s+\d+(?:[.,]\d+)?\s+[{_CJK_CHAR}]|"
    rf"\d+\s+[月日年月号][\s\d月日年月号]*|"
    rf"[第约近逾超]\s+\d+(?:[.,]\d+)?\s+[{_CJK_CHAR}])"
)


def is_textual_line_breaking_installed() -> bool:
    return _INSTALLED


def install_textual_line_breaking() -> None:
    """安装 Rich/Textual 当前使用的换行入口，限定在 TUI 启动路径调用。"""

    global _INSTALLED
    if _INSTALLED:
        return

    import rich._wrap as rich_wrap
    import rich.text as rich_text
    import textual._wrap as textual_wrap
    import textual.content as textual_content
    import textual.document._wrapped_document as wrapped_document

    required = [
        (rich_wrap, "divide_line"),
        (rich_text, "divide_line"),
        (textual_content, "divide_line"),
        (textual_wrap, "compute_wrap_offsets"),
        (wrapped_document, "compute_wrap_offsets"),
    ]
    for module, name in required:
        if not hasattr(module, name):
            msg = f"Textual/Rich wrap entrypoint missing: {module.__name__}.{name}"
            raise RuntimeError(msg)

    rich_wrap.divide_line = divide_uax14_line
    rich_text.divide_line = divide_uax14_line
    textual_content.divide_line = divide_uax14_line
    textual_wrap.compute_wrap_offsets = compute_uax14_wrap_offsets
    wrapped_document.compute_wrap_offsets = compute_uax14_wrap_offsets
    _INSTALLED = True


def divide_uax14_line(text: str, width: int, fold: bool = True) -> list[int]:
    return _compute_wrap_offsets(text, width, fold=fold, cell_width=rich_cell_len)


def compute_uax14_wrap_offsets(
    text: str,
    width: int,
    tab_size: int,
    fold: bool = True,
    precomputed_tab_sections: list[tuple[str, int]] | None = None,
) -> list[int]:
    del precomputed_tab_sections
    tab_size = max(1, min(tab_size, max(width, 1)))

    def cell_width(segment: str) -> int:
        total = 0
        column = 0
        for char in segment:
            if char == "\t":
                char_width = tab_size - (column % tab_size)
            else:
                char_width = get_character_cell_size(char)
            total += char_width
            column += char_width
        return total

    return _compute_wrap_offsets(text, width, fold=fold, cell_width=cell_width)


def _compute_wrap_offsets(
    text: str,
    width: int,
    *,
    fold: bool,
    cell_width: Callable[[str], int],
) -> list[int]:
    if width <= 0 or not text:
        return []

    allowed_breaks = _allowed_line_breaks(text)
    required_breaks = _required_safe_breaks(text, allowed_breaks) if fold else set()
    break_candidates = (allowed_breaks | required_breaks) if fold else _space_line_breaks(text)
    grapheme_breaks = set(grapheme_cluster_boundaries(text))
    breaks: list[int] = []
    line_start = 0
    last_break: int | None = None
    index = 1

    while index <= len(text):
        current_width = cell_width(text[line_start:index])
        if current_width <= width:
            if index in break_candidates:
                last_break = index
            index += 1
            continue

        if not fold:
            if last_break is not None and last_break > line_start:
                breaks.append(last_break)
                line_start = last_break
                last_break = None
                index = line_start + 1
                continue
            index += 1
            continue

        if last_break is not None and last_break > line_start:
            break_at = last_break
        else:
            break_at = _nearest_grapheme_before(index, line_start, grapheme_breaks)
            if break_at <= line_start:
                break_at = index

        breaks.append(break_at)
        line_start = break_at
        last_break = None
        index = line_start + 1

    return breaks


def _allowed_line_breaks(text: str) -> set[int]:
    breaks = set(line_break_boundaries(text))
    breaks.discard(0)
    breaks.discard(len(text))
    breaks.difference_update(_cjk_numeric_phrase_inner_breaks(text))
    return breaks


def _required_safe_breaks(text: str, allowed_breaks: set[int]) -> set[int]:
    if not _CJK_RE.search(text):
        return set()
    grapheme_breaks = set(grapheme_cluster_boundaries(text))
    safe_breaks: set[int] = set()
    for boundary in grapheme_breaks:
        if boundary <= 0 or boundary >= len(text) or boundary in allowed_breaks:
            continue
        before = text[boundary - 1]
        after = text[boundary]
        if _CJK_RE.fullmatch(before) and _CJK_RE.fullmatch(after):
            safe_breaks.add(boundary)
    return safe_breaks


def _cjk_numeric_phrase_inner_breaks(text: str) -> set[int]:
    blocked: set[int] = set()
    for match in _CJK_NUMERIC_PHRASE_RE.finditer(text):
        blocked.update(range(match.start() + 1, match.end()))
    return blocked


def _space_line_breaks(text: str) -> set[int]:
    return {index + 1 for index, char in enumerate(text[:-1]) if char.isspace()}


def _nearest_grapheme_before(index: int, line_start: int, grapheme_breaks: set[int]) -> int:
    for boundary in range(index - 1, line_start, -1):
        if boundary in grapheme_breaks:
            return boundary
    return line_start
