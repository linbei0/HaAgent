"""
tests/unit/tui/test_unicode_wrap.py - TUI Unicode 换行规则测试

验证 HaAgent TUI 的底层 Rich/Textual 换行适配不会破坏原文，同时改善 CJK 与数字短语断行。
"""

from __future__ import annotations

import rich._wrap
import rich.text
import textual._wrap
import textual.content
import textual.document._wrapped_document

from haagent.tui.typography.wrap import (
    compute_uax14_wrap_offsets,
    divide_uax14_line,
    install_textual_line_breaking,
)


def _split_at_offsets(text: str, offsets: list[int]) -> list[str]:
    starts = [0, *offsets]
    ends = [*offsets, len(text)]
    return [text[start:end] for start, end in zip(starts, ends)]


def test_continuous_cjk_wraps_inside_run() -> None:
    text = "人工智能助手正在整理这段中文内容"

    offsets = divide_uax14_line(text, 12)
    lines = _split_at_offsets(text, offsets)

    assert offsets
    assert lines[0] == "人工智能助手"
    assert "".join(lines) == text


def test_cjk_numeric_phrases_keep_original_spaces_but_do_not_break_inside_phrase() -> None:
    text = "各地举行建国 250 周年庆祝活动，约 400 架无人机参与，7 月 20 日结束。"

    offsets = divide_uax14_line(text, 18)
    lines = _split_at_offsets(text, offsets)

    assert "建国 250 周年" in text
    assert "约 400 架" in text
    assert "7 月 20 日" in text
    assert all(not line.endswith(("250 ", "400 ", "7 ", "月 ", "20 ")) for line in lines)
    assert all(not line.startswith(("周年", "架", "月", "20", "日")) for line in lines[1:])
    assert "".join(lines) == text


def test_english_spacing_keeps_normal_wrap_opportunities() -> None:
    text = "OpenAI 4 model responds quickly"

    offsets = divide_uax14_line(text, 8)
    lines = _split_at_offsets(text, offsets)

    assert lines[0] == "OpenAI "
    assert "".join(lines) == text


def test_textual_offsets_preserve_emoji_and_fullwidth_text() -> None:
    text = "状态✅正常，版本ＡＢＣ正在运行"

    offsets = compute_uax14_wrap_offsets(text, 10, tab_size=4)
    lines = _split_at_offsets(text, offsets)

    assert offsets
    assert all("️" not in line for line in lines[1:])
    assert "".join(lines) == text


def test_fold_false_keeps_oversized_chunk_together() -> None:
    text = "人工智能助手"

    assert divide_uax14_line(text, 4, fold=False) == []


def test_install_textual_line_breaking_patches_rich_and_textual_entrypoints() -> None:
    install_textual_line_breaking()

    assert rich._wrap.divide_line is divide_uax14_line
    assert rich.text.divide_line is divide_uax14_line
    assert textual.content.divide_line is divide_uax14_line
    assert textual._wrap.compute_wrap_offsets is compute_uax14_wrap_offsets
    assert textual.document._wrapped_document.compute_wrap_offsets is compute_uax14_wrap_offsets
