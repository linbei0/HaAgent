"""
haagent/tui/memory/presenter.py - 记忆面板 Presenter

把记忆候选面板的展示输入收束成一个可测试 Interface。
"""

from __future__ import annotations

from dataclasses import dataclass

from haagent.memory import MemoryCandidate
from haagent.tui.design.renderers import memory_panel_text


@dataclass(frozen=True)
class MemoryPanelPresenter:
    candidates: list[MemoryCandidate]
    selected_index: int
    detail_mode: bool
    notice: str | None = None
    error: str | None = None

    def render(self) -> str:
        return memory_panel_text(
            candidates=self.candidates,
            selected_index=self.selected_index,
            detail_mode=self.detail_mode,
            notice=self.notice,
            error=self.error,
        )
