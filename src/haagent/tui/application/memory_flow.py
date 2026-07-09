"""
src/haagent/tui/application/memory_flow.py - TUI 记忆候选审查流程

从主 App 迁出记忆模式的开关、候选加载、上下选择、详情切换和确认/拒绝逻辑。
正式记忆的确认与拒绝全部经 AssistantService 走确定性服务，前端只做导航和展示。
"""

from __future__ import annotations

from typing import Any

from haagent.memory import MemoryCandidate
from haagent.tui.memory.presenter import MemoryPanelPresenter


class MemoryFlow:
    """封装记忆候选审查模式的全部状态与交互。"""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.mode = False
        self.detail_mode = False
        self.candidates: list[MemoryCandidate] = []
        self.selected = 0
        self.error: str | None = None
        self.notice: str | None = None

    def toggle(self) -> None:
        self.mode = not self.mode
        self.detail_mode = False
        if self.mode:
            self.load_candidates()
            self._app._set_prompt_value(self._app._prompt_input(), "")
        else:
            self._app._prompt_input().focus()
        self._app._refresh()

    def enter(self) -> None:
        if not self.mode or not self.candidates:
            return
        self.detail_mode = not self.detail_mode
        self._app._refresh()

    def move(self, delta: int) -> None:
        if not self.mode or self.detail_mode or not self.candidates:
            return
        next_index = self.selected + delta
        self.selected = min(max(next_index, 0), len(self.candidates) - 1)
        self._app._refresh()

    def first(self) -> None:
        if self.mode and self.candidates and not self.detail_mode:
            self.selected = 0
            self._app._refresh()

    def last(self) -> None:
        if self.mode and self.candidates and not self.detail_mode:
            self.selected = len(self.candidates) - 1
            self._app._refresh()

    def confirm(self) -> None:
        if not self.mode or not self.candidates:
            return
        candidate = self._selected_candidate()
        try:
            self._app.service.confirm_memory_candidate(candidate.candidate_id)
        except Exception as error:
            self.notice = f"Memory confirm failed: {error}"
            self._app._append_block("Memory warning", f"Memory confirm failed: {error}")
        else:
            self.notice = f"已确认记忆候选：{candidate.candidate_id}"
            self._app._append_line(f"记忆已确认：{candidate.candidate_id}")
        self.detail_mode = False
        self.load_candidates()
        self._app._refresh()

    def reject(self) -> None:
        if not self.mode or not self.candidates:
            return
        candidate = self._selected_candidate()
        try:
            self._app.service.reject_memory_candidate(candidate.candidate_id, "rejected from TUI")
        except Exception as error:
            self.notice = f"Memory reject failed: {error}"
            self._app._append_block("Memory warning", f"Memory reject failed: {error}")
        else:
            self.notice = f"已拒绝记忆候选：{candidate.candidate_id}"
            self._app._append_line(f"记忆已拒绝：{candidate.candidate_id}")
        self.detail_mode = False
        self.load_candidates()
        self._app._refresh()

    def cancel(self) -> bool:
        """Esc 处理：先退出详情，再退出记忆模式；返回是否消费了事件。"""
        if not self.mode:
            return False
        if self.detail_mode:
            self.detail_mode = False
        else:
            self.mode = False
            self._app._prompt_input().focus()
        self._app._refresh()
        return True

    def handle_key(self, key: str) -> bool:
        if key == "up":
            self.move(-1)
            return True
        if key == "down":
            self.move(1)
            return True
        if key == "g":
            self.first()
            return True
        if key in {"G", "shift+g", "upper_g"}:
            self.last()
            return True
        return False

    def load_candidates(self, *, silent: bool = False) -> None:
        try:
            self.candidates = self._app.service.list_memory_candidates(status="pending")
            self.error = None
            if self.selected >= len(self.candidates):
                self.selected = max(0, len(self.candidates) - 1)
        except Exception as error:
            self.candidates = []
            self.error = str(error)
            if not silent:
                self._app._append_block("Memory warning", f"Memory candidates unavailable: {error}")

    def panel_text(self) -> str:
        return MemoryPanelPresenter(
            candidates=self.candidates,
            selected_index=self.selected,
            detail_mode=self.detail_mode,
            notice=self.notice,
            error=self.error,
        ).render()

    def _selected_candidate(self) -> MemoryCandidate:
        return self.candidates[self.selected]
