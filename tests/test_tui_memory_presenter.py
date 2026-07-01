from __future__ import annotations

from haagent.tui.memory_presenter import MemoryPanelPresenter
from tests.test_tui_app import _memory_candidate


def test_memory_panel_presenter_renders_selection_and_detail() -> None:
    candidates = [_memory_candidate("cand_first", "第一条"), _memory_candidate("cand_second", "第二条")]

    list_text = MemoryPanelPresenter(
        candidates=candidates,
        selected_index=1,
        detail_mode=False,
    ).render()
    detail_text = MemoryPanelPresenter(
        candidates=candidates,
        selected_index=1,
        detail_mode=True,
    ).render()

    assert "> cand_second" in list_text
    assert "candidate_id: cand_second" in detail_text
    assert "candidate_id: cand_first" not in detail_text
