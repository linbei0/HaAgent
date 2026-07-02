"""
tests/unit/tui/test_memory_presenter.py - TUI 记忆面板渲染单元测试

验证记忆候选列表和详情文本的纯渲染逻辑，不启动 Textual app。
"""

from __future__ import annotations

from haagent.memory import CandidateEvidence, MemoryCandidate
from haagent.tui.memory_presenter import MemoryPanelPresenter


def _memory_candidate(candidate_id: str = "cand_abc123", title: str = "用户身份与爱好") -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        scope="user",
        category="user_preferences",
        title=title,
        body="用户叫小明，喜欢唱跳rap篮球。",
        evidence=CandidateEvidence(
            source_type="extraction",
            evidence_summary="用户明确要求记住自己的名字和爱好。",
            session_id="session-test",
            turn_index=1,
            episode_path=".runs/episode-test",
            source_summary="用户明确要求记住自己的名字和爱好。",
            basis="用户说：我叫小明，喜欢唱跳rap篮球，记住我的爱好。",
            category_rationale="这是跨 workspace 可复用的用户偏好和身份信息。",
        ),
        source="extraction",
        status="pending",
        created_at="2026-06-26T00:00:00+00:00",
        updated_at="2026-06-26T00:00:00+00:00",
        tags=["profile"],
        risk_flags=[],
    )


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
