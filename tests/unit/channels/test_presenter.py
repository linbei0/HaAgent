"""
tests/unit/channels/test_presenter.py - ChannelPresenter 事件映射测试
"""

from __future__ import annotations

from haagent.channels.presenter import (
    ChannelPresenter,
    FinalizeText,
    SendInteractionPrompt,
    SendText,
    SetTyping,
)
from haagent.runtime.events.types import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    SessionLifecycleEvent,
    ToolActivityEvent,
)


def test_delta_aggregates_to_single_finalize() -> None:
    presenter = ChannelPresenter()
    actions = []
    actions.extend(presenter.handle(AssistantDeltaEvent("s", 1, 1, "hel")))
    actions.extend(presenter.handle(AssistantDeltaEvent("s", 1, 1, "lo")))
    actions.extend(presenter.handle(AssistantMessageEvent("s", 1, 1, "hello")))
    assert not any(isinstance(a, SendText) for a in actions)
    finals = [a for a in actions if isinstance(a, FinalizeText)]
    assert len(finals) == 1
    assert finals[0].text == "hello"


def test_typing_closed_on_success_failure_cancel() -> None:
    presenter = ChannelPresenter()
    start = presenter.handle(SessionLifecycleEvent("s", 1, "turn_started", "start"))
    assert any(isinstance(a, SetTyping) and a.active for a in start)

    for state in ("turn_finished",):
        presenter2 = ChannelPresenter()
        presenter2.handle(SessionLifecycleEvent("s", 1, "turn_started", "start"))
        end = presenter2.handle(SessionLifecycleEvent("s", 1, state, "end", status="ok"))
        assert any(isinstance(a, SetTyping) and not a.active for a in end)

    presenter3 = ChannelPresenter()
    presenter3.handle(SessionLifecycleEvent("s", 1, "turn_started", "start"))
    fail = presenter3.handle(
        FailureNoticeEvent("s", 1, "failed", "model", "error", "boom", "ep/1")
    )
    assert any(isinstance(a, SetTyping) and not a.active for a in fail)


def test_tool_summary_is_limited() -> None:
    import time

    presenter = ChannelPresenter(tool_summary_silence_seconds=0.0)
    presenter.handle(SessionLifecycleEvent("s", 1, "turn_started", "start"))
    presenter._last_user_visible_at = time.monotonic() - 1.0
    actions = presenter.handle(
        ToolActivityEvent("s", 1, 1, "shell", "started", "x" * 500)
    )
    texts = [a for a in actions if isinstance(a, SendText)]
    assert texts
    assert len(texts[0].text) <= 200


def test_approval_not_resent_by_presenter() -> None:
    presenter = ChannelPresenter()
    actions = presenter.handle(
        ApprovalStateEvent("s", 1, 1, "shell", "requested", "run?", None)
    )
    assert not any(isinstance(a, SendInteractionPrompt) for a in actions)
    assert actions == []


def test_failure_shows_redacted_episode_id() -> None:
    presenter = ChannelPresenter()
    actions = presenter.handle(
        FailureNoticeEvent(
            "s",
            1,
            "failed",
            "model",
            "error",
            "network",
            r"E:\ws\.runs\episodes\ep-secret\package",
        )
    )
    texts = [a for a in actions if isinstance(a, (SendText, FinalizeText))]
    assert texts
    body = texts[0].text
    assert "ep-secret" in body or "失败" in body
    assert "secret-token" not in body
