from __future__ import annotations

from haagent.runtime.chat_turn import ChatEventMapper


def test_chat_event_mapper_converts_tool_finished_without_tui_or_session() -> None:
    event = ChatEventMapper.to_chat_event(
        {
            "event_type": "tool_finished",
            "turn": 2,
            "tool_name": "file_read",
            "result": {"status": "success", "path": "README.md", "content": "hello"},
        },
        turn_index=1,
    )

    assert event.event_type == "tool_finished"
    assert event.message == "finished tool file_read"
    assert event.payload["model_turn"] == 2
    assert event.payload["tool_name"] == "file_read"
    assert event.payload["status"] == "success"
    assert event.payload["result_summary"]["path"] == "README.md"


def test_chat_event_mapper_converts_assistant_delta_without_transcript_bloat() -> None:
    event = ChatEventMapper.to_chat_event(
        {
            "event_type": "assistant_delta",
            "turn": 1,
            "delta": "半句实时输出",
        },
        turn_index=1,
    )

    assert event.event_type == "assistant_delta"
    assert event.message == "半句实时输出"
    assert event.payload == {
        "model_turn": 1,
        "delta": "半句实时输出",
    }
