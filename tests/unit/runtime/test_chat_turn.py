from __future__ import annotations

from haagent.runtime.session.turn import runtime_event_message, runtime_event_payload


def test_runtime_event_helpers_compact_tool_finished_without_tui_or_session() -> None:
    payload = {
        "turn": 2,
        "tool_name": "file_read",
        "result": {"status": "success", "path": "README.md", "content": "hello"},
    }

    assert runtime_event_message("tool_finished", payload) == "finished tool file_read"
    compact_payload = runtime_event_payload("tool_finished", payload)
    assert compact_payload["model_turn"] == 2
    assert compact_payload["tool_name"] == "file_read"
    assert compact_payload["status"] == "success"
    assert compact_payload["result_summary"]["path"] == "README.md"


def test_runtime_event_helpers_compact_assistant_delta_without_transcript_bloat() -> None:
    payload = {"turn": 1, "delta": "半句实时输出"}

    assert runtime_event_message("assistant_delta", payload) == "半句实时输出"
    assert runtime_event_payload("assistant_delta", payload) == {
        "model_turn": 1,
        "delta": "半句实时输出",
    }
