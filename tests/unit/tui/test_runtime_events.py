"""
tests/unit/tui/test_runtime_events.py - TUI 强类型 runtime 事件处理测试

直接使用 RuntimeUiEvent 验证 TUI adapter 的状态更新与展示动作。
"""

from __future__ import annotations

from haagent.runtime.events import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RUNTIME_UI_EVENT_TYPES,
    SessionLifecycleEvent,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.tui.application.runtime_events import RUNTIME_UI_EVENT_HANDLERS, handle_runtime_ui_event


def test_runtime_ui_event_handler_registry_covers_protocol_types() -> None:
    assert set(RUNTIME_UI_EVENT_HANDLERS) == set(RUNTIME_UI_EVENT_TYPES)


class FakeRuntimeEventApp:
    def __init__(self) -> None:
        self._state = "idle"
        self._memory_notice = None
        self._memory_mode = False
        self._memory_detail_mode = True
        self._last_failure = None
        self._sandbox_status = None
        self.assistant_deltas: list[tuple[int, str]] = []
        self.assistant_messages: list[tuple[int, str]] = []
        self.tool_activities: list[tuple[int, str, str, str]] = []
        self.tool_diagnostics: list[tuple[int, str, str]] = []
        self.blocks: list[tuple[str, str]] = []
        self.lines: list[str] = []
        self.answer_questions: list[str] = []
        self.memory_loads = 0
        self.finalized_streams = 0
        self.refreshes = 0

    def _merge_assistant_delta(self, turn_index: int, delta: str) -> None:
        self.assistant_deltas.append((turn_index, delta))

    def _finalize_assistant_message(self, turn_index: int, content: str) -> None:
        self.assistant_messages.append((turn_index, content))

    def _record_tool_activity(self, turn_index: int, tool_name: str, status: str, summary: str) -> None:
        self.tool_activities.append((turn_index, tool_name, status, summary))

    def _record_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        self.tool_diagnostics.append((turn_index, tool_name, message))

    def _append_block(self, title: str, body: str) -> None:
        self.blocks.append((title, body))

    def _append_line(self, line: str) -> None:
        self.lines.append(line)

    def _set_answer_required(self, question: str) -> None:
        self.answer_questions.append(question)

    def _load_memory_candidates(self, *, silent: bool = False) -> None:
        assert silent is True
        self.memory_loads += 1

    def _finalize_streaming_assistant_if_needed(self) -> None:
        self.finalized_streams += 1

    def _refresh(self) -> None:
        self.refreshes += 1


def test_runtime_ui_event_handler_updates_assistant_stream() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, AssistantDeltaEvent("session-1", 2, 1, "半句"))
    handle_runtime_ui_event(app, AssistantMessageEvent("session-1", 2, 1, "整句"))

    assert app.assistant_deltas == [(2, "半句")]
    assert app.assistant_messages == [(2, "整句")]
    assert app.refreshes == 2


def test_runtime_ui_event_handler_records_tool_activity() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, ToolActivityEvent("session-1", 1, 2, "file_read", "finished", "读取 README"))

    assert app.tool_activities == [(1, "file_read", "done", "读取 README")]


def test_runtime_ui_event_handler_tracks_approval_state() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        ApprovalStateEvent(
            "session-1",
            1,
            2,
            "shell",
            "requested",
            "允许运行？",
            None,
        ),
    )
    handle_runtime_ui_event(
        app,
        ApprovalStateEvent("session-1", 1, 2, "shell", "granted", "", True),
    )

    assert app.tool_activities[0] == (1, "shell", "approval", "等待审批")
    assert app.tool_activities[1] == (1, "shell", "running", "审批已允许")
    assert app.lines == []
    assert app._state == "running"


def test_runtime_ui_event_handler_keeps_approval_denials_visible() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        ApprovalStateEvent("session-1", 1, 2, "shell", "denied", "", False),
    )

    assert app.tool_activities == [(1, "shell", "failed", "审批已拒绝")]
    assert app.lines == ["审批已拒绝：shell"]


def test_runtime_ui_event_handler_routes_tool_result_compaction_to_tool_detail() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        WarningNoticeEvent(
            session_id="session-1",
            turn_index=1,
            title="Tool result compacted",
            message="web_search result compacted from 1854 to 929 chars",
            notice_kind="tool_result_microcompact",
            surface="tool_detail",
            details={"tool_name": "web_search", "original_chars": 1854, "final_chars": 929},
        ),
    )

    assert app.blocks == []
    assert app.tool_diagnostics == [(1, "web_search", "结果已压缩 1854 -> 929 字符")]


def test_runtime_ui_event_handler_hides_loop_guidance() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        WarningNoticeEvent(
            session_id="session-1",
            turn_index=1,
            title="Loop guidance",
            message="File change succeeded. Consider reading back notes.md.",
            notice_kind="loop_guidance",
            surface="hidden",
            details={"tool_name": "file_write"},
        ),
    )

    assert app.blocks == []
    assert app.lines == []
    assert app.tool_diagnostics == []


def test_runtime_ui_event_handler_tracks_user_input_state() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        UserInputStateEvent("session-1", 1, 2, "request_user_input", "requested", "你要哪个文件？"),
    )
    handle_runtime_ui_event(
        app,
        UserInputStateEvent(
            "session-1",
            1,
            2,
            "request_user_input",
            "received",
            "你要哪个文件？",
            approved=True,
        ),
    )

    assert app.answer_questions == ["你要哪个文件？"]
    assert app.blocks == [("Answer required", "你要哪个文件？")]
    assert app.lines == ["回答已提交：request_user_input"]
    assert app._state == "running"


def test_runtime_ui_event_handler_shows_failure_notice() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        FailureNoticeEvent(
            "session-1",
            1,
            "failed",
            "model_call",
            "provider",
            "HTTP 404",
            "E:/workspace/.runs/episode",
        ),
    )

    assert app._state == "failed"
    assert app._last_failure is not None
    assert app.blocks[0][0] == "Failure"
    assert "HTTP 404" in app.blocks[0][1]
    assert app.finalized_streams == 1


def test_runtime_ui_event_handler_opens_memory_notice() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, MemoryNoticeEvent("session-1", 1, "发现 1 条候选", count=1))

    assert app.blocks == [("Memory", "发现 1 条候选")]
    assert app._memory_notice == "发现 1 条候选"
    assert app._memory_mode is True
    assert app._memory_detail_mode is False
    assert app.memory_loads == 1


def test_runtime_ui_event_handler_tracks_sandbox_status_from_session_lifecycle() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        SessionLifecycleEvent(
            session_id="session-1",
            turn_index=1,
            state="turn_started",
            message="started",
            details={
                "sandbox": {
                    "backend": "docker",
                    "availability": {"degraded": False, "reason": ""},
                },
            },
        ),
    )

    assert app._sandbox_status == {
        "backend": "docker",
        "degraded": False,
        "reason": "",
    }
