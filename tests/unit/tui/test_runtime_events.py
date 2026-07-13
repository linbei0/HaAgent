"""
tests/unit/tui/test_runtime_events.py - TUI 强类型 runtime 事件处理测试

直接使用 RuntimeUiEvent 验证 TUI adapter 的状态更新与展示动作。
"""

from __future__ import annotations

from haagent.runtime.events import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantIntermediateEvent,
    AssistantMessageEvent,
    FailureNoticeEvent,
    MemoryNoticeEvent,
    RUNTIME_UI_EVENT_TYPES,
    SessionLifecycleEvent,
    TaskProgressEvent,
    ToolActivityEvent,
    UserInputStateEvent,
    WarningNoticeEvent,
)
from haagent.tui.application.runtime_events import RUNTIME_UI_EVENT_HANDLERS, handle_runtime_ui_event
from haagent.tui.widgets.conversation_timeline import ConversationTimeline
from haagent.tui.widgets.timeline_models import TimelineItem


def test_runtime_ui_event_handler_registry_covers_protocol_types() -> None:
    assert set(RUNTIME_UI_EVENT_HANDLERS) == set(RUNTIME_UI_EVENT_TYPES)


class _FakeConversation:
    def __init__(self, app: "FakeRuntimeEventApp") -> None:
        self._app = app

    def merge_assistant_delta(self, turn_index: int, model_turn: int | None, delta: str) -> None:
        del model_turn
        self._app.assistant_deltas.append((turn_index, delta))

    def finalize_intermediate_message(self, turn_index: int, model_turn: int | None, content: str) -> None:
        self._app.assistant_intermediates.append((turn_index, model_turn, content))

    def finalize_assistant_message(self, turn_index: int, model_turn: int | None, content: str) -> None:
        del model_turn
        self._app.assistant_messages.append((turn_index, content))
        self._app.presentation_texts.append(f"助手\n{content}")

    def record_tool_activity(self, turn_index: int, tool_name: str, status: str, summary: str) -> None:
        self._app.tool_activities.append((turn_index, tool_name, status, summary))

    def record_tool_diagnostic(self, turn_index: int, tool_name: str, message: str) -> None:
        self._app.tool_diagnostics.append((turn_index, tool_name, message))

    def append_block(self, title: str, body: str, *, turn_index: int | None = None) -> None:
        self._app.blocks.append((title, body))

    def append_line(self, line: str, *, turn_index: int | None = None) -> None:
        self._app.lines.append(line)

    def finalize_streaming_if_needed(self) -> None:
        self._app.finalized_streams += 1


class _FakeMemoryFlow:
    def __init__(self, app: "FakeRuntimeEventApp") -> None:
        self._app = app
        self.notice = None
        self.mode = False
        self.detail_mode = True

    def load_candidates(self, *, silent: bool = False) -> None:
        assert silent is True
        self._app.memory_loads += 1


class FakeRuntimeEventApp:
    def __init__(self) -> None:
        self._state = "idle"
        self._last_failure = None
        self._sandbox_status = None
        self._active_turn_index = 0
        self.assistant_deltas: list[tuple[int, str]] = []
        self.assistant_intermediates: list[tuple[int, int | None, str]] = []
        self.assistant_messages: list[tuple[int, str]] = []
        self.tool_activities: list[tuple[int, str, str, str]] = []
        self.tool_diagnostics: list[tuple[int, str, str]] = []
        self.blocks: list[tuple[str, str]] = []
        self.lines: list[str] = []
        self.answer_questions: list[str] = []
        self.task_progress_events: list[TaskProgressEvent] = []
        self.presentation_texts: list[str] = []
        self.presentation_detail_ids: list[str | None] = []
        self.progress_status_text = ""
        self.progress_status_severity = ""
        self.memory_loads = 0
        self.finalized_streams = 0
        self.refreshes = 0
        self.streaming_refresh_schedules = 0
        self._conversation = _FakeConversation(self)
        self.memory_flow = _FakeMemoryFlow(self)

    def _set_answer_required(self, question: str) -> None:
        self.answer_questions.append(question)

    def _refresh(self) -> None:
        self.refreshes += 1

    def _schedule_streaming_refresh(self) -> None:
        # 真实 App 会 16～50ms 批量刷新 timeline；测试只统计调度次数。
        self.streaming_refresh_schedules += 1

    def set_progress_status(self, status) -> None:
        self.progress_status_text = status.text
        self.progress_status_severity = status.severity

    def clear_progress_status(self) -> None:
        self.progress_status_text = ""
        self.progress_status_severity = ""

    def query_one(self, selector: str, widget_type):
        assert selector == "#conversation"
        return self

    @property
    def plain_text(self) -> str:
        return "\n".join(self.presentation_texts)

    def add_presentation_item(self, item, details) -> None:
        self.presentation_texts.append(f"{item.title}\n{item.summary}")
        self.presentation_detail_ids.append(item.detail_id)

    def replace_presentation_item(self, item, details) -> bool:
        if item.detail_id is None:
            return False
        try:
            index = self.presentation_detail_ids.index(item.detail_id)
        except ValueError:
            return False
        self.presentation_texts[index] = f"{item.title}\n{item.summary}"
        return True

    def count_presentations_containing(self, text: str) -> int:
        return sum(1 for item in self.presentation_texts if text in item)


class TimelineRuntimeEventApp(FakeRuntimeEventApp):
    def __init__(self) -> None:
        super().__init__()
        self.timeline = ConversationTimeline()

        class _TimelineConversation(_FakeConversation):
            def finalize_assistant_message(self, turn_index: int, model_turn: int | None, content: str) -> None:
                del model_turn
                self._app.assistant_messages.append((turn_index, content))
                self._app.timeline._items.append(
                    TimelineItem(
                        item_id=next(self._app.timeline._ids),
                        role="assistant",
                        turn_index=turn_index,
                        content=content,
                    )
                )
                self._app.timeline._mark_plain_text_dirty()

        self._conversation = _TimelineConversation(self)

    def query_one(self, selector: str, widget_type):
        assert selector == "#conversation"
        return self.timeline

    @property
    def plain_text(self) -> str:
        return self.timeline.plain_text

    def count_presentations_containing(self, text: str) -> int:
        return self.plain_text.count(text)


def test_runtime_ui_event_handler_updates_assistant_stream() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, AssistantDeltaEvent("session-1", 2, 1, "半句"))
    handle_runtime_ui_event(app, AssistantDeltaEvent("session-1", 2, 1, "续写"))
    handle_runtime_ui_event(app, AssistantMessageEvent("session-1", 2, 1, "整句"))

    assert app.assistant_deltas == [(2, "半句"), (2, "续写")]
    assert app.assistant_messages == [(2, "整句")]
    # delta 热路径禁止全量 _refresh（会打 status/keyring）；只调度批量 timeline 刷新。
    assert app.refreshes == 1
    assert app.streaming_refresh_schedules == 2


def test_assistant_delta_does_not_trigger_full_refresh() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, AssistantDeltaEvent("session-1", 1, 1, "a"))
    handle_runtime_ui_event(app, AssistantDeltaEvent("session-1", 1, 1, "b"))

    assert app.assistant_deltas == [(1, "a"), (1, "b")]
    assert app.refreshes == 0
    assert app.streaming_refresh_schedules == 2


def test_runtime_ui_event_handler_routes_intermediate_assistant_message() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        AssistantIntermediateEvent("session-1", 2, 3, "完整审查报告"),
    )

    assert app.assistant_intermediates == [(2, 3, "完整审查报告")]
    assert app.refreshes == 1


def test_runtime_ui_event_handler_does_not_persist_read_tool_activity() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, ToolActivityEvent("session-1", 1, 2, "file_read", "finished", "读取 README"))

    assert app.tool_activities == []
    assert app.plain_text == ""


def test_runtime_ui_event_handler_routes_read_tool_to_progress_status() -> None:
    app = FakeRuntimeEventApp()
    event = ToolActivityEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=2,
        tool_name="file_read",
        status="started",
        summary="",
        args_summary={"path": "README.md"},
    )

    handle_runtime_ui_event(app, event)

    text = app.query_one("#conversation", ConversationTimeline).plain_text
    assert app.progress_status_text == "正在阅读文件..."
    assert app.tool_activities == []
    assert "file_read" not in text
    assert "README.md" not in text


def test_runtime_ui_event_handler_routes_web_fetch_to_progress_status() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="web_fetch",
            status="started",
            summary="fetching url",
            args_summary={"url": "https://example.com"},
        ),
    )

    assert app.progress_status_text == "正在阅读资料..."
    assert app.tool_activities == []
    assert "web_fetch" not in app.plain_text
    assert "example.com" not in app.plain_text


def test_runtime_ui_event_handler_routes_apply_patch_to_effect_summary() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="apply_patch",
            status="finished",
            summary="modified 2 files",
            result_status="success",
        ),
    )

    assert app.tool_activities == []
    assert "已修改文件" in app.plain_text
    assert "2 个文件有变更" in app.plain_text
    assert "apply_patch" not in app.plain_text


def test_runtime_ui_event_handler_groups_tool_failures_before_assistant_final() -> None:
    app = FakeRuntimeEventApp()
    first_error = "stderr: " + ("x" * 500)

    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="web_fetch",
            status="failed",
            summary="fetch failed",
            args_summary={"url": "https://example.com/one"},
            error_type="network",
            error_message=first_error,
        ),
    )
    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="web_fetch",
            status="failed",
            summary="fetch failed again",
            args_summary={"url": "https://example.com/two"},
            error_type="network",
            error_message="second failure",
        ),
    )
    handle_runtime_ui_event(app, AssistantMessageEvent("session-1", 1, 2, "最终回答"))

    text = app.plain_text
    assert "web_fetch 失败 2 次，已使用已有上下文继续" in text
    assert app.count_presentations_containing("web_fetch 失败") == 1
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")
    assert "example.com" not in text
    assert first_error not in text
    assert "step-001" not in text
    assert "model_turn_started" not in text
    assert "task_step_progress" not in text


def test_runtime_ui_event_handler_inserts_late_tool_failures_before_assistant_final() -> None:
    app = TimelineRuntimeEventApp()

    handle_runtime_ui_event(app, AssistantMessageEvent("session-1", 1, 2, "最终回答"))
    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="web_fetch",
            status="failed",
            summary="fetch failed",
            error_type="network",
            error_message="first failure",
        ),
    )
    handle_runtime_ui_event(
        app,
        ToolActivityEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            tool_name="web_fetch",
            status="failed",
            summary="fetch failed again",
            error_type="network",
            error_message="second failure",
        ),
    )

    text = app.plain_text
    assert "已处理 1 项 >" in text
    assert "web_fetch 失败 2 次，已使用已有上下文继续" not in text
    assert text.index("已处理 1 项") < text.index("最终回答")

    app.timeline.toggle_process_group(1)
    text = app.plain_text
    assert "web_fetch 失败 2 次，已使用已有上下文继续" in text
    assert text.count("web_fetch") == 1
    assert text.index("web_fetch 失败 2 次") < text.index("最终回答")


def test_runtime_ui_event_handler_groups_late_task_problems_before_assistant_final() -> None:
    app = TimelineRuntimeEventApp()

    handle_runtime_ui_event(app, AssistantMessageEvent("session-1", 1, 2, "最终回答"))
    handle_runtime_ui_event(
        app,
        TaskProgressEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            event_name="task_step_blocked",
            step_id="step-timeout",
            title="工具超时",
            status="blocked",
            summary="tool timed out",
            category="tool_timeout",
            suggested_action="retry_with_narrower_command",
        ),
    )
    handle_runtime_ui_event(
        app,
        TaskProgressEvent(
            session_id="session-1",
            turn_index=1,
            model_turn=2,
            event_name="task_step_blocked",
            step_id="step-blocked",
            title="任务受阻",
            status="blocked",
            summary="task blocked",
            category="tool_failed",
            suggested_action="resume_or_replan",
        ),
    )

    text = app.plain_text
    assert "已处理 1 项 >" in text
    assert "任务遇到问题 2 项：工具超时、工具失败" not in text
    assert text.index("已处理 1 项") < text.index("最终回答")
    assert "step-timeout" not in text
    assert "step-blocked" not in text

    app.timeline.toggle_process_group(1)
    text = app.plain_text
    assert "任务遇到问题 2 项：工具超时、工具失败" in text
    assert text.count("任务遇到问题") == 1
    assert text.index("任务遇到问题 2 项") < text.index("最终回答")
    assert "step-timeout" not in text
    assert "step-blocked" not in text


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

    assert app.tool_activities == []
    assert "需要确认：shell" in app.plain_text
    assert "建议：在弹窗中确认或拒绝" in app.plain_text
    assert app._state == "waiting approval"

    handle_runtime_ui_event(
        app,
        ApprovalStateEvent("session-1", 1, 2, "shell", "granted", "", True),
    )

    assert app.tool_activities == []
    assert "审批已允许" not in app.plain_text
    assert app.lines == []
    assert app._state == "running"


def test_runtime_ui_event_handler_keeps_approval_denials_visible() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        ApprovalStateEvent("session-1", 1, 2, "shell", "denied", "", False),
    )

    assert app.tool_activities == []
    assert app.lines == []
    assert "审批已拒绝：shell" in app.plain_text
    assert "建议：调整请求或选择其他方案" in app.plain_text


def test_runtime_ui_event_handler_routes_compression_diagnostic_to_tool_detail() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        WarningNoticeEvent(
            session_id="session-1",
            turn_index=1,
            title="压缩诊断",
            message="旧工具消息降级：web_search 1854 chars -> 929 chars",
            notice_kind="compression_diagnostic",
            surface="tool_detail",
            details={
                "stage": "historical_tool_message",
                "subject": "web_search",
                "original_chars": 1854,
                "final_chars": 929,
            },
        ),
    )

    assert app.blocks == []
    assert app.tool_diagnostics == [(1, "web_search", "旧工具消息降级：web_search 1854 chars -> 929 chars")]


def test_runtime_ui_event_handler_does_not_special_case_legacy_microcompact_notice() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        WarningNoticeEvent(
            session_id="session-1",
            turn_index=1,
            title="Runtime warning",
            message="Unknown runtime event: tool_result_microcompact",
            notice_kind="runtime_warning",
            surface="tool_detail",
            details={
                "tool_name": "web_search",
                "original_chars": 1854,
                "final_chars": 929,
            },
        ),
    )

    assert app.tool_diagnostics == [(1, "web_search", "Unknown runtime event: tool_result_microcompact")]


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

    assert app.answer_questions == ["你要哪个文件？"]
    assert app.blocks == []
    assert "需要补充信息" in app.plain_text
    assert "你要哪个文件？" in app.plain_text

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

    assert app.blocks == []
    assert app.lines == []
    assert "回答已提交" not in app.plain_text
    assert app._state == "running"


def test_runtime_ui_event_handler_shows_cancelled_user_input_notice() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(
        app,
        UserInputStateEvent(
            "session-1",
            1,
            2,
            "request_user_input",
            "received",
            "你要哪个文件？",
            approved=False,
        ),
    )

    assert app.lines == []
    assert "回答已取消：request_user_input" in app.plain_text
    assert "建议：补充信息后重试或调整任务" in app.plain_text


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


def test_runtime_ui_event_handler_suppresses_plain_turn_lifecycle_task_progress() -> None:
    app = FakeRuntimeEventApp()
    started = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_step_started",
        step_id="step-001",
        title="你好",
        status="running",
        summary="started task step step-001: 你好",
        category="none",
        suggested_action="none",
    )
    finished = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_step_finished",
        step_id="step-001",
        title="你好",
        status="completed",
        summary="completed task step step-001: 你好",
        category="none",
        suggested_action="none",
        evidence_count=1,
        checkpoint_count=1,
    )

    handle_runtime_ui_event(app, started)
    handle_runtime_ui_event(app, finished)

    assert app.task_progress_events == []
    assert app.plain_text == ""


def test_runtime_ui_event_handler_does_not_route_task_step_progress_to_timeline() -> None:
    app = FakeRuntimeEventApp()
    event = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_step_progress",
        step_id="step-001",
        title="你好",
        status="running",
        summary="model turn started",
        category="model_turn_started",
    )

    handle_runtime_ui_event(app, event)

    assert app.task_progress_events == []
    assert "任务进度" not in app.plain_text
    assert "step-001" not in app.plain_text
    assert "model_turn_started" not in app.plain_text


def test_runtime_ui_event_handler_routes_recovery_to_notice() -> None:
    app = FakeRuntimeEventApp()
    event = TaskProgressEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        event_name="task_recovery_suggested",
        step_id="step-001",
        title="运行测试",
        status="blocked",
        summary="",
        category="verification_failed",
        suggested_action="修复后重新运行测试",
        evidence_count=1,
        checkpoint_count=1,
        reason_chars=140,
    )

    handle_runtime_ui_event(app, event)

    assert app.task_progress_events == []
    assert "任务遇到问题：验证失败" in app.plain_text
    assert "建议：修复后重新运行测试" in app.plain_text
    assert "详情：" not in app.plain_text
    assert app.presentation_detail_ids == ["task:1:problems"]
    assert "reason_chars" not in app.plain_text


def test_runtime_ui_event_handler_opens_memory_notice() -> None:
    app = FakeRuntimeEventApp()

    handle_runtime_ui_event(app, MemoryNoticeEvent("session-1", 1, "发现 1 条候选", count=1))

    assert app.blocks == [("Memory", "发现 1 条候选")]
    assert app.memory_flow.notice == "发现 1 条候选"
    assert app.memory_flow.mode is True
    assert app.memory_flow.detail_mode is False
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
