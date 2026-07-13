"""
tests/unit/runtime/test_ui_events.py - Runtime-TUI 事件协议测试

验证 runtime 原始事件会被转换为 TUI 可直接消费的强类型事件。
"""

from haagent.runtime.events import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    AssistantIntermediateEvent,
    FailureNoticeEvent,
    RAW_RUNTIME_UI_EVENT_REGISTRY,
    RuntimeUiEventMapper,
    ToolActivityEvent,
    WarningNoticeEvent,
)
from haagent.runtime.events.bus import (
    ModelRetryExhaustedBusEvent,
    ModelRetryScheduledBusEvent,
    bus_event_from_dict,
    bus_event_to_dict,
)


def test_model_retry_bus_events_round_trip_with_safe_fields() -> None:
    scheduled = ModelRetryScheduledBusEvent(
        turn=1,
        attempt=1,
        next_attempt=2,
        category="rate_limited",
        delay_seconds=1.2,
        source="retry_after",
    )
    exhausted = ModelRetryExhaustedBusEvent(
        turn=1,
        attempt=3,
        category="server",
        status_code=503,
        request_id="req_123",
    )

    assert bus_event_from_dict(bus_event_to_dict(scheduled)) == scheduled
    assert bus_event_from_dict(bus_event_to_dict(exhausted)) == exhausted


def test_runtime_ui_event_registry_lists_supported_raw_event_types() -> None:
    assert set(RAW_RUNTIME_UI_EVENT_REGISTRY) == {
        "assistant_delta",
        "assistant_intermediate_message",
        "assistant_message",
        "tool_started",
        "tool_finished",
        "tool_failed",
        "approval_requested",
        "approval_granted",
        "approval_denied",
        "edit_diff_requested",
        "edit_diff_granted",
        "edit_diff_denied",
        "user_input_requested",
        "user_input_received",
        "guardrail_triggered",
        "compression_diagnostic",
        "loop_suggestion_added",
        "interaction_reused",
        "failure",
        "worker_started",
        "worker_completed",
        "worker_failed",
        "worker_stopped",
        "task_plan_created",
        "task_step_started",
        "task_step_progress",
        "task_step_finished",
        "task_step_blocked",
        "task_checkpoint_saved",
        "task_recovery_suggested",
        "task_budget_warning",
        "model_retry_scheduled",
        "model_retry_exhausted",
        "model_protocol_fallback",
        "model_fallback",
    }


def test_runtime_ui_event_mapper_groups_assistant_delta() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {"event_type": "assistant_delta", "turn": 2, "delta": "正在整理"},
        session_id="session-1",
        turn_index=1,
    )
    assert event == AssistantDeltaEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=2,
        delta="正在整理",
    )


def test_runtime_ui_event_mapper_preserves_intermediate_model_turn() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "assistant_intermediate_message",
            "turn": 3,
            "content": "完整审查报告",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == AssistantIntermediateEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=3,
        content="完整审查报告",
    )


def test_runtime_ui_event_mapper_exposes_model_retry_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {"event_type": "model_retry_scheduled", "turn": 1, "attempt": 1, "next_attempt": 2, "category": "rate_limited", "delay_seconds": 1.23456, "source": "backoff"},
        session_id="session-1", turn_index=1,
    )

    assert isinstance(event, WarningNoticeEvent)
    assert event.notice_kind == "model_retry"
    assert "1.2 秒后" in event.message


def test_runtime_ui_event_mapper_marks_exhausted_model_retry_as_error_notice() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "model_retry_exhausted",
            "turn": 1,
            "attempt": 3,
            "category": "server",
            "status_code": 503,
        },
        session_id="session-1",
        turn_index=1,
    )

    assert isinstance(event, WarningNoticeEvent)
    assert event.notice_kind == "model_retry_exhausted"
    assert "已耗尽" in event.message



def test_runtime_ui_event_mapper_groups_tool_events_as_activity() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "tool_finished",
            "turn": 3,
            "tool_name": "file_read",
            "result": {"status": "ok", "content": "hello"},
        },
        session_id="session-1",
        turn_index=1,
    )

    assert isinstance(event, ToolActivityEvent)
    assert event.session_id == "session-1"
    assert event.turn_index == 1
    assert event.model_turn == 3
    assert event.tool_name == "file_read"
    assert event.status == "finished"
    assert event.summary


def test_runtime_ui_event_mapper_groups_worker_events_as_activity() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "worker_completed",
            "agent_id": "explorer-1",
            "task_id": "task-1",
            "team_id": "team-1",
            "subagent_type": "explorer",
            "description": "Inspect project",
            "status": "completed",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == ToolActivityEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=None,
        tool_name="agent:explorer-1",
        status="finished",
        summary="worker completed: Inspect project",
        args_summary={
            "agent_id": "explorer-1",
            "task_id": "task-1",
            "team_id": "team-1",
            "subagent_type": "explorer",
        },
        result_status="completed",
    )


def test_runtime_ui_event_mapper_groups_approval_events() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "approval_requested",
            "turn": 1,
            "tool_name": "shell",
            "question": "允许运行 shell？",
            "args_summary": {"command": "pytest"},
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == ApprovalStateEvent(
        session_id="session-1",
        turn_index=1,
        model_turn=1,
        tool_name="shell",
        state="requested",
        question="允许运行 shell？",
        approved=None,
        answer="",
        args_summary={"command": "pytest"},
    )


def test_runtime_ui_event_mapper_groups_failure_notice() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "failure",
            "status": "failed",
            "failed_stage": "model_call",
            "failure_category": "provider",
            "reason": "HTTP 404",
            "episode_path": "E:/workspace/.runs/episode",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == FailureNoticeEvent(
        session_id="session-1",
        turn_index=1,
        status="failed",
        failed_stage="model_call",
        failure_category="provider",
        reason="HTTP 404",
        episode_path="E:/workspace/.runs/episode",
    )


def test_runtime_ui_event_mapper_groups_compression_diagnostic_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "compression_diagnostic",
            "turn": 2,
            "message_index": 4,
            "stage": "historical_tool_message",
            "subject": "web_fetch",
            "original_chars": 20000,
            "final_chars": 2400,
            "decision": "collapsed",
            "reason": "long_text_result",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == WarningNoticeEvent(
        session_id="session-1",
        turn_index=1,
        title="压缩诊断",
        message="旧工具消息降级：web_fetch 20000 chars -> 2400 chars",
        notice_kind="compression_diagnostic",
        surface="tool_detail",
        details={
            "turn": 2,
            "message_index": 4,
            "stage": "historical_tool_message",
            "subject": "web_fetch",
            "original_chars": 20000,
            "final_chars": 2400,
            "decision": "collapsed",
            "reason": "long_text_result",
        },
    )


def test_runtime_ui_event_mapper_labels_all_compression_stages() -> None:
    expected = {
        "tool_output_artifact": "工具输出落盘",
        "historical_tool_message": "旧工具消息降级",
        "context_section": "上下文 section 折叠",
        "session_memory": "会话记忆压缩",
        "full_compact": "自动 full compact",
    }

    for stage, label in expected.items():
        event = RuntimeUiEventMapper.to_ui_event(
            {
                "event_type": "compression_diagnostic",
                "stage": stage,
                "subject": "subject",
                "original_chars": 10,
                "final_chars": 5,
                "decision": "collapsed",
                "reason": "test",
            },
            session_id="session-1",
            turn_index=1,
        )

        assert isinstance(event, WarningNoticeEvent)
        assert label in event.message


def test_runtime_ui_event_mapper_groups_loop_suggestion_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "loop_suggestion_added",
            "turn": 2,
            "trigger": "tool_argument_invalid",
            "tool_name": "file_read",
            "message": "请改用相对路径重试。",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == WarningNoticeEvent(
        session_id="session-1",
        turn_index=1,
        title="Loop guidance",
        message="请改用相对路径重试。",
        notice_kind="loop_guidance",
        surface="hidden",
        details={
            "turn": 2,
            "trigger": "tool_argument_invalid",
            "tool_name": "file_read",
            "message": "请改用相对路径重试。",
        },
    )


def test_runtime_ui_event_mapper_groups_reused_interaction_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "interaction_reused",
            "turn": 4,
            "interaction_type": "approval",
            "tool_name": "shell",
            "question": "允许运行？",
            "status": "reused",
            "approved": True,
            "resolved_turn": 2,
            "signature": "sig",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == WarningNoticeEvent(
        session_id="session-1",
        turn_index=1,
        title="Interaction reused",
        message="approval for shell reused from turn 2",
        details={
            "turn": 4,
            "interaction_type": "approval",
            "tool_name": "shell",
            "question": "允许运行？",
            "status": "reused",
            "approved": True,
            "resolved_turn": 2,
            "signature": "sig",
        },
    )


def test_runtime_ui_event_mapper_turns_unknown_events_into_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {"event_type": "new_runtime_event", "value": "kept"},
        session_id="session-1",
        turn_index=1,
    )

    assert isinstance(event, WarningNoticeEvent)
    assert event.title == "Runtime warning"
    assert event.notice_kind == "runtime_warning"
    assert event.surface == "timeline"
    assert "new_runtime_event" in event.message
    assert event.details == {"value": "kept"}
