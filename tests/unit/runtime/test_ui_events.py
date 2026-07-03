"""
tests/unit/runtime/test_ui_events.py - Runtime-TUI 事件协议测试

验证 runtime 原始事件会被转换为 TUI 可直接消费的强类型事件。
"""

from haagent.runtime.events import (
    ApprovalStateEvent,
    AssistantDeltaEvent,
    FailureNoticeEvent,
    RAW_RUNTIME_UI_EVENT_REGISTRY,
    RuntimeUiEventMapper,
    ToolActivityEvent,
    WarningNoticeEvent,
)


def test_runtime_ui_event_registry_lists_supported_raw_event_types() -> None:
    assert set(RAW_RUNTIME_UI_EVENT_REGISTRY) == {
        "assistant_delta",
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
        "tool_result_microcompact",
        "loop_suggestion_added",
        "safety_abort",
        "interaction_reused",
        "failure",
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


def test_runtime_ui_event_mapper_groups_tool_result_microcompact_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "tool_result_microcompact",
            "turn": 2,
            "message_index": 4,
            "tool_name": "web_fetch",
            "original_chars": 20000,
            "final_chars": 2400,
            "decision": "collapsed",
            "reason": "old_tool_result_over_budget",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == WarningNoticeEvent(
        session_id="session-1",
        turn_index=1,
        title="Tool result compacted",
        message="web_fetch result compacted from 20000 to 2400 chars",
        notice_kind="tool_result_microcompact",
        surface="tool_detail",
        details={
            "turn": 2,
            "message_index": 4,
            "tool_name": "web_fetch",
            "original_chars": 20000,
            "final_chars": 2400,
            "decision": "collapsed",
            "reason": "old_tool_result_over_budget",
        },
    )


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


def test_runtime_ui_event_mapper_groups_safety_abort_as_warning() -> None:
    event = RuntimeUiEventMapper.to_ui_event(
        {
            "event_type": "safety_abort",
            "turn": 3,
            "violation_type": "workspace_boundary",
            "message": "blocked outside workspace",
        },
        session_id="session-1",
        turn_index=1,
    )

    assert event == WarningNoticeEvent(
        session_id="session-1",
        turn_index=1,
        title="Safety abort",
        message="blocked outside workspace",
        notice_kind="safety_abort",
        surface="timeline",
        details={
            "turn": 3,
            "violation_type": "workspace_boundary",
            "message": "blocked outside workspace",
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
