"""
tests/unit/runtime/test_steering.py - 运行中引导（steering）测试

验证 SteeringChannel 行为、turn loop 边界注入、完成竞态防护与取消时 partial 保留。
"""

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from haagent.models.types import ModelResponse, ToolCall
from haagent.runtime.execution.cancellation import RunCancelled
from haagent.runtime.execution.steering import SteeringChannel
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.orchestration.turns import (
    TurnLoopState,
    _handle_no_tool_response,
    run_turn_loop,
)
from haagent.runtime.events.bus import bus_event_to_dict
from haagent.runtime.session.turn_completion import ChatTurnResult, turn_summary

from tests.unit.runtime.test_run_turns import _deps, _FakeRouter, _FakeWriter, _replace_dep


def test_steering_channel_post_drain_and_has_pending() -> None:
    channel = SteeringChannel()
    assert channel.has_pending() is False
    channel.post("  用中文回答  ")
    channel.post("")
    channel.post("   ")
    assert channel.has_pending() is True
    assert channel.drain() == ["用中文回答"]
    assert channel.has_pending() is False
    assert channel.drain() == []


def test_steering_channel_concurrent_posts_are_not_lost() -> None:
    channel = SteeringChannel()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index in range(200):
            pool.submit(channel.post, f"msg-{index}")
    drained = channel.drain()
    assert sorted(drained) == sorted(f"msg-{index}" for index in range(200))


def test_turn_loop_injects_steering_before_next_model_call(tmp_path) -> None:
    channel = SteeringChannel()
    writer = _FakeWriter(tmp_path / "episode")
    emitted_events: list[object] = []
    model_messages: list[list[dict[str, Any]]] = []

    class _SteeringRouter(_FakeRouter):
        def dispatch(self, tool_name, args, interaction_handler=None, *, turn=None):
            # 模拟用户在工具执行期间提交引导。
            channel.post("改成只分析不修改")
            return super().dispatch(tool_name, args, interaction_handler, turn=turn)

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, invocation, **kwargs):
            model_messages.append(list(invocation.messages))
            if len(model_messages) == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[ToolCall(name="file_read", args={"path": "a.txt"}, id="c1")],
                )
            return ModelResponse(content="已按引导调整", tool_calls=[])

    deps = _deps(
        router=_SteeringRouter({"status": "success", "content": "read"}),
        writer=writer,
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "workspace_root", tmp_path)
    deps = _replace_dep(deps, "steering_channel", channel)

    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    assert len(model_messages) == 2
    steering_messages = [
        message
        for message in model_messages[1]
        if message.get("role") == "user" and "改成只分析不修改" in str(message.get("content", ""))
    ]
    assert len(steering_messages) == 1
    assert "[用户在任务执行中插入的引导" in steering_messages[0]["content"]
    assert any(record.get("event") == "steering_injected" for record in writer.transcript)
    payloads = [bus_event_to_dict(event) for event in emitted_events]
    injected = [p for p in payloads if p.get("event_type") == "steering_injected"]
    assert injected == [{"event_type": "steering_injected", "turn": 2, "content": "改成只分析不修改"}]
    assert channel.has_pending() is False


def test_no_tool_response_defers_completion_when_steering_pending(tmp_path) -> None:
    channel = SteeringChannel()
    channel.post("请补充性能影响分析")
    writer = _FakeWriter(tmp_path)
    transitions: list[RunStatus] = []
    emitted_events: list[object] = []
    deps = _deps(
        router=_FakeRouter({}),
        writer=writer,
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=[RunStatus.EXECUTING],
            transition=transitions.append,
            finish=lambda status: SimpleNamespace(status=status),
        ),
    )
    deps = _replace_dep(deps, "steering_channel", channel)
    state = TurnLoopState(messages=[], context_id="ctx")

    result = _handle_no_tool_response(
        turn=2,
        model_response=ModelResponse(content="初步结论如下", tool_calls=[]),
        state=state,
        deps=deps,
    )

    assert result is None
    assert transitions == []
    deferred = [
        record
        for record in writer.transcript
        if record.get("event") == "completion_candidate_deferred" and record.get("reason") == "pending_steering"
    ]
    assert len(deferred) == 1
    # 顺序：先保留本轮回答，再注入引导。
    assert state.messages[-2]["role"] == "assistant"
    assert state.messages[-2]["content"] == "初步结论如下"
    assert state.messages[-1]["role"] == "user"
    assert "请补充性能影响分析" in state.messages[-1]["content"]
    assert channel.has_pending() is False


def test_no_tool_response_completes_normally_without_steering(tmp_path) -> None:
    writer = _FakeWriter(tmp_path)
    deps = _deps(
        router=_FakeRouter({}),
        writer=writer,
        recorder=SimpleNamespace(
            state_history=[RunStatus.EXECUTING],
            transition=lambda status: None,
            finish=lambda status: status,
        ),
    )
    deps = _replace_dep(deps, "steering_channel", SteeringChannel())
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = _handle_no_tool_response(
        turn=1,
        model_response=ModelResponse(content="完成", tool_calls=[], termination="completed"),
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=deps,
    )

    assert result is RunStatus.COMPLETED


def test_cancelled_stream_writes_partial_transcript(tmp_path) -> None:
    writer = _FakeWriter(tmp_path / "episode")

    class _CancelledGateway:
        provider_name = "fake"

        def generate(self, invocation, event_sink=None, **kwargs):
            del invocation, kwargs
            assert event_sink is not None
            event_sink("这是被打断前")
            event_sink("的部分回答")
            raise RunCancelled("user cancelled current run")

    deps = _deps(
        router=_FakeRouter({}),
        writer=writer,
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: SimpleNamespace(status=status),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _CancelledGateway())

    with pytest.raises(RunCancelled):
        run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    partial_records = [
        record for record in writer.transcript if record.get("event") == "model_response_partial"
    ]
    assert partial_records == [
        {
            "event": "model_response_partial",
            "turn": 1,
            "content": "这是被打断前的部分回答",
            "interrupted_by": "user_cancel",
        },
    ]


def test_episode_package_exposes_last_partial_response_text() -> None:
    from haagent.runtime.episodes.package_types import EpisodePackage

    package = EpisodePackage(
        path=None,
        metadata=SimpleNamespace(),
        failure=SimpleNamespace(),
        context_manifest=SimpleNamespace(),
        transcript=[
            {"event": "model_call", "turn": 1},
            {"event": "model_response_partial", "turn": 1, "content": "写到一半"},
        ],
        tool_calls=[],
        verification_commands=[],
    )
    assert package.last_partial_response_text() == "写到一半"
    assert package.final_response_text() == "none"


def test_final_response_with_partial_marks_interrupted_reply() -> None:
    from haagent.runtime.session.turn_completion import _final_response_with_partial

    package = SimpleNamespace(
        final_response_text=lambda: "none",
        last_partial_response_text=lambda: "写到一半",
    )
    assert _final_response_with_partial(package, "cancelled") == "写到一半\n\n[回复被用户打断，未完成]"
    # 完整回答存在或非取消状态时保持原值。
    package_full = SimpleNamespace(
        final_response_text=lambda: "完整回答",
        last_partial_response_text=lambda: "不应使用",
    )
    assert _final_response_with_partial(package_full, "cancelled") == "完整回答"
    assert _final_response_with_partial(package, "completed") == "none"


def test_turn_summary_includes_user_steering_line() -> None:
    from pathlib import Path

    result = ChatTurnResult(
        session_id="s",
        turn_index=1,
        status="completed",
        episode_path=Path("episode-1"),
        provider="fake",
        final_response="done",
        verification_status="not_run",
    )
    summary = turn_summary("原始请求", result, steering_texts=["改用中文", "只读不写"])
    assert "user_steering: 改用中文 | 只读不写" in summary
    assert "user_steering" not in turn_summary("原始请求", result)
