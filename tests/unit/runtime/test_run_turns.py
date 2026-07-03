"""
tests/unit/runtime/test_run_turns.py - 单轮工具执行循环测试

验证工具 observation 写入 trace 与模型可见消息之间的分层行为。
"""

from types import SimpleNamespace
from typing import Any
from pathlib import Path

from haagent.models.gateway import ToolCall
from haagent.models.gateway import ModelResponse
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.orchestration.turns import (
    TurnLoopDependencies,
    TurnLoopState,
    _handle_no_tool_response,
    _run_tool_calls,
    run_turn_loop,
)


class _FakeRouter:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.waited_task_ids: list[str] = []

    def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None) -> dict[str, Any]:
        del tool_name, args, interaction_handler
        return dict(self.result)

    def raise_for_error(self, result: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected terminal error: {result}")

    def wait_for_agent_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        del timeout
        self.waited_task_ids.append(task_id)
        return {
            "task_id": task_id,
            "agent_id": "explorer-test",
            "team_id": "team-test",
            "status": "completed",
            "summary": "worker inspected the project",
            "result_excerpt": "worker inspected the project",
            "usage": {},
            "error": "",
        }


class _FakeWriter:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.cwd()
        self.transcript: list[dict[str, Any]] = []
        self.failure_records: list[dict[str, Any] | None] = []

    def append_transcript(self, record: dict[str, Any]) -> None:
        self.transcript.append(record)

    def append_tool_call(self, record: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected skipped tool call: {record}")

    def write_failure_attribution(self, record: dict[str, Any] | None) -> None:
        self.failure_records.append(record)


def test_same_turn_duplicate_tool_result_is_collapsed_for_model_context() -> None:
    writer = _FakeWriter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = TurnLoopDependencies(
        model_gateway=SimpleNamespace(),
        writer=writer,
        recorder=SimpleNamespace(transition=lambda status: None, finish=lambda status: None),
        router=_FakeRouter(
            {
                "status": "success",
                "content": "raw content",
                "model_visible": {"content": "visible content"},
            }
        ),
        task_goal="test duplicate observations",
        allowed_tools=["file_read"],
        tool_registry=SimpleNamespace(),
        verification_commands=[],
        workspace_root=object(),
        max_turns=3,
        raise_if_cancelled=lambda: None,
        emit_event=lambda event: None,
        microcompact_old_tool_messages=lambda messages, writer, turn, emit_event: None,
        interaction_handler=None,
        interaction_resolver=SimpleNamespace(),
        safety_guard=SimpleNamespace(check=lambda name, args, result: None),
        interaction_bridge_factory=lambda turn, resolver: None,
        record_guardrail=lambda violation, turn: None,
        record_suggestion=lambda turn, suggestion: None,
        tool_error_is_terminal=lambda result: False,
        update_in_band_verification_progress=lambda name, args, result, commands, passed: None,
        all_declared_verification_commands_passed=lambda commands, passed: False,
        successful_file_change_without_declared_verification=lambda observations, commands: False,
        verification_observation=lambda result: {},
        verification_evidence=lambda result: "",
        verification_loop_limit_evidence=lambda max_turns, result: "",
    )

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "notes.txt"}, id="call_1"),
            ToolCall(name="file_read", args={"path": "notes.txt"}, id="call_2"),
        ],
        state=state,
        deps=deps,
    )

    assert len(writer.transcript) == 2
    assert writer.transcript[1]["result"]["content"] == "raw content"
    assert "visible content" in state.messages[0]["content"]
    assert "raw content" not in state.messages[0]["content"]
    assert "same_as_previous" in state.messages[1]["content"]
    assert "raw content" not in state.messages[1]["content"]


def test_agent_tool_result_is_tracked_as_pending_worker_task() -> None:
    router = _FakeRouter(
        {
            "status": "running",
            "agent_id": "explorer-test",
            "task_id": "task-pending",
            "team_id": "team-test",
        }
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router)

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(
                name="agent",
                args={"description": "Inspect", "prompt": "Read files", "subagent_type": "explorer"},
                id="call_agent",
            ),
        ],
        state=state,
        deps=deps,
    )

    assert state.pending_worker_task_ids == ["task-pending"]


def test_no_tool_response_waits_for_pending_worker_before_finalizing() -> None:
    router = _FakeRouter({})
    writer = _FakeWriter()
    emitted_events: list[dict[str, object]] = []
    transitions: list[object] = []
    state = TurnLoopState(
        messages=[],
        context_id="ctx",
        pending_worker_task_ids=["task-pending"],
    )
    deps = _deps(
        router=router,
        writer=writer,
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=[],
            transition=transitions.append,
            finish=lambda status: SimpleNamespace(status=status),
        ),
    )

    result = _handle_no_tool_response(
        turn=2,
        model_response=ModelResponse(content="worker 已启动，请稍候", tool_calls=[]),
        state=state,
        deps=deps,
    )

    assert result is None
    assert router.waited_task_ids == ["task-pending"]
    assert state.pending_worker_task_ids == []
    assert emitted_events == []
    assert transitions == []
    assert "Worker notifications" in state.messages[-1]["content"]
    assert "worker inspected the project" in state.messages[-1]["content"]


def test_turn_loop_collects_pending_worker_notifications_before_model_can_poll() -> None:
    router = _FakeRouter(
        {
            "status": "running",
            "agent_id": "explorer-test",
            "task_id": "task-pending",
            "team_id": "team-test",
        }
    )
    writer = _FakeWriter()
    transitions: list[object] = []
    finish_statuses: list[object] = []
    model_messages: list[list[dict[str, Any]]] = []

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            del tool_schemas
            model_messages.append(list(messages))
            if len(model_messages) == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="agent",
                            args={"description": "Inspect", "prompt": "Read files", "subagent_type": "explorer"},
                            id="call_agent",
                        )
                    ],
                )
            if any("Worker notifications" in str(message.get("content", "")) for message in messages):
                return ModelResponse(content="worker inspected the project", tool_calls=[])
            return ModelResponse(
                content="",
                tool_calls=[ToolCall(name="task_get", args={"task_id": "task-pending"}, id="call_poll")],
            )

    deps = _deps(
        router=router,
        writer=writer,
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=transitions.append,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "all_declared_verification_commands_passed", lambda commands, passed: True)

    result = run_turn_loop(
        state=TurnLoopState(
            messages=[],
            context_id="ctx",
            verification_engine=SimpleNamespace(run=lambda commands: SimpleNamespace(status="success")),
        ),
        deps=deps,
    )

    assert result is not None
    assert finish_statuses == [RunStatus.COMPLETED]
    assert router.waited_task_ids == ["task-pending"]
    assert len(model_messages) == 2
    tool_call_names = [
        call["name"]
        for record in writer.transcript
        for call in record.get("tool_calls", [])
    ]
    assert "task_get" not in tool_call_names
    assert any(record["event"] == "worker_notifications_collected" for record in writer.transcript)


def test_turn_loop_allows_unlimited_max_turns(tmp_path: Path) -> None:
    finish_statuses: list[object] = []

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            del messages, tool_schemas
            return ModelResponse(content="done", tool_calls=[])

    deps = _deps(
        router=_FakeRouter({}),
        writer=_FakeWriter(tmp_path / "episode"),
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "max_turns", None)
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = run_turn_loop(
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=deps,
    )

    assert result is not None
    assert finish_statuses == [RunStatus.COMPLETED]


def _deps(
    *,
    router: _FakeRouter,
    writer: _FakeWriter | None = None,
    emit_event=lambda event: None,
    recorder=None,
) -> TurnLoopDependencies:
    return TurnLoopDependencies(
        model_gateway=SimpleNamespace(),
        writer=writer or _FakeWriter(),
        recorder=recorder or SimpleNamespace(transition=lambda status: None, finish=lambda status: None, state_history=[]),
        router=router,
        task_goal="test",
        allowed_tools=["agent", "file_read"],
        tool_registry=SimpleNamespace(allowed_definitions=lambda names: []),
        verification_commands=[],
        workspace_root=object(),
        max_turns=3,
        raise_if_cancelled=lambda: None,
        emit_event=emit_event,
        microcompact_old_tool_messages=lambda messages, writer, turn, emit_event: None,
        interaction_handler=None,
        interaction_resolver=SimpleNamespace(),
        safety_guard=SimpleNamespace(check=lambda name, args, result: None),
        interaction_bridge_factory=lambda turn, resolver: None,
        record_guardrail=lambda violation, turn: None,
        record_suggestion=lambda turn, suggestion: None,
        tool_error_is_terminal=lambda result: False,
        update_in_band_verification_progress=lambda name, args, result, commands, passed: None,
        all_declared_verification_commands_passed=lambda commands, passed: False,
        successful_file_change_without_declared_verification=lambda observations, commands: False,
        verification_observation=lambda result: {},
        verification_evidence=lambda result: "",
        verification_loop_limit_evidence=lambda max_turns, result: "",
    )


def _replace_dep(deps: TurnLoopDependencies, field_name: str, value: Any) -> TurnLoopDependencies:
    values = {name: getattr(deps, name) for name in deps.__dataclass_fields__}
    values[field_name] = value
    return TurnLoopDependencies(**values)
