"""
tests/unit/runtime/test_run_turns.py - 单轮工具执行循环测试

验证工具 observation 写入 trace 与模型可见消息之间的分层行为。
"""

from types import SimpleNamespace
from typing import Any, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import threading
import time

import pytest

from haagent.models.gateway_retry import execute_model_request
from haagent.models.types import ModelCallError, ModelFailureDetails, ModelUsage, ToolCall
from haagent.models.types import ModelResponse
from haagent.runtime.execution.retry import RetryController, RetryPolicy
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.orchestration.turns import (
    TurnLoopDependencies,
    TurnLoopState,
    _handle_no_tool_response,
    _run_tool_calls,
    run_turn_loop,
)
from haagent.runtime.performance import PerformanceTrace
from haagent.tools.registry import ToolDefinition, default_tool_runtime_registry


class _FakeRouter:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.waited_task_ids: list[str] = []
        self.skipped: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None, *, turn=None) -> dict[str, Any]:
        del tool_name, args, interaction_handler, turn
        return dict(self.result)

    def record_skipped(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        self.skipped.append((tool_name, dict(args), dict(result)))
        return result

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
        self._temporary_directory: TemporaryDirectory[str] | None = None
        if path is None:
            self._temporary_directory = TemporaryDirectory(prefix="haagent-turns-")
            self.path = Path(self._temporary_directory.name)
        else:
            self.path = path
        self.transcript: list[dict[str, Any]] = []
        self.failure_records: list[dict[str, Any] | None] = []

    def append_transcript(self, record: dict[str, Any]) -> None:
        self.transcript.append(record)

    def append_tool_call(self, record: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected skipped tool call: {record}")

    def append_model_usage(self, **kwargs) -> None:
        self.model_usage = kwargs

    def write_failure_attribution(self, record: dict[str, Any] | None) -> None:
        self.failure_records.append(record)


class _PerToolRouter:
    def __init__(
        self,
        results: dict[str, dict[str, Any]],
        delays: dict[str, float] | None = None,
        *,
        track_active: bool = False,
    ) -> None:
        self.results = results
        self.delays = delays or {}
        self.calls: list[str] = []
        self.skipped: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.start_order: list[str] = []
        self.end_order: list[str] = []
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.track_active = track_active
        self.call_starts: dict[str, float] = {}
        self.call_ends: dict[str, float] = {}

    def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None, *, turn=None) -> dict[str, Any]:
        del turn
        with self._lock:
            self.calls.append(tool_name)
            self.start_order.append(tool_name)
            self.call_starts[tool_name] = time.perf_counter()
            if self.track_active:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
        if interaction_handler is not None and args.get("interact"):
            interaction_handler(SimpleNamespace(tool_name=tool_name))
        delay_key = str(args.get("delay_key") or tool_name)
        time.sleep(self.delays.get(delay_key, self.delays.get(tool_name, 0.0)))
        result_key = str(args.get("result_key") or tool_name)
        result = dict(self.results[result_key])
        with self._lock:
            self.end_order.append(tool_name)
            self.call_ends[tool_name] = time.perf_counter()
            if self.track_active:
                self.active -= 1
        return result

    def record_skipped(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        self.skipped.append((tool_name, dict(args), dict(result)))
        return result

    def raise_for_error(self, result: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected terminal error: {result}")

    def wait_for_agent_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        raise AssertionError(f"unexpected worker wait: {task_id}, timeout={timeout}")


def test_read_batch_respects_max_parallel_read_tools() -> None:
    results = {f"file_read_{index}": {"status": "success", "content": str(index)} for index in range(10)}
    delays = {f"file_read_{index}": 0.12 for index in range(10)}
    router = _PerToolRouter(results, delays=delays, track_active=True)
    # unique names via delay_key/result_key but real tool name file_read
    tool_calls = [
        ToolCall(
            name="file_read",
            args={"path": f"{index}.txt", "delay_key": f"file_read_{index}", "result_key": f"file_read_{index}"},
            id=f"call_{index}",
        )
        for index in range(10)
    ]
    deps = _deps(router=router, max_parallel_read_tools=4)
    _run_tool_calls(turn=1, tool_calls_with_ids=tool_calls, state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)
    assert router.max_active <= 4
    assert router.max_active > 1
    assert len(router.calls) == 10


def test_serial_barrier_between_write_and_following_read() -> None:
    router = _PerToolRouter(
        {
            "file_read": {"status": "success", "content": "r"},
            "grep": {"status": "success", "content": "g"},
            "file_write": {"status": "success", "content": "w"},
            "shell": {"status": "success", "content": "s"},
        },
        delays={"file_read": 0.08, "grep": 0.08, "file_write": 0.08, "shell": 0.01},
        track_active=True,
    )
    # second file_read distinguished by result_key after write
    results = {
        "file_read_a": {"status": "success", "content": "a"},
        "grep": {"status": "success", "content": "g"},
        "file_write": {"status": "success", "content": "w"},
        "file_read_b": {"status": "success", "content": "b"},
        "shell": {"status": "success", "content": "s"},
    }
    delays = {
        "file_read_a": 0.08,
        "grep": 0.08,
        "file_write": 0.08,
        "file_read_b": 0.05,
        "shell": 0.01,
    }
    router = _PerToolRouter(results, delays=delays, track_active=True)
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "a.txt", "delay_key": "file_read_a", "result_key": "file_read_a"}, id="c1"),
            ToolCall(name="grep", args={"pattern": "x", "delay_key": "grep", "result_key": "grep"}, id="c2"),
            ToolCall(name="file_write", args={"path": "c.txt", "delay_key": "file_write", "result_key": "file_write"}, id="c3"),
            ToolCall(name="file_read", args={"path": "c.txt", "delay_key": "file_read_b", "result_key": "file_read_b"}, id="c4"),
            ToolCall(name="shell", args={"command": "echo", "delay_key": "shell", "result_key": "shell"}, id="c5"),
        ],
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=_deps(router=router),
    )
    # write must start after first batch ends; second read after write
    assert router.start_order[:3] == ["file_read", "grep", "file_write"] or (
        set(router.start_order[:2]) == {"file_read", "grep"}
        and router.start_order[2] == "file_write"
    )
    write_start = router.call_starts["file_write"]
    assert write_start >= router.call_ends.get("grep", write_start) - 0.05
    assert router.start_order.index("shell") > router.start_order.index("file_write")
    # second file_read is after write in start_order (indices 3 then maybe shell)
    assert router.start_order.count("file_read") == 2
    first_read_idx = router.start_order.index("file_read")
    second_read_idx = router.start_order.index("file_read", first_read_idx + 1)
    write_idx = router.start_order.index("file_write")
    assert first_read_idx < write_idx < second_read_idx


def test_workspace_writes_do_not_overlap() -> None:
    router = _PerToolRouter(
        {
            "file_write_a": {"status": "success"},
            "file_write_b": {"status": "success"},
        },
        delays={"file_write_a": 0.12, "file_write_b": 0.12},
        track_active=True,
    )
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(
                name="file_write",
                args={"path": "a.txt", "delay_key": "file_write_a", "result_key": "file_write_a"},
                id="w1",
            ),
            ToolCall(
                name="file_write",
                args={"path": "b.txt", "delay_key": "file_write_b", "result_key": "file_write_b"},
                id="w2",
            ),
        ],
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=_deps(router=router),
    )
    assert router.max_active == 1
    assert router.start_order == ["file_write", "file_write"]


def test_read_batch_failure_skips_unstarted_and_later_calls() -> None:
    results = {
        "r0": {"status": "error", "error": {"type": "boom", "message": "fail"}},
        "r1": {"status": "success", "content": "ok"},
        "r2": {"status": "success", "content": "later"},
        "r3": {"status": "success", "content": "later"},
        "write": {"status": "success"},
    }
    delays = {"r0": 0.05, "r1": 0.15, "r2": 0.05, "r3": 0.05, "write": 0.01}
    router = _PerToolRouter(results, delays=delays, track_active=True)
    state = TurnLoopState(messages=[], context_id="ctx")
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "0", "delay_key": "r0", "result_key": "r0"}, id="c0"),
            ToolCall(name="file_read", args={"path": "1", "delay_key": "r1", "result_key": "r1"}, id="c1"),
            ToolCall(name="file_read", args={"path": "2", "delay_key": "r2", "result_key": "r2"}, id="c2"),
            ToolCall(name="file_read", args={"path": "3", "delay_key": "r3", "result_key": "r3"}, id="c3"),
            ToolCall(name="file_write", args={"path": "w", "delay_key": "write", "result_key": "write"}, id="cw"),
        ],
        state=state,
        deps=_deps(router=router, max_parallel_read_tools=2),
    )
    assert "file_write" not in router.calls
    assert len(router.skipped) >= 1
    skipped_ids = [
        message["tool_call_id"]
        for message in state.messages
        if message.get("tool_call_id") and "tool_call_skipped" in message.get("content", "")
    ]
    assert skipped_ids
    assert any(item[2].get("execution_state") == "not_started" for item in router.skipped)


def test_serial_write_failure_skips_later_calls() -> None:
    router = _PerToolRouter(
        {
            "file_write": {
                "status": "error",
                "error": {"type": "write_failed", "message": "nope"},
            },
            "file_read": {"status": "success", "content": "x"},
            "shell": {"status": "success"},
        }
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_write", args={"path": "a.txt"}, id="w"),
            ToolCall(name="file_read", args={"path": "a.txt"}, id="r"),
            ToolCall(name="shell", args={"command": "echo"}, id="s"),
        ],
        state=state,
        deps=_deps(router=router),
    )
    assert router.calls == ["file_write"]
    assert [name for name, _, _ in router.skipped] == ["file_read", "shell"]


def test_not_started_emits_tool_failed_without_tool_started() -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    router = _PerToolRouter(
        {
            "file_write": {
                "status": "error",
                "error": {"type": "write_failed", "message": "nope"},
            },
            "file_read": {"status": "success"},
        }
    )
    events: list[object] = []
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_write", args={"path": "a.txt"}, id="w"),
            ToolCall(name="file_read", args={"path": "a.txt"}, id="r"),
        ],
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=_deps(router=router, emit_event=events.append),
    )
    payloads = [bus_event_to_dict(event) for event in events]
    started_tools = [p["tool_name"] for p in payloads if p.get("event_type") == "tool_started"]
    failed = [p for p in payloads if p.get("event_type") == "tool_failed"]
    assert started_tools == ["file_write"]
    not_started = [p for p in failed if p.get("execution_state") == "not_started"]
    assert len(not_started) == 1
    assert not_started[0]["tool_name"] == "file_read"


def test_high_risk_read_only_tool_is_serial() -> None:
    high_read = ToolDefinition(
        name="risky_read",
        description="high risk read",
        risk_level="high",
        parameters={"type": "object", "properties": {}, "required": []},
        execution_effect="read_only",
    )
    registry = default_tool_runtime_registry({"risky_read": high_read})
    router = _PerToolRouter(
        {
            "risky_read": {"status": "success", "content": "x"},
        },
        delays={"risky_read": 0.15},
        track_active=True,
    )
    # two same-name tools with delay keys
    results = {"a": {"status": "success"}, "b": {"status": "success"}}
    delays = {"a": 0.15, "b": 0.15}
    router = _PerToolRouter(results, delays=delays, track_active=True)
    started = time.perf_counter()
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="risky_read", args={"delay_key": "a", "result_key": "a"}, id="a"),
            ToolCall(name="risky_read", args={"delay_key": "b", "result_key": "b"}, id="b"),
        ],
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=_deps(
            router=router,
            allowed_tools=["risky_read"],
            tool_registry=registry,
            max_parallel_read_tools=4,
        ),
    )
    elapsed = time.perf_counter() - started
    assert router.max_active == 1
    assert elapsed >= 0.28


def test_same_turn_multiple_tool_calls_execute_concurrently() -> None:
    router = _PerToolRouter(
        {
            "file_read": {"status": "success", "content": "first"},
            "grep": {"status": "success", "content": "second"},
        },
        delays={"file_read": 0.25, "grep": 0.25},
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router)

    started = time.perf_counter()
    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "a.txt"}, id="call_1"),
            ToolCall(name="grep", args={"pattern": "needle", "root": "."}, id="call_2"),
        ],
        state=state,
        deps=deps,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.40
    assert sorted(router.calls) == ["file_read", "grep"]


def test_fake_writer_uses_an_isolated_temporary_directory() -> None:
    writer = _FakeWriter()

    assert writer.path != Path.cwd()


def test_turn_loop_emits_key_step_progress_events() -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    router = _PerToolRouter(
        {
            "file_read": {"status": "success", "content": "notes"},
        },
    )
    emitted_events: list[object] = []
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router, emit_event=emitted_events.append)

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[ToolCall(name="file_read", args={"path": "notes.txt"}, id="call_1")],
        state=state,
        deps=deps,
    )

    progress_events = [
        bus_event_to_dict(event)
        for event in emitted_events
        if bus_event_to_dict(event).get("event_type") == "task_step_progress"
    ]
    assert [event["category"] for event in progress_events] == ["tool_batch_finished"]
    assert progress_events[0]["step_id"] == "step-001"
    assert progress_events[0]["evidence_count"] == 1


def test_turn_loop_emits_model_turn_progress_event(tmp_path: Path) -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    emitted_events: list[object] = []
    finish_statuses: list[object] = []

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            del messages, tool_schemas
            return ModelResponse(content="done", tool_calls=[])

    deps = _deps(
        router=_FakeRouter({}),
        writer=_FakeWriter(tmp_path / "episode"),
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    progress_events = [
        bus_event_to_dict(event)
        for event in emitted_events
        if bus_event_to_dict(event).get("event_type") == "task_step_progress"
    ]
    assert progress_events[0]["category"] == "model_turn_started"
    assert progress_events[0]["step_id"] == "step-001"


def test_turn_loop_emits_intermediate_assistant_content_before_tool_followup(tmp_path: Path) -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    emitted_events: list[object] = []
    intermediate_content = "阶段分析\n\n## 发现\n- 事件映射保留完整内容\n- 工具调用后还需继续验证"
    responses = iter(
        [
            ModelResponse(
                content=intermediate_content,
                tool_calls=[ToolCall(name="file_read", args={"path": "README.md"})],
            ),
            ModelResponse(content="最终总结", tool_calls=[]),
        ],
    )

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            del messages, tool_schemas
            return next(responses)

    state_history = [RunStatus.PLANNING]

    def transition(status: RunStatus) -> None:
        state_history.append(status)

    deps = _deps(
        router=_FakeRouter({"status": "success", "content": "read"}),
        writer=_FakeWriter(tmp_path / "episode"),
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=state_history,
            transition=transition,
            finish=lambda status: SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    assistant_events = [
        bus_event_to_dict(event)
        for event in emitted_events
        if bus_event_to_dict(event).get("event_type")
        in {"assistant_intermediate_message", "assistant_message"}
    ]
    assert assistant_events == [
        {
            "event_type": "assistant_intermediate_message",
            "turn": 1,
            "content": intermediate_content,
        },
        {
            "event_type": "assistant_message",
            "turn": 2,
            "content": "最终总结",
        },
    ]


def test_turn_loop_preserves_short_tool_narration_as_process_output(tmp_path: Path) -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    emitted_events: list[object] = []
    responses = iter(
        [
            ModelResponse(
                content="现在让我继续读取关键文件：",
                tool_calls=[ToolCall(name="file_read", args={"path": "README.md"})],
            ),
            ModelResponse(content="最终总结", tool_calls=[]),
        ],
    )

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            del messages, tool_schemas
            return next(responses)

    state_history = [RunStatus.PLANNING]

    def transition(status: RunStatus) -> None:
        state_history.append(status)

    deps = _deps(
        router=_FakeRouter({"status": "success", "content": "read"}),
        writer=_FakeWriter(tmp_path / "episode"),
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=state_history,
            transition=transition,
            finish=lambda status: SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    event_types = [bus_event_to_dict(event).get("event_type") for event in emitted_events]
    assert "assistant_intermediate_message" in event_types
    assert "assistant_message" in event_types


def test_turn_loop_keeps_full_response_final_when_only_memory_settlement_tool_is_called(tmp_path: Path) -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    emitted_events: list[object] = []
    model_calls = 0

    class _ModelGateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas):
            nonlocal model_calls
            del messages, tool_schemas
            model_calls += 1
            if model_calls > 1:
                raise AssertionError("memory settlement must not force a redundant final model turn")
            return ModelResponse(
                content="完整审查报告\n\n## 结论\n应优先修复事件链路。",
                tool_calls=[ToolCall(name="start_memory_update", args={"reason": "记录审查结论"})],
            )

    state_history = [RunStatus.PLANNING]

    def transition(status: RunStatus) -> None:
        state_history.append(status)

    deps = _deps(
        router=_FakeRouter(
            {
                "status": "success",
                "memory_update_requested": True,
                "reason": "记录审查结论",
            },
        ),
        writer=_FakeWriter(tmp_path / "episode"),
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=state_history,
            transition=transition,
            finish=lambda status: SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "workspace_root", tmp_path)

    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    assert model_calls == 1
    assistant_events = [
        bus_event_to_dict(event)
        for event in emitted_events
        if bus_event_to_dict(event).get("event_type").startswith("assistant_")
        and bus_event_to_dict(event).get("event_type") != "assistant_delta"
    ]
    assert assistant_events == [
        {
            "event_type": "assistant_message",
            "turn": 1,
            "content": "完整审查报告\n\n## 结论\n应优先修复事件链路。",
        },
    ]


def test_tool_not_allowed_does_not_abort_turn_as_terminal() -> None:
    """误调未授权工具名应写入 observation 并继续，而不是 raise_for_error 终态失败。"""
    from haagent.runtime.orchestration.orchestrator import _tool_error_is_terminal

    class _RaiseTrackingRouter(_FakeRouter):
        def __init__(self) -> None:
            super().__init__(
                {
                    "status": "error",
                    "error": {
                        "type": "tool_not_allowed",
                        "message": "tool is not allowed: read_file",
                    },
                }
            )
            self.raised: list[dict[str, Any]] = []

        def raise_for_error(self, result: dict[str, Any]) -> None:
            self.raised.append(dict(result))
            raise AssertionError(f"unexpected terminal error: {result}")

    router = _RaiseTrackingRouter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _replace_dep(
        _deps(router=router),
        "tool_error_is_terminal",
        _tool_error_is_terminal,
    )

    early = _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="read_file", args={"path": "a.txt"}, id="call_bad"),
        ],
        state=state,
        deps=deps,
    )

    assert early is None
    assert router.raised == []
    assert any(
        message.get("tool_call_id") == "call_bad" and "tool_not_allowed" in message.get("content", "")
        for message in state.messages
    )


def test_parallel_tool_failure_does_not_skip_sibling_tool_result() -> None:
    # 两个都已 in-flight 时 sibling 仍应完成；用 delay 保证并发启动
    router = _PerToolRouter(
        {
            "file_read": {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "message": "missing path"},
            },
            "grep": {"status": "success", "content": "grep result"},
        },
        delays={"file_read": 0.08, "grep": 0.08},
        track_active=True,
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router, max_parallel_read_tools=2)

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={}, id="call_error"),
            ToolCall(name="grep", args={"pattern": "needle", "root": "."}, id="call_success"),
        ],
        state=state,
        deps=deps,
    )

    assert sorted(router.calls) == ["file_read", "grep"]
    assert router.max_active == 2
    assert len(state.messages) == 3
    assert state.messages[0]["tool_call_id"] == "call_error"
    assert state.messages[1]["tool_call_id"] == "call_success"
    assert "tool_argument_invalid" in state.messages[0]["content"]
    assert "grep result" in state.messages[1]["content"]
    assert state.messages[2]["role"] == "user"


def test_tool_failure_emits_recovery_suggestion_event() -> None:
    router = _PerToolRouter(
        {
            "file_read": {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "message": "missing path"},
            },
        }
    )
    from haagent.runtime.events.bus import bus_event_to_dict

    state = TurnLoopState(messages=[], context_id="ctx")
    emitted_events: list[object] = []
    deps = _deps(router=router, emit_event=emitted_events.append)

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[ToolCall(name="file_read", args={}, id="call_error")],
        state=state,
        deps=deps,
    )

    payloads = [bus_event_to_dict(event) for event in emitted_events]
    assert [event["event_type"] for event in payloads] == [
        "tool_started",
        "tool_failed",
        "task_recovery_suggested",
    ]
    recovery = payloads[-1]
    assert recovery["step_id"] == "step-001"
    assert recovery["category"] == "model_format_error"
    assert recovery["suggested_action"] == "correct_tool_arguments"


def test_verification_failure_emits_checkpoint_recovery_and_budget_warning() -> None:
    from haagent.runtime.events.bus import bus_event_to_dict

    emitted_events: list[object] = []
    transitions: list[object] = []
    state = TurnLoopState(messages=[], context_id="ctx")
    verification_result = SimpleNamespace(
        status="failed",
        failed_command="uv run pytest -q",
        timeout=False,
        failure_reason="exit_code",
        exit_code=1,
        stdout_excerpt="failure details",
        stderr_excerpt="",
    )
    deps = _deps(
        router=_FakeRouter({}),
        emit_event=emitted_events.append,
        recorder=SimpleNamespace(
            state_history=[RunStatus.EXECUTING],
            transition=transitions.append,
            finish=lambda status: SimpleNamespace(status=status),
        ),
    )
    deps = _replace_dep(deps, "verification_commands", ["uv run pytest -q"])
    deps = _replace_dep(deps, "max_turns", 2)
    state.verification_engine = SimpleNamespace(run=lambda commands: verification_result)

    result = _handle_no_tool_response(
        turn=2,
        model_response=ModelResponse(content="done", tool_calls=[]),
        state=state,
        deps=deps,
    )

    assert result is not None
    payloads = [bus_event_to_dict(event) for event in emitted_events]
    event_types = [event["event_type"] for event in payloads]
    assert "task_checkpoint_saved" in event_types
    assert "task_recovery_suggested" in event_types
    assert "task_budget_warning" in event_types
    recovery = next(event for event in payloads if event["event_type"] == "task_recovery_suggested")
    assert recovery["category"] == "verification_failed"
    assert recovery["suggested_action"] == "repair_and_rerun_verification"


def test_parallel_tool_results_keep_original_tool_call_order() -> None:
    router = _PerToolRouter(
        {
            "file_read": {"status": "success", "content": "slow first"},
            "grep": {"status": "success", "content": "fast second"},
        },
        delays={"file_read": 0.20, "grep": 0.01},
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router)

    _run_tool_calls(
        turn=1,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "a.txt"}, id="call_slow"),
            ToolCall(name="grep", args={"pattern": "needle", "root": "."}, id="call_fast"),
        ],
        state=state,
        deps=deps,
    )

    tool_result_messages = [message for message in state.messages if "tool_call_id" in message]
    assert [message["tool_call_id"] for message in tool_result_messages] == ["call_slow", "call_fast"]


def test_parallel_tool_calls_share_one_interaction_bridge() -> None:
    router = _PerToolRouter(
        {
            "file_read": {"status": "success", "content": "first"},
            "grep": {"status": "success", "content": "second"},
        }
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    bridge_calls: list[int] = []
    interaction_calls: list[str] = []

    def bridge_factory(turn: int, resolver):
        del resolver
        bridge_calls.append(turn)

        def handle(request):
            interaction_calls.append(request.tool_name)
            return SimpleNamespace(approved=True)

        return handle

    deps = _replace_dep(_deps(router=router), "interaction_handler", object())
    deps = _replace_dep(deps, "interaction_bridge_factory", bridge_factory)

    _run_tool_calls(
        turn=3,
        tool_calls_with_ids=[
            ToolCall(name="file_read", args={"path": "a.txt", "interact": True}, id="call_1"),
            ToolCall(name="grep", args={"pattern": "needle", "root": ".", "interact": True}, id="call_2"),
        ],
        state=state,
        deps=deps,
    )

    assert bridge_calls == [3]
    assert sorted(interaction_calls) == ["file_read", "grep"]


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
        tool_registry=default_tool_runtime_registry(),
        verification_commands=[],
        workspace_root=object(),
        max_turns=3,
        raise_if_cancelled=lambda: None,
        emit_event=lambda event: None,
        compress_historical_tool_messages=lambda messages, writer, turn, emit_event: None,
        interaction_handler=None,
        interaction_resolver=SimpleNamespace(),
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


def test_turn_loop_adds_loaded_image_attachment_to_next_model_call(tmp_path: Path) -> None:
    image_path = tmp_path / "attachments" / "img-one.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"image-bytes")
    loaded_attachment = {
        "id": "img-one",
        "filename": "img-one.png",
        "mime_type": "image/png",
        "size_bytes": 11,
        "width": 2,
        "height": 1,
        "sha256": "a" * 64,
        "relative_path": "attachments/img-one.png",
        "path": str(image_path),
    }
    router = _FakeRouter(
        {
            "status": "success",
            "loaded_image_attachment": loaded_attachment,
            "model_visible": {
                "message": "图片已加载，将在下一次模型调用中作为视觉输入。",
                "image_id": "img-one",
            },
        }
    )
    writer = _FakeWriter()
    model_messages: list[list[dict[str, Any]]] = []
    finish_statuses: list[object] = []

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
                            name="load_image_attachment",
                            args={"image_id": "img-one"},
                            id="call_image",
                        )
                    ],
                )
            return ModelResponse(content="loaded image observed", tool_calls=[])

    deps = _deps(
        router=router,
        writer=writer,
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _ModelGateway())
    deps = _replace_dep(deps, "allowed_tools", ["load_image_attachment"])
    deps = _replace_dep(deps, "all_declared_verification_commands_passed", lambda commands, passed: True)

    result = run_turn_loop(
        state=TurnLoopState(messages=[], context_id="ctx"),
        deps=deps,
    )

    assert result is not None
    assert len(model_messages) == 2
    second_call_content = model_messages[1][-1]["content"]
    assert isinstance(second_call_content, list)
    assert second_call_content[0] == {
        "type": "text",
        "text": "Loaded historical image: img-one",
    }
    assert second_call_content[1]["type"] == "image_attachment"
    assert second_call_content[1]["relative_path"] == "attachments/img-one.png"
    assert second_call_content[1]["path"] == str(image_path)


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


def test_orchestrator_writes_performance_json_for_tool_success(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    (tmp_path / "alpha.txt").write_text("needle appears here", encoding="utf-8")
    task_path.write_text(
        """
goal: Read a file and finish
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    class _Gateway:
        provider_name = "performance-tool"

        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            messages,
            tool_schemas,
            event_sink=None,
            cancellation_token=None,
            retry_event_sink=None,
            retry_exhausted_sink=None,
            telemetry_sink=None,
        ):
            del tool_schemas, event_sink, cancellation_token, retry_event_sink, retry_exhausted_sink
            self.calls += 1
            if telemetry_sink is not None:
                telemetry_sink(
                    __import__("haagent.models.telemetry", fromlist=["ModelTransportEvent"]).ModelTransportEvent(
                        kind="attempt_started",
                        attempt=1,
                        elapsed_ms=0.0,
                    ),
                )
                telemetry_sink(
                    __import__("haagent.models.telemetry", fromlist=["ModelTransportEvent"]).ModelTransportEvent(
                        kind="attempt_finished",
                        attempt=1,
                        elapsed_ms=5.0,
                    ),
                )
            if self.calls == 1:
                return ModelResponse(
                    "",
                    [ToolCall("file_read", {"path": "alpha.txt"})],
                    usage=ModelUsage(input_tokens=120, output_tokens=4, total_tokens=124, raw_source="test"),
                )
            return ModelResponse("done", [], usage=ModelUsage(input_tokens=80, output_tokens=2, total_tokens=82, raw_source="test"))

        def metadata(self):
            return SimpleNamespace(provider="performance-tool", model="m", endpoint=None, base_url=None, profile_name=None)

        def capabilities(self):
            from haagent.models.capabilities import ModelCapabilities

            return ModelCapabilities(
                tools="supported",
                streaming="supported",
                vision="unknown",
                reasoning="unknown",
                tools_mode="native",
                protocols=frozenset({"chat_completions"}),
            )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=_Gateway(),
        max_turns=3,
    ).run(task_path)

    performance_path = result.episode_path / "performance.json"
    assert performance_path.exists()
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    assert performance["performance_schema_version"] == "1.0"
    assert performance["status"] == "completed"
    assert performance["model_turns"][0]["turn"] == 1
    assert performance["model_turns"][0]["input_tokens"] == 120
    prefix = performance["model_turns"][0]["stable_prefix_fingerprint"]
    assert prefix.startswith("sha256:")
    assert prefix != "sha256:" + ("0" * 64)
    assert performance["cache_diagnostics"]["tool_schema"]["status"] in {"hit", "miss"}
    assert performance["tools"][0]["tool_name"] == "file_read"
    assert performance["tools"][0]["status"] == "success"


def test_orchestrator_writes_performance_json_after_retry_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Fail after retry
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    class _RetryThenFailGateway:
        provider_name = "performance-retry-fail"

        def __init__(self) -> None:
            self._retry_controller = RetryController(
                RetryPolicy(max_attempts=2, minimum_delay_seconds=0, base_delay_seconds=0),
                sleep=lambda _: None,
            )
            self._attempts = 0

        def generate(
            self,
            messages,
            tool_schemas,
            event_sink=None,
            cancellation_token=None,
            retry_event_sink=None,
            retry_exhausted_sink=None,
            telemetry_sink=None,
        ):
            del messages, tool_schemas

            def invoke(on_delta: Callable[[str], None] | None, attempt: int) -> ModelResponse:
                del on_delta
                self._attempts = attempt
                raise ModelCallError(
                    "temporary upstream failure",
                    details=ModelFailureDetails(category="server", status_code=500, retryable=True),
                )

            return execute_model_request(
                self._retry_controller,
                provider=self.provider_name,
                invoke=invoke,
                event_sink=event_sink,
                cancellation_token=cancellation_token,
                retry_event_sink=retry_event_sink,
                retry_exhausted_sink=retry_exhausted_sink,
                telemetry_sink=telemetry_sink,
            )

        def metadata(self):
            return SimpleNamespace(provider=self.provider_name, model="m", endpoint=None, base_url=None, profile_name=None)

        def capabilities(self):
            from haagent.models.capabilities import ModelCapabilities

            return ModelCapabilities(
                tools="supported",
                streaming="supported",
                vision="unknown",
                reasoning="unknown",
                tools_mode="native",
                protocols=frozenset({"chat_completions"}),
            )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=_RetryThenFailGateway(),
        max_turns=1,
    ).run(task_path)

    performance = json.loads((result.episode_path / "performance.json").read_text(encoding="utf-8"))
    assert result.status is RunStatus.FAILED
    assert performance["status"] == "failed"
    assert performance["model_turns"][0]["turn"] == 1
    assert performance["model_turns"][0]["attempt_count"] == 2
    assert [item["attempt"] for item in performance["model_turns"][0]["attempts"]] == [1, 2]
    assert performance["model_turns"][0]["attempts"][0]["status"] == "failed"
    assert performance["model_turns"][0]["attempts"][1]["status"] == "failed"


def test_turn_loop_records_tool_performance_via_router_sink(tmp_path: Path) -> None:
    trace = PerformanceTrace.start()
    recorded: list[tuple[int, str, float, str, str]] = []

    class _Router:
        def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None, *, turn=None):
            del args, interaction_handler
            assert turn == 1
            recorded.append((turn, tool_name, 3.0, "read_only", "success"))
            trace.record_tool(turn, tool_name, 3.0, "read_only", "success")
            return {"status": "success", "content": "ok"}

        def raise_for_error(self, result: dict[str, Any]) -> None:
            del result

        def record_skipped(self, tool_name, args, result):
            return result

        def wait_for_agent_task(self, task_id: str, timeout: float | None = None):
            raise AssertionError(f"unexpected wait {task_id}")

    class _Gateway:
        provider_name = "fake"

        def generate(self, *, messages, tool_schemas, telemetry_sink=None):
            del messages, tool_schemas, telemetry_sink
            return ModelResponse(
                "",
                [ToolCall(name="file_read", args={"path": "README.md"})],
                usage=ModelUsage(input_tokens=120, output_tokens=1, total_tokens=121, raw_source="test"),
            )

    finish_statuses: list[object] = []
    deps = _deps(
        router=_Router(),  # type: ignore[arg-type]
        writer=_FakeWriter(tmp_path / "episode"),
        recorder=SimpleNamespace(
            state_history=[RunStatus.PLANNING],
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status, episode_path="episode"),
        ),
    )
    deps = _replace_dep(deps, "model_gateway", _Gateway())
    deps = _replace_dep(deps, "performance_trace", trace)
    deps = _replace_dep(deps, "max_turns", 1)
    deps = _replace_dep(deps, "tool_error_is_terminal", lambda result: False)

    # first generate returns tool call; second would be needed for completion — force terminal via max_turns
    result = run_turn_loop(state=TurnLoopState(messages=[], context_id="ctx"), deps=deps)

    assert result is not None
    assert recorded and recorded[0][1] == "file_read"
    assert trace.to_dict()["tools"][0]["tool_name"] == "file_read"
    assert trace.to_dict()["model_turns"][0]["input_tokens"] == 120


def _deps(
    *,
    router: _FakeRouter,
    writer: _FakeWriter | None = None,
    emit_event=lambda event: None,
    recorder=None,
    allowed_tools: list[str] | None = None,
    tool_registry=None,
    max_parallel_read_tools: int = 4,
) -> TurnLoopDependencies:
    tools = allowed_tools or [
        "agent",
        "file_read",
        "grep",
        "file_write",
        "shell",
        "code_run",
        "request_user_input",
        "fake_tool",
    ]
    return TurnLoopDependencies(
        model_gateway=SimpleNamespace(),
        writer=writer or _FakeWriter(),
        recorder=recorder or SimpleNamespace(transition=lambda status: None, finish=lambda status: None, state_history=[]),
        router=router,
        task_goal="test",
        allowed_tools=tools,
        tool_registry=tool_registry or default_tool_runtime_registry(),
        verification_commands=[],
        workspace_root=object(),
        max_turns=3,
        raise_if_cancelled=lambda: None,
        emit_event=emit_event,
        compress_historical_tool_messages=lambda messages, writer, turn, emit_event: None,
        interaction_handler=None,
        interaction_resolver=SimpleNamespace(),
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
        max_parallel_read_tools=max_parallel_read_tools,
    )


def _replace_dep(deps: TurnLoopDependencies, field_name: str, value: Any) -> TurnLoopDependencies:
    values = {name: getattr(deps, name) for name in deps.__dataclass_fields__}
    values[field_name] = value
    return TurnLoopDependencies(**values)


def test_progress_guard_warn_injects_single_suggestion() -> None:
    from haagent.runtime.execution.progress_guard import ProgressGuard

    guard = ProgressGuard()
    writer = _FakeWriter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(
        router=_FakeRouter({"status": "success", "model_visible": {"content": "same"}}),
        writer=writer,
    )
    deps = _replace_dep(deps, "progress_guard", guard)
    deps = _replace_dep(deps, "progress_guard_mode", "warn")
    tool_calls = [ToolCall(name="file_read", args={"path": "a.md"}, id="c1")]
    for turn in range(1, 4):
        _run_tool_calls(turn=turn, tool_calls_with_ids=tool_calls, state=state, deps=deps)
    suggestion_msgs = [
        message for message in state.messages if str(message.get("content", "")).startswith("[Suggestion] ProgressGuard")
    ]
    assert len(suggestion_msgs) == 1
    assert any(item.get("event") == "progress_guard_warning" for item in writer.transcript)
    # 第 4 次同模式在 warn 模式下不 block，也不重复刷屏
    _run_tool_calls(turn=4, tool_calls_with_ids=tool_calls, state=state, deps=deps)
    suggestion_msgs = [
        message for message in state.messages if str(message.get("content", "")).startswith("[Suggestion] ProgressGuard")
    ]
    assert len(suggestion_msgs) == 1


def test_progress_guard_block_continue_recovers() -> None:
    from haagent.runtime.execution.human_interaction import HumanInteractionRequest, HumanInteractionResponse
    from haagent.runtime.execution.progress_guard import ProgressGuard

    guard = ProgressGuard()
    blocked_states: list[object] = []
    finish_statuses: list[object] = []
    answers = iter(["continue"])

    def handler(request: HumanInteractionRequest) -> HumanInteractionResponse:
        assert request.tool_name == "progress_guard"
        assert request.args_summary.get("choices") == ["continue", "replan", "stop"]
        return HumanInteractionResponse(approved=True, answer=next(answers))

    writer = _FakeWriter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(
        router=_FakeRouter({"status": "success", "model_visible": {"content": "loop"}}),
        writer=writer,
        recorder=SimpleNamespace(
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status),
            state_history=[RunStatus.EXECUTING],
        ),
    )
    deps = _replace_dep(deps, "progress_guard", guard)
    deps = _replace_dep(deps, "progress_guard_mode", "block")
    deps = _replace_dep(deps, "interaction_handler", handler)
    deps = _replace_dep(deps, "interaction_bridge_factory", lambda turn, resolver: handler)
    deps = _replace_dep(
        deps,
        "on_progress_blocked",
        lambda turn, decision: blocked_states.append((turn, decision.pattern)),
    )
    tool_calls = [ToolCall(name="file_read", args={"path": "x.md"}, id="c1")]
    for turn in range(1, 5):
        result = _run_tool_calls(turn=turn, tool_calls_with_ids=tool_calls, state=state, deps=deps)
        assert result is None
    assert blocked_states
    assert any(item.get("event") == "progress_guard_recovered" for item in writer.transcript)
    assert finish_statuses == []


def test_progress_guard_block_without_handler_fails_loop_limit() -> None:
    from haagent.runtime.execution.progress_guard import ProgressGuard

    guard = ProgressGuard()
    finish_statuses: list[object] = []
    writer = _FakeWriter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(
        router=_FakeRouter({"status": "success", "model_visible": {"content": "loop"}}),
        writer=writer,
        recorder=SimpleNamespace(
            transition=lambda status: None,
            finish=lambda status: finish_statuses.append(status) or SimpleNamespace(status=status),
            state_history=[RunStatus.EXECUTING],
        ),
    )
    deps = _replace_dep(deps, "progress_guard", guard)
    deps = _replace_dep(deps, "progress_guard_mode", "block")
    tool_calls = [ToolCall(name="file_read", args={"path": "x.md"}, id="c1")]
    result = None
    for turn in range(1, 5):
        result = _run_tool_calls(turn=turn, tool_calls_with_ids=tool_calls, state=state, deps=deps)
        if result is not None:
            break
    assert result is not None
    assert finish_statuses == [RunStatus.FAILED]
    assert any(item.get("event") == "progress_guard_blocked" for item in writer.transcript)
    assert writer.failure_records
    assert writer.failure_records[-1]["category"] == "Loop Limit Failure"


def test_progress_block_callback_persists_recoverable_working_state() -> None:
    from haagent.runtime.execution.progress_guard import ProgressDecision
    from haagent.runtime.orchestration.orchestrator import _record_progress_block_working_state

    persisted: list[dict[str, object]] = []
    orchestrator = object.__new__(RunOrchestrator)
    orchestrator._working_state = None
    orchestrator._working_state_sink = lambda value: persisted.append(value)

    _record_progress_block_working_state(
        orchestrator,
        turn=4,
        decision=ProgressDecision(level="block", reason="same observation", pattern="identical_pair"),
    )

    assert persisted
    assert persisted[0]["last_updated_turn"] == 4
    assert "progress_guard blocked" in persisted[0]["next_steps"][0]


def test_progress_guard_skips_running_pending_tools() -> None:
    from haagent.runtime.execution.progress_guard import ProgressGuard

    guard = ProgressGuard()
    writer = _FakeWriter()
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(
        router=_FakeRouter(
            {
                "status": "running",
                "execution_state": "running",
                "task_id": "task-1",
                "model_visible": {"status": "running"},
            }
        ),
        writer=writer,
    )
    deps = _replace_dep(deps, "progress_guard", guard)
    deps = _replace_dep(deps, "progress_guard_mode", "block")
    tool_calls = [ToolCall(name="agent", args={"prompt": "explore"}, id="c1")]
    for turn in range(1, 6):
        _run_tool_calls(turn=turn, tool_calls_with_ids=tool_calls, state=state, deps=deps)
    assert not any(item.get("event") == "progress_guard_warning" for item in writer.transcript)
    assert not any(item.get("event") == "progress_guard_blocked" for item in writer.transcript)

