"""
tests/unit/runtime/test_run_turns.py - 单轮工具执行循环测试

验证工具 observation 写入 trace 与模型可见消息之间的分层行为。
"""

from types import SimpleNamespace
from typing import Any
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from haagent.models.types import ToolCall
from haagent.models.types import ModelResponse
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
    def __init__(self, results: dict[str, dict[str, Any]], delays: dict[str, float] | None = None) -> None:
        self.results = results
        self.delays = delays or {}
        self.calls: list[str] = []

    def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None) -> dict[str, Any]:
        self.calls.append(tool_name)
        if interaction_handler is not None and args.get("interact"):
            interaction_handler(SimpleNamespace(tool_name=tool_name))
        time.sleep(self.delays.get(tool_name, 0.0))
        return dict(self.results[tool_name])

    def raise_for_error(self, result: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected terminal error: {result}")

    def wait_for_agent_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        raise AssertionError(f"unexpected worker wait: {task_id}, timeout={timeout}")


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


def test_parallel_tool_failure_does_not_skip_sibling_tool_result() -> None:
    router = _PerToolRouter(
        {
            "file_read": {
                "status": "error",
                "error": {"type": "tool_argument_invalid", "message": "missing path"},
            },
            "grep": {"status": "success", "content": "grep result"},
        }
    )
    state = TurnLoopState(messages=[], context_id="ctx")
    deps = _deps(router=router)

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
        tool_registry=SimpleNamespace(),
        verification_commands=[],
        workspace_root=object(),
        max_turns=3,
        raise_if_cancelled=lambda: None,
        emit_event=lambda event: None,
        compress_historical_tool_messages=lambda messages, writer, turn, emit_event: None,
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
        compress_historical_tool_messages=lambda messages, writer, turn, emit_event: None,
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

