"""
tests/test_run_turns.py - 单轮工具执行循环测试

验证工具 observation 写入 trace 与模型可见消息之间的分层行为。
"""

from types import SimpleNamespace
from typing import Any

from haagent.models.gateway import ToolCall
from haagent.runtime.run_turns import TurnLoopDependencies, TurnLoopState, _run_tool_calls


class _FakeRouter:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def dispatch(self, tool_name: str, args: dict[str, Any], interaction_handler=None) -> dict[str, Any]:
        del tool_name, args, interaction_handler
        return dict(self.result)

    def raise_for_error(self, result: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected terminal error: {result}")


class _FakeWriter:
    def __init__(self) -> None:
        self.transcript: list[dict[str, Any]] = []

    def append_transcript(self, record: dict[str, Any]) -> None:
        self.transcript.append(record)

    def append_tool_call(self, record: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected skipped tool call: {record}")

    def write_failure_attribution(self, record: dict[str, Any]) -> None:
        raise AssertionError(f"unexpected failure attribution: {record}")


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
