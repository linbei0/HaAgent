from __future__ import annotations

from pathlib import Path

from haagent.models.gateway import ModelResponse
from haagent.runtime.execution.cancellation import CancellationToken
from haagent.runtime.session.turn import ChatTurnRequest, ChatTurnRunner
from haagent.runtime.execution.path_policy import default_path_policy
from haagent.runtime.orchestration.recorder import RunResult
from haagent.runtime.orchestration.state import RunStatus
from haagent.runtime.contracts.task import load_task
from haagent.tools.registry import ToolDefinition, default_tool_runtime_registry


class _Gateway:
    provider_name = "test"

    def generate(self, messages, tool_schemas):
        return ModelResponse("done", [])


class _Orchestrator:
    def __init__(self, captured: dict[str, object], **kwargs) -> None:
        self._captured = captured
        self._captured.update(kwargs)

    def run(self, task_path: Path) -> RunResult:
        self._captured["task"] = load_task(task_path)
        return RunResult(RunStatus.COMPLETED, [RunStatus.CREATED, RunStatus.COMPLETED], task_path.parent)


def test_chat_turn_runner_writes_task_and_calls_orchestrator(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    result = ChatTurnRunner().run(
        ChatTurnRequest(
            prompt="  summarize docs  ",
            workspace_root=tmp_path,
            runs_root=tmp_path / ".runs",
            model_gateway=_Gateway(),
            max_turns=3,
            session_summary="summary",
            session_compaction={"decision": "kept"},
            tool_result_microcompact_count=2,
            working_state={"current_goal": "", "key_findings": [], "completed_actions": [], "next_steps": [], "last_updated_turn": 0},
            path_policy=default_path_policy(tmp_path),
            enable_web=True,
            target_paths=["README.md"],
            event_sink=lambda event: None,
            interaction_handler=None,
            cancellation_token=CancellationToken(),
            orchestrator_factory=lambda **kwargs: _Orchestrator(captured, **kwargs),
        ),
    )

    task = captured["task"]
    assert task.goal == "summarize docs"
    assert task.workspace_root == str(tmp_path.resolve())
    assert task.target_paths == ["README.md"]
    assert "web_search" in task.allowed_tools
    assert captured["session_summary"] == "summary"
    assert captured["tool_result_microcompact_count"] == 2
    assert result.status == RunStatus.COMPLETED


def test_chat_turn_runner_allows_dynamic_mcp_tool_in_task_contract(tmp_path: Path) -> None:
    dynamic = ToolDefinition(
        name="mcp__fixture__echo",
        description="Echo text",
        risk_level="high",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    registry = default_tool_runtime_registry({"mcp__fixture__echo": dynamic})
    captured: dict[str, object] = {}

    result = ChatTurnRunner().run(
        ChatTurnRequest(
            prompt="use echo",
            workspace_root=tmp_path,
            runs_root=tmp_path / ".runs",
            model_gateway=_Gateway(),
            max_turns=1,
            session_summary=None,
            session_compaction=None,
            tool_result_microcompact_count=0,
            working_state=None,
            path_policy=default_path_policy(tmp_path),
            enable_web=False,
            target_paths=[],
            event_sink=lambda event: None,
            interaction_handler=None,
            cancellation_token=CancellationToken(),
            orchestrator_factory=lambda **kwargs: _Orchestrator(captured, **kwargs),
            tool_registry=registry,
            mcp_tool_names=["mcp__fixture__echo"],
        ),
    )

    task = captured["task"]
    assert "mcp__fixture__echo" in task.allowed_tools
    assert "list_mcp_resources" in task.allowed_tools
    assert "read_mcp_resource" in task.allowed_tools
    assert captured["tool_registry"] is registry
    assert result.status == RunStatus.COMPLETED
