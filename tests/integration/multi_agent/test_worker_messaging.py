"""
tests/integration/multi_agent/test_worker_messaging.py - worker 运行中通信测试

验证运行中的 worker 可以在 turn 边界消费排队消息。
"""

import threading
from pathlib import Path

from haagent.models.gateway import ModelResponse
from haagent.multi_agent.runtime import MultiAgentRuntime
from haagent.runtime.execution.path_policy import default_path_policy


class _TurnSequencedGateway:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.first_turn_started = threading.Event()
        self.allow_first_turn_finish = threading.Event()

    def generate(self, messages, tool_schemas):
        model_input = "\n".join(
            content
            for content in (message.get("content") for message in messages)
            if isinstance(content, str) and content
        )
        self.calls.append(model_input)
        if len(self.calls) == 1:
            self.first_turn_started.set()
            self.allow_first_turn_finish.wait(timeout=5)
            return ModelResponse(content="first turn", tool_calls=[])
        return ModelResponse(content=f"second turn saw: {model_input}", tool_calls=[])


def test_running_worker_consumes_queued_message_on_next_turn(tmp_path: Path) -> None:
    gateway = _TurnSequencedGateway()
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=gateway,
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
        worker_max_turns=4,
    )
    worker = runtime.spawn_worker(description="Loop", prompt="Keep going", subagent_type="worker")
    assert gateway.first_turn_started.wait(timeout=5)

    queued = runtime.send_message(worker["agent_id"], "direction changed")
    assert queued["status"] == "queued"

    gateway.allow_first_turn_finish.set()
    done = runtime.wait_for_task(worker["task_id"], timeout=5)

    assert done["status"] == "completed"
    output = runtime.task_output(worker["task_id"])
    assert "direction changed" in output["output"]
    assert "direction changed" in gateway.calls[-1]
    assert runtime.store.read_worker_messages(worker["team_id"], worker["agent_id"]) == []
