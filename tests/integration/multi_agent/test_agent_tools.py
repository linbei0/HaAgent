"""
tests/integration/multi_agent/test_agent_tools.py - agent 工具集成测试

验证 coordinator 可通过 ToolRouter 启动、继续和停止 in-process worker。
"""

import json
from pathlib import Path
from types import SimpleNamespace

from haagent.models.fake import FakeModelGateway
from haagent.models.types import ModelResponse, ToolCall
from haagent.multi_agent.runtime import MultiAgentRuntime, _failure_summary_from_episode
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.execution.path_policy import default_path_policy
from haagent.tools.router import ToolRouter


class _FileReadGateway:
    provider_name = "fake"

    def generate(self, messages, tool_schemas):
        if not any(message.get("role") == "tool" for message in messages):
            return ModelResponse(
                content="read README",
                tool_calls=[ToolCall(name="file_read", args={"path": "README.md"}, id="call-file-read")],
            )
        return ModelResponse(content="worker read README", tool_calls=[])


class _NeverFinishGateway:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, tool_schemas):
        paths = [".", "tests", "src", "docs", "examples", "src/haagent", "tests/unit"]
        path = paths[self.calls % len(paths)]
        self.calls += 1
        return ModelResponse(
            content="keep reading",
            tool_calls=[ToolCall(name="file_list", args={"path": path, "max_depth": 1}, id=f"call-list-{self.calls}")],
        )


class _SequenceGateway:
    provider_name = "fake"

    def __init__(self, *responses: ModelResponse) -> None:
        self._responses = list(responses)

    def generate(self, messages, tool_schemas):
        if not self._responses:
            return ModelResponse(content="", tool_calls=[])
        return self._responses.pop(0)


def _writer(tmp_path: Path) -> EpisodeWriter:
    task = tmp_path / "task.yaml"
    task.write_text("goal: test\nworkspace_root: .\n", encoding="utf-8")
    return EpisodeWriter.create(tmp_path / ".runs", task)


def test_agent_tool_starts_worker_and_records_trace(tmp_path: Path) -> None:
    gateway = FakeModelGateway(ModelResponse(content="worker done", tool_calls=[]))
    events: list[dict[str, object]] = []
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=gateway,
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_read", "file_list"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=events.append,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=tmp_path / ".haagent" / "teams",
    )
    writer = _writer(tmp_path)
    router = ToolRouter(
        ["agent", "send_message", "task_stop"],
        writer,
        workspace_root=tmp_path,
        path_policy=default_path_policy(tmp_path),
        agent_runtime=runtime,
    )

    result = router.dispatch(
        "agent",
        {
            "description": "Inspect project",
            "prompt": "Say done",
            "subagent_type": "explorer",
            "team": "team-test",
        },
    )

    assert result["status"] == "running"
    assert result["team_id"] == "team-test"
    assert result["agent_id"].startswith("explorer-")
    notification = runtime.wait_for_task(result["task_id"], timeout=5)
    from haagent.runtime.events.bus import bus_event_to_dict

    assert notification["status"] == "completed"
    payloads = [bus_event_to_dict(event) for event in events]
    assert [event["event_type"] for event in payloads] == ["worker_started", "worker_completed"]
    assert payloads[0]["agent_id"] == result["agent_id"]
    assert payloads[1]["status"] == "completed"

    trace_lines = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(trace_lines[0])["tool_name"] == "agent"


def test_send_message_queues_running_worker_message(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-session",
        model_gateway=FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
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
    )
    first = runtime.spawn_worker(
        description="Inspect",
        prompt="Say done",
        subagent_type="explorer",
        team_id="team-test",
    )

    result = runtime.send_message(first["agent_id"], "continue")

    assert result["status"] == "queued"
    assert result["task_id"] == first["task_id"]
    messages = runtime.store.read_worker_messages("team-test", first["agent_id"])
    assert [message.content for message in messages] == ["continue"]


def test_send_message_can_continue_worker_from_new_runtime_for_same_leader(tmp_path: Path) -> None:
    team_root = tmp_path / ".haagent" / "teams"
    common_kwargs = {
        "runs_root": tmp_path / ".runs",
        "workspace_root": tmp_path,
        "leader_session_id": "leader-session-continuation",
        "model_gateway": FakeModelGateway(ModelResponse(content="done", tool_calls=[])),
        "path_policy": default_path_policy(tmp_path),
        "inherited_allowed_tools": ["agent", "send_message", "task_stop", "file_read"],
        "inherited_approval_allowed_tools": [],
        "inherited_approved_tools": [],
        "event_sink": None,
        "interaction_handler": None,
        "enable_web": False,
        "mcp_tool_names": [],
        "tool_registry": None,
        "mcp_runtime": None,
        "team_root": team_root,
    }
    first_runtime = MultiAgentRuntime(**common_kwargs)
    first = first_runtime.spawn_worker(
        description="Inspect",
        prompt="Say done",
        subagent_type="explorer",
        team_id="team-test",
    )
    assert first_runtime.wait_for_task(first["task_id"], timeout=5)["status"] == "completed"
    next_runtime = MultiAgentRuntime(**common_kwargs)

    result = next_runtime.send_message(first["agent_id"], "continue")

    assert result["status"] == "running"
    assert result["agent_id"] == first["agent_id"]
    assert next_runtime.wait_for_task(result["task_id"], timeout=5)["status"] == "completed"


def test_worker_tool_calls_are_written_to_worker_episode(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello worker\n", encoding="utf-8")
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-worker-trace",
        model_gateway=_FileReadGateway(),
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
    )

    started = runtime.spawn_worker(
        description="Read README",
        prompt="Read README.md",
        subagent_type="explorer",
        team_id="team-trace",
    )
    notification = runtime.wait_for_task(started["task_id"], timeout=5)

    worker_episode = Path(str(notification["episode_path"]))
    tool_calls = (worker_episode / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()
    transcript = (worker_episode / "transcript.jsonl").read_text(encoding="utf-8")
    assert any(json.loads(line)["tool_name"] == "file_read" for line in tool_calls)
    assert "hello worker" in transcript


def test_worker_runtime_allows_more_than_three_turns_before_loop_limit(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-worker-turns",
        model_gateway=_NeverFinishGateway(),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "file_list"],
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

    started = runtime.spawn_worker(
        description="Loop until limit",
        prompt="Keep calling file_list",
        subagent_type="explorer",
        team_id="team-turns",
    )
    notification = runtime.wait_for_task(started["task_id"], timeout=10)

    assert notification["status"] == "failed"
    assert "exceeded max_turns=4" in notification["error"]


def test_send_message_restarts_completed_worker_with_same_task_id(tmp_path: Path) -> None:
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-worker-retry",
        model_gateway=_SequenceGateway(
            ModelResponse(content="first success", tool_calls=[]),
            ModelResponse(content="", tool_calls=[]),
        ),
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
    )
    first = runtime.spawn_worker(
        description="Inspect",
        prompt="Say done",
        subagent_type="explorer",
        team_id="team-retry",
    )
    first_notification = runtime.wait_for_task(first["task_id"], timeout=5)
    assert first_notification["status"] == "completed"

    second = runtime.send_message(first["agent_id"], "continue")

    assert second["status"] == "running"
    assert second["task_id"] == first["task_id"]
    assert second["restarted"] is True
    assert "prior interactive context" in second["status_note"]
    second_notification = runtime.wait_for_task(second["task_id"], timeout=5)
    assert second_notification["task_id"] == second["task_id"]
    team = runtime.store.load_team("team-retry")
    assert team is not None
    worker = team.agents[0]
    assert worker.task_id == first["task_id"]
    assert worker.restart_count == 1
    assert "prior interactive context" in worker.status_note
    assert worker.session_id.endswith("-restart1")


def test_task_tools_read_worker_state_and_output(tmp_path: Path) -> None:
    team_root = tmp_path / ".haagent" / "teams"
    runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-task-tools",
        model_gateway=FakeModelGateway(ModelResponse(content="worker final output", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "task_get", "task_list", "task_output"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=team_root,
    )
    router = ToolRouter(
        ["agent", "task_get", "task_list", "task_output"],
        _writer(tmp_path),
        workspace_root=tmp_path,
        path_policy=default_path_policy(tmp_path),
        agent_runtime=runtime,
    )
    started = router.dispatch(
        "agent",
        {
            "description": "Inspect project",
            "prompt": "Say done",
            "subagent_type": "explorer",
            "team": "team-tools",
        },
    )
    runtime.wait_for_task(started["task_id"], timeout=5)

    task_get = router.dispatch("task_get", {"task_id": started["task_id"]})
    task_list = router.dispatch("task_list", {"status": "completed"})
    task_output = router.dispatch("task_output", {"task_id": started["task_id"], "max_chars": 200})

    assert task_get["status"] == "success"
    assert task_get["task"]["task_id"] == started["task_id"]
    assert task_get["task"]["status"] == "completed"
    assert task_list["status"] == "success"
    assert task_list["tasks"][0]["task_id"] == started["task_id"]
    assert task_output["status"] == "success"
    assert "worker final output" in task_output["output"]
    assert "episode_path" in task_output

    MultiAgentRuntime._task_registry.clear()
    reloaded_runtime = MultiAgentRuntime(
        runs_root=tmp_path / ".runs",
        workspace_root=tmp_path,
        leader_session_id="leader-task-tools",
        model_gateway=FakeModelGateway(ModelResponse(content="unused", tool_calls=[])),
        path_policy=default_path_policy(tmp_path),
        inherited_allowed_tools=["agent", "send_message", "task_stop", "task_get", "task_list", "task_output"],
        inherited_approval_allowed_tools=[],
        inherited_approved_tools=[],
        event_sink=None,
        interaction_handler=None,
        enable_web=False,
        mcp_tool_names=[],
        tool_registry=None,
        mcp_runtime=None,
        team_root=team_root,
    )

    reloaded_output = reloaded_runtime.task_output(started["task_id"], max_chars=200)

    assert reloaded_output["status"] == "success"
    assert "worker final output" in reloaded_output["output"]


def test_failure_summary_uses_episode_failure_evidence_when_reason_is_missing(tmp_path: Path) -> None:
    episode = tmp_path / ".runs" / "episode-failed"
    episode.mkdir(parents=True)
    (episode / "failure.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "failure": {
                    "stage": "executing",
                    "category": "Loop Limit Failure",
                    "evidence": "exceeded max_turns=6",
                },
            },
        ),
        encoding="utf-8",
    )

    summary = _failure_summary_from_episode(
        SimpleNamespace(
            status="failed",
            final_response=None,
            reason=None,
            episode_path=episode,
        ),
    )

    assert summary == "Loop Limit Failure: exceeded max_turns=6"
