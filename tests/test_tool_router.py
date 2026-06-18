import json
from pathlib import Path

import pytest

from agent_foundry.episode import EpisodeWriter
from agent_foundry.tools import ToolRouter, ToolRoutingError


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Route a tool
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    return EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)


def test_tool_router_runs_fake_tool_and_writes_trace(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=["fake_tool"], episode_writer=writer)

    result = router.dispatch("fake_tool", {"value": 42})

    assert result == {"status": "success", "args": {"value": 42}}
    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(trace)
    assert record["tool_name"] == "fake_tool"
    assert record["status"] == "success"
    assert record["result"] == result


def test_tool_router_rejects_disallowed_tool(tmp_path: Path) -> None:
    writer = make_writer(tmp_path)
    router = ToolRouter(allowed_tools=[], episode_writer=writer)

    with pytest.raises(ToolRoutingError, match="not allowed"):
        router.dispatch("fake_tool", {})

    trace = (writer.path / "tool-calls.jsonl").read_text(encoding="utf-8")
    record = json.loads(trace)
    assert record["tool_name"] == "fake_tool"
    assert record["status"] == "error"
