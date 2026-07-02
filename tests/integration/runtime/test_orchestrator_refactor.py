"""
tests/integration/runtime/test_orchestrator_refactor.py - RunOrchestrator 重构护栏测试

锁定 orchestrator 分层重构期间必须保持不变的 transcript 与失败归因行为。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.models.gateway import ModelResponse, ToolCall
from haagent.runtime.orchestrator import RunOrchestrator
from haagent.runtime.state import RunStatus


class BadFileReadGateway:
    provider_name = "bad-file-read"

    def generate(self, messages, tool_schemas):
        return ModelResponse("bad args", [ToolCall("file_read", {"offset": 1})])


class RecoverableFileSearchGateway:
    provider_name = "recoverable-file-search"

    def __init__(self) -> None:
        self.call_count = 0

    def generate(self, messages, tool_schemas):
        self.call_count += 1
        if self.call_count == 1:
            return ModelResponse("", [ToolCall("file_search", {"query": "needle", "root": "alpha.txt"})])
        return ModelResponse("done after suggestion", [])


def test_tool_observation_is_written_before_terminal_tool_routing_failure(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Exercise terminal tool argument failure
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=BadFileReadGateway(),
    ).run(task_path)

    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failure = json.loads((result.episode_path / "failure.json").read_text(encoding="utf-8"))

    assert result.status is RunStatus.FAILED
    assert any(record.get("event") == "model_call" for record in transcript)
    assert any(record.get("event") == "model_response" for record in transcript)
    observation = next(record for record in transcript if record.get("event") == "tool_observation")
    assert observation["tool_name"] == "file_read"
    assert observation["result"]["error"]["type"] == "tool_argument_invalid"
    assert failure["failure"] == {
        "stage": "executing",
        "category": "Tool Argument Failure",
        "evidence": "missing required argument: path",
    }


def test_recoverable_tool_argument_error_continues_turn_loop(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("needle appears here\n", encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Recover from a tool argument error
constraints: []
allowed_tools:
  - file_search
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    gateway = RecoverableFileSearchGateway()

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    tool_call = json.loads((result.episode_path / "tool-calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observation = next(record for record in transcript if record.get("event") == "tool_observation")

    assert result.status is RunStatus.COMPLETED
    assert gateway.call_count == 2
    assert tool_call["status"] == "error"
    assert tool_call["error"]["type"] == "tool_argument_invalid"
    assert observation["result"]["suggested_tool"] == {
        "name": "file_read",
        "args": {"path": "alpha.txt", "keyword": "needle"},
    }
