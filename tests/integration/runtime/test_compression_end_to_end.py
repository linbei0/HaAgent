"""
tests/integration/runtime/test_compression_end_to_end.py - 统一压缩端到端测试

验证长工具结果在多轮历史中会产生统一 compression_diagnostic 事件。
"""

from __future__ import annotations

import json
from pathlib import Path

from haagent.models.types import ModelResponse, ToolCall
from haagent.runtime.orchestration.orchestrator import RunOrchestrator
from haagent.runtime.orchestration.state import RunStatus


class LongFileGateway:
    provider_name = "compression-e2e"

    def __init__(self) -> None:
        self.turn = 0
        self.model_inputs: list[str] = []

    def generate(self, messages, tool_schemas):
        self.turn += 1
        self.model_inputs.append("\n".join(str(message.get("content", "")) for message in messages))
        if self.turn == 1:
            return ModelResponse("", [ToolCall("file_read", {"path": "large.txt"})])
        return ModelResponse("done", [])


def test_historical_tool_result_emits_compression_diagnostic(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    large_content = "HEAD-" + ("body-" * 2000) + "TAIL"
    (workspace / "large.txt").write_text(large_content, encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        f"""
goal: Read large file
workspace_root: {workspace}
constraints: []
allowed_tools:
  - file_read
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    from haagent.runtime.events.bus import bus_event_to_dict

    runtime_events: list[object] = []
    gateway = LongFileGateway()

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=2,
        event_sink=runtime_events.append,
    ).run(task_path)

    assert result.status is RunStatus.COMPLETED
    assert large_content not in gateway.model_inputs[1]
    diagnostics = [
        bus_event_to_dict(event)
        for event in runtime_events
        if bus_event_to_dict(event).get("event_type") == "compression_diagnostic"
    ]
    assert diagnostics
    assert diagnostics[0]["stage"] == "historical_tool_message"
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event") == "compression_diagnostic" for record in transcript)
