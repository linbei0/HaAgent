import json
from pathlib import Path

import pytest

from agent_foundry.episode import EpisodeWriter
from agent_foundry.model_gateway import (
    FakeModelGateway,
    ModelCallError,
    ModelResponse,
    OpenAIResponsesGateway,
    ToolCall,
)
from agent_foundry.task import TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        goal="Exercise model gateway",
        constraints=[],
        allowed_tools=["fake_tool"],
        acceptance_criteria=[],
        verification_commands=[],
    )


def make_writer(tmp_path: Path) -> EpisodeWriter:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Exercise model gateway
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    return EpisodeWriter.create(runs_root=tmp_path / ".runs", task_path=task_path)


def test_fake_model_gateway_returns_configured_response() -> None:
    gateway = FakeModelGateway(
        response=ModelResponse(
            content="planned",
            tool_calls=[ToolCall(name="fake_tool", args={"value": 1})],
        ),
    )

    response = gateway.generate(make_task())

    assert response.content == "planned"
    assert response.tool_calls == [ToolCall(name="fake_tool", args={"value": 1})]


def test_openai_gateway_uses_unified_response_shape() -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {"output_text": "provider text"}

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    response = gateway.generate(make_task())

    assert response == ModelResponse(content="provider text", tool_calls=[])
    assert captured["api_key"] == "test-key"
    assert captured["payload"] == {
        "model": "gpt-test",
        "input": "Goal: Exercise model gateway\nConstraints:\n- none\nAcceptance criteria:\n- none",
    }


def test_openai_gateway_failure_is_explicit() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        raise RuntimeError("provider unavailable")

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    with pytest.raises(ModelCallError, match="provider unavailable"):
        gateway.generate(make_task())


def test_model_call_is_written_to_transcript_by_orchestrator(tmp_path: Path) -> None:
    from agent_foundry.orchestrator import RunOrchestrator

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Record model transcript
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=FakeModelGateway(),
    ).run(task_path)

    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event") == "model_call" for record in transcript)
    assert any(record.get("event") == "model_response" for record in transcript)
