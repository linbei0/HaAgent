"""
tests/test_model_gateway.py - ModelGateway 接口与 provider 行为测试

验证 fake model、OpenAI 适配和模型失败显式暴露。
"""

import json
from pathlib import Path

import pytest

from agentfoundry.models.fake import FakeModelGateway
from agentfoundry.models.gateway import (
    DEFAULT_RESPONSES_ENDPOINT,
    ModelCallError,
    ModelResponse,
    OpenAIResponsesGateway,
    ToolCall,
)
from agentfoundry.runtime.episode import EpisodeWriter
from agentfoundry.runtime.task_contract import TaskSpec


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


def test_fake_model_gateway_accepts_optional_observations() -> None:
    gateway = FakeModelGateway()

    response = gateway.generate(make_task(), observations=[{"tool_name": "fake_tool"}])

    assert response.tool_calls == []


def test_fake_model_gateway_keeps_old_usage_and_records_optional_inputs() -> None:
    gateway = FakeModelGateway(response=ModelResponse(content="done", tool_calls=[]))

    legacy_response = gateway.generate(make_task())
    response = gateway.generate(
        make_task(),
        model_input="context text",
        tool_schemas=[{"name": "fake_tool"}],
        observations=[{"tool_name": "fake_tool"}],
    )

    assert legacy_response.content == "done"
    assert response.content == "done"
    assert gateway.calls[-1]["model_input"] == "context text"
    assert gateway.calls[-1]["tool_schemas"] == [{"name": "fake_tool"}]
    assert gateway.calls[-1]["observations"] == [{"tool_name": "fake_tool"}]


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


def test_openai_gateway_defaults_to_official_responses_endpoint() -> None:
    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        transport=lambda payload, api_key: {"output_text": "ok"},
    )

    assert gateway.responses_endpoint == DEFAULT_RESPONSES_ENDPOINT


def test_openai_gateway_reads_base_url_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        transport=lambda payload, api_key: {"output_text": "ok"},
    )

    assert gateway.responses_endpoint == "https://compatible.example/v1/responses"


@pytest.mark.parametrize(
    ("base_url", "expected_endpoint"),
    [
        ("https://compatible.example/v1/responses", "https://compatible.example/v1/responses"),
        ("https://compatible.example/v1", "https://compatible.example/v1/responses"),
        ("https://compatible.example", "https://compatible.example/v1/responses"),
        ("compatible.example", "https://compatible.example/v1/responses"),
    ],
)
def test_openai_gateway_normalizes_base_url(base_url: str, expected_endpoint: str) -> None:
    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        base_url=base_url,
        transport=lambda payload, api_key: {"output_text": "ok"},
    )

    assert gateway.responses_endpoint == expected_endpoint


def test_openai_gateway_payload_uses_model_input_and_tools() -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        return {"output_text": "provider text"}

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    gateway.generate(
        make_task(),
        model_input="standardized context",
        tool_schemas=[{"name": "fake_tool", "description": "test", "parameters": {}}],
    )

    assert captured["payload"] == {
        "model": "gpt-test",
        "input": "standardized context",
        "tools": [{"name": "fake_tool", "description": "test", "parameters": {}}],
    }


def test_openai_gateway_normalizes_tool_call_response() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "",
            "output": [
                {
                    "type": "function_call",
                    "name": "fake_tool",
                    "arguments": "{\"value\": 1}",
                },
            ],
        }

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    response = gateway.generate(make_task())

    assert response == ModelResponse(
        content="",
        tool_calls=[ToolCall(name="fake_tool", args={"value": 1})],
    )


def test_openai_gateway_text_only_response_with_output_message_has_no_tool_calls() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "provider text",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "provider text"}],
                },
                {"type": "output_text", "text": "provider text"},
            ],
        }

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    response = gateway.generate(make_task())

    assert response == ModelResponse(content="provider text", tool_calls=[])


def test_openai_gateway_rejects_unknown_output_type() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "provider text",
            "output": [{"type": "image_generation_call"}],
        }

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    with pytest.raises(ModelCallError, match="unsupported OpenAI output type"):
        gateway.generate(make_task())


def test_openai_gateway_rejects_invalid_tool_arguments_json() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "",
            "output": [
                {
                    "type": "function_call",
                    "name": "fake_tool",
                    "arguments": "{not-json",
                },
            ],
        }

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    with pytest.raises(ModelCallError, match="invalid tool arguments JSON"):
        gateway.generate(make_task())


def test_openai_gateway_rejects_missing_tool_name_or_arguments() -> None:
    def missing_name_transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "",
            "output": [{"type": "function_call", "arguments": "{}"}],
        }

    def missing_arguments_transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "",
            "output": [{"type": "function_call", "name": "fake_tool"}],
        }

    with pytest.raises(ModelCallError, match="missing tool name"):
        OpenAIResponsesGateway(
            api_key="test-key",
            model="gpt-test",
            transport=missing_name_transport,
        ).generate(make_task())

    with pytest.raises(ModelCallError, match="missing tool arguments"):
        OpenAIResponsesGateway(
            api_key="test-key",
            model="gpt-test",
            transport=missing_arguments_transport,
        ).generate(make_task())


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
    from agentfoundry.runtime.orchestrator import RunOrchestrator

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
    model_call = next(record for record in transcript if record.get("event") == "model_call")
    assert model_call["context_id"] == "0001"
    assert any(record.get("event") == "model_response" for record in transcript)
