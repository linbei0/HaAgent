"""
tests/test_model_gateway.py - ModelGateway 接口与 provider 行为测试

验证 fake model、OpenAI 适配和模型失败显式暴露。
"""

import json
from pathlib import Path

import pytest

from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import (
    DEFAULT_CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_RESPONSES_ENDPOINT,
    ModelCallError,
    ModelResponse,
    OpenAIChatCompletionsGateway,
    OpenAIResponsesGateway,
    ToolCall,
)
from haagent.models.credentials import FakeCredentialStore, save_insecure_api_key
from haagent.models.provider_profile import ProviderProfileError, load_provider_profile
from haagent.runtime.episode import EpisodeWriter
from haagent.runtime.task_contract import TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        goal="Exercise model gateway",
        constraints=[],
        allowed_tools=["fake_tool"],
        acceptance_criteria=[],
        verification_commands=[],
    )


def generate(
    gateway,
    *,
    model_input: str = "standardized context",
    tool_schemas: list[dict[str, object]] | None = None,
    observations: list[dict[str, object]] | None = None,
    messages: list[dict[str, object]] | None = None,
) -> ModelResponse:
    if messages is None:
        messages = [{"role": "user", "content": model_input}]
        if observations:
            for obs in observations:
                messages.append({
                    "role": "tool",
                    "tool_call_id": "",
                    "name": obs.get("tool_name", ""),
                    "content": str(obs.get("result", "")),
                })
    return gateway.generate(
        messages,
        tool_schemas or [],
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

    response = generate(gateway)

    assert response.content == "planned"
    assert response.tool_calls == [ToolCall(name="fake_tool", args={"value": 1})]


def test_fake_model_gateway_accepts_optional_observations() -> None:
    gateway = FakeModelGateway()

    response = generate(gateway, observations=[{"tool_name": "fake_tool"}])

    assert response.tool_calls == []


def test_fake_model_gateway_finishes_when_fake_tool_is_not_available() -> None:
    gateway = FakeModelGateway()

    response = generate(gateway, tool_schemas=[{"name": "file_read"}])

    assert response.tool_calls == []
    assert response.content == "Fake model has no fake_tool available; relying on verification."


def test_fake_model_gateway_records_current_inputs() -> None:
    gateway = FakeModelGateway(response=ModelResponse(content="done", tool_calls=[]))

    response = generate(
        gateway,
        model_input="context text",
        tool_schemas=[{"name": "fake_tool"}],
        observations=[{"tool_name": "fake_tool"}],
    )

    assert response.content == "done"
    assert gateway.calls[-1]["model_input"] == "context text"
    assert gateway.calls[-1]["tool_schemas"] == [{"name": "fake_tool"}]
    tool_msgs = [m for m in gateway.calls[-1]["messages"] if m.get("role") == "tool"]
    assert any(m.get("name") == "fake_tool" for m in tool_msgs)


def test_provider_profile_loads_named_profile_and_api_key_env(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    profile = load_provider_profile(
        "deepseek",
        config_path=config_path,
        environ={"DEEPSEEK_API_KEY": "secret-key"},
    )

    assert profile.name == "deepseek"
    assert profile.provider == "openai-chat"
    assert profile.base_url == "https://api.deepseek.com"
    assert profile.model == "deepseek-v4-pro"
    assert profile.api_key_env == "DEEPSEEK_API_KEY"
    assert profile.api_key == "secret-key"


def test_provider_profile_loads_api_key_from_keyring_when_env_missing(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "credential_source": "keyring",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    profile = load_provider_profile(
        "deepseek",
        config_path=config_path,
        environ={},
        credential_store=FakeCredentialStore({"profile:deepseek": "keyring-secret"}),
    )

    assert profile.api_key == "keyring-secret"
    assert profile.credential_source == "keyring"
    assert profile.credential_source_used == "keyring"


def test_provider_profile_loads_api_key_from_explicit_insecure_file(tmp_path: Path) -> None:
    config_dir = tmp_path / ".haagent"
    config_path = config_dir / "providers.json"
    config_dir.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "credential_source": "insecure_file",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    save_insecure_api_key("deepseek", "plain-secret", config_dir=config_dir)

    profile = load_provider_profile(
        "deepseek",
        config_path=config_path,
        environ={},
        credential_store=FakeCredentialStore({}),
        config_dir=config_dir,
    )

    assert profile.api_key == "plain-secret"
    assert profile.credential_source == "insecure_file"
    assert profile.credential_source_used == "insecure_file"


def test_provider_profile_missing_name_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "openai-main",
                        "provider": "openai",
                        "base_url": "https://api.openai.com",
                        "model": "gpt-4.1-mini",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="provider profile not found: deepseek"):
        load_provider_profile("deepseek", config_path=config_path, environ={"OPENAI_API_KEY": "key"})


def test_provider_profile_missing_api_key_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "deepseek",
                        "provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-v4-pro",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="API key is not available"):
        load_provider_profile(
            "deepseek",
            config_path=config_path,
            environ={},
            credential_store=FakeCredentialStore({}),
        )


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

    response = generate(gateway)

    assert response == ModelResponse(content="provider text", tool_calls=[])
    assert captured["api_key"] == "test-key"
    assert captured["payload"] == {"model": "gpt-test", "input": [{"role": "user", "content": "standardized context"}]}


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


@pytest.mark.parametrize(
    ("base_url", "expected_endpoint"),
    [
        (None, DEFAULT_CHAT_COMPLETIONS_ENDPOINT),
        (
            "https://compatible.example/v1/chat/completions",
            "https://compatible.example/v1/chat/completions",
        ),
        ("https://compatible.example/v1", "https://compatible.example/v1/chat/completions"),
        ("https://compatible.example", "https://compatible.example/v1/chat/completions"),
        ("compatible.example", "https://compatible.example/v1/chat/completions"),
    ],
)
def test_openai_chat_gateway_normalizes_endpoint(
    base_url: str | None,
    expected_endpoint: str,
) -> None:
    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        base_url=base_url,
        transport=lambda payload, api_key: {
            "choices": [{"message": {"content": "ok"}}],
        },
    )

    assert gateway.chat_completions_endpoint == expected_endpoint


def test_openai_chat_gateway_text_only_response_uses_messages_payload() -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        captured["api_key"] = api_key
        return {"choices": [{"message": {"content": "chat text"}}]}

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        model="chat-test",
        transport=transport,
    )

    response = generate(gateway, model_input="standardized context")

    assert response == ModelResponse(content="chat text", tool_calls=[])
    assert captured["api_key"] == "test-key"
    assert captured["payload"] == {
        "model": "chat-test",
        "messages": [{"role": "user", "content": "standardized context"}],
    }


def test_openai_chat_gateway_normalizes_tool_calls_and_tools_payload() -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "fake_tool",
                                    "arguments": "{\"value\": 1}",
                                },
                            },
                        ],
                    },
                },
            ],
        }

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        model="chat-test",
        transport=transport,
    )

    response = generate(
        gateway,
        model_input="context with tools",
        tool_schemas=[
            {
                "name": "fake_tool",
                "description": "test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
    )

    assert response == ModelResponse(
        content="",
        tool_calls=[ToolCall(name="fake_tool", args={"value": 1})],
    )
    assert captured["payload"] == {
        "model": "chat-test",
        "messages": [{"role": "user", "content": "context with tools"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "fake_tool",
                    "description": "test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }


def test_openai_chat_gateway_rejects_invalid_tool_arguments_json() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "fake_tool",
                                    "arguments": "{not-json",
                                },
                            },
                        ],
                    },
                },
            ],
        }

    gateway = OpenAIChatCompletionsGateway(api_key="test-key", transport=transport)

    with pytest.raises(ModelCallError, match="invalid tool arguments JSON"):
        generate(gateway)


def test_openai_chat_gateway_rejects_non_object_tool_arguments() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "fake_tool",
                                    "arguments": "[1, 2]",
                                },
                            },
                        ],
                    },
                },
            ],
        }

    gateway = OpenAIChatCompletionsGateway(api_key="test-key", transport=transport)

    with pytest.raises(ModelCallError, match="tool arguments must be a JSON object"):
        generate(gateway)


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

    generate(
        gateway,
        model_input="standardized context",
        tool_schemas=[{"name": "fake_tool", "description": "test", "parameters": {}}],
    )

    assert captured["payload"] == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "standardized context"}],
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

    response = generate(gateway)

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

    response = generate(gateway)

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
        generate(gateway)


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
        generate(gateway)


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
        generate(OpenAIResponsesGateway(
            api_key="test-key",
            model="gpt-test",
            transport=missing_name_transport,
        ))

    with pytest.raises(ModelCallError, match="missing tool arguments"):
        generate(OpenAIResponsesGateway(
            api_key="test-key",
            model="gpt-test",
            transport=missing_arguments_transport,
        ))


def test_openai_gateway_failure_is_explicit() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        raise RuntimeError("provider unavailable")

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    with pytest.raises(ModelCallError, match="provider unavailable"):
        generate(gateway)


def test_model_call_is_written_to_transcript_by_orchestrator(tmp_path: Path) -> None:
    from haagent.runtime.orchestrator import RunOrchestrator

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
