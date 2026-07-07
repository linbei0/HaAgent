"""
tests/integration/models/test_model_gateway.py - ModelGateway 接口与 provider 行为测试

验证 fake model、OpenAI 适配和模型失败显式暴露。
"""

import base64
import json
import socket
from pathlib import Path

import pytest

from haagent.models.catalog import ModelCatalogProvider
from haagent.models.fake import FakeModelGateway
from haagent.models.gateway import (
    AnthropicMessagesGateway,
    DEFAULT_CHAT_COMPLETIONS_ENDPOINT,
    DEFAULT_RESPONSES_ENDPOINT,
    GoogleGeminiGateway,
    ModelCallError,
    ModelResponse,
    ModelUsage,
    OpenAIChatCompletionsGateway,
    OpenAIResponsesGateway,
    ToolCall,
)
from haagent.models.credentials import FakeCredentialStore
from haagent.models.gateway_registry import (
    catalog_provider_capability,
    gateway_from_profile,
    GatewayRegistryError,
)
from haagent.models.model_connections import (
    ModelSelection,
    ProviderConnectionRecord,
    ProviderProfile,
    ProviderProfileError,
    list_provider_connection_records,
    load_model_selection_profile,
    provider_connection_credential_status,
    save_provider_connection,
    save_provider_connection_with_key,
)
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.contracts.task import TaskSpec


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


def _image_user_message(tmp_path: Path) -> dict[str, object]:
    image_path = tmp_path / "attachments" / "img.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image-bytes")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_attachment",
                "mime_type": "image/png",
                "path": str(image_path),
                "relative_path": "attachments/img.png",
            },
        ],
    }


def test_gateway_registry_maps_anthropic_catalog_provider_to_native_gateway() -> None:
    provider = ModelCatalogProvider(
        id="anthropic",
        name="Anthropic",
        env_names=["ANTHROPIC_API_KEY"],
        api_base_url=None,
        provider_package="@ai-sdk/anthropic",
        documentation_url="https://docs.anthropic.com/",
        models=[],
    )

    capability = catalog_provider_capability(provider)

    assert capability.status == "runnable"
    assert capability.gateway_provider == "anthropic"


def test_gateway_registry_tolerates_catalog_provider_without_provider_package() -> None:
    provider = ModelCatalogProvider(
        id="openrouter",
        name="OpenRouter",
        env_names=["OPENROUTER_API_KEY"],
        api_base_url="https://openrouter.ai/api/v1",
        provider_package=None,
        documentation_url="https://openrouter.ai/docs",
        models=[],
    )

    capability = catalog_provider_capability(provider)

    assert capability.status == "runnable"
    assert capability.gateway_provider == "openai-chat"


@pytest.mark.parametrize(
    ("provider_id", "name", "api_base_url", "provider_package"),
    [
        ("requesty", "Requesty", "https://router.requesty.ai/v1", "@ai-sdk/openai-compatible"),
        ("deepseek", "DeepSeek", "https://api.deepseek.com", "@ai-sdk/openai-compatible"),
        ("lmstudio", "LMStudio", "http://127.0.0.1:1234/v1", "@ai-sdk/openai-compatible"),
        ("ollama-cloud", "Ollama Cloud", "https://ollama.com/v1", "@ai-sdk/openai-compatible"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "@openrouter/ai-sdk-provider"),
    ],
)
def test_gateway_registry_maps_openai_compatible_catalog_provider_to_chat_gateway(
    provider_id: str,
    name: str,
    api_base_url: str,
    provider_package: str,
) -> None:
    provider = ModelCatalogProvider(
        id=provider_id,
        name=name,
        env_names=[f"{provider_id.upper().replace('-', '_')}_API_KEY"],
        api_base_url=api_base_url,
        provider_package=provider_package,
        documentation_url=f"https://{provider_id}.example/docs",
        models=[],
    )

    capability = catalog_provider_capability(provider)

    assert capability.status == "runnable"
    assert capability.gateway_provider == "openai-chat"


def test_gateway_registry_builds_existing_openai_chat_gateway() -> None:
    gateway = gateway_from_profile(
        ProviderProfile(
            name="router",
            provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-5.2-chat",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="keyring",
            credential_source_used="direct",
            api_key="sk-test",
        )
    )

    assert isinstance(gateway, OpenAIChatCompletionsGateway)


def test_gateway_registry_builds_anthropic_gateway() -> None:
    gateway = gateway_from_profile(
        ProviderProfile(
            name="anthropic-main",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
            credential_source="keyring",
            credential_source_used="direct",
            api_key="sk-test",
        )
    )

    assert isinstance(gateway, AnthropicMessagesGateway)


def test_gateway_registry_builds_google_gemini_gateway() -> None:
    gateway = gateway_from_profile(
        ProviderProfile(
            name="google-main",
            provider="google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-2.5-pro",
            api_key_env="GEMINI_API_KEY",
            credential_source="keyring",
            credential_source_used="direct",
            api_key="gemini-test-key",
        )
    )

    assert isinstance(gateway, GoogleGeminiGateway)


def test_gateway_metadata_redacts_secret_like_endpoint_parts() -> None:
    gateway = OpenAIChatCompletionsGateway(
        api_key="sk-test-secret",
        model="gpt-test",
        base_url="https://user:pass@example.test/v1?api_key=secret",
    )

    metadata = gateway.metadata()

    assert metadata.provider == "openai-chat"
    assert metadata.model == "gpt-test"
    assert metadata.endpoint == "https://example.test/v1/chat/completions"
    assert metadata.base_url == "https://example.test/v1"
    serialized = json.dumps(metadata.__dict__, ensure_ascii=False)
    assert "sk-test-secret" not in serialized
    assert "api_key" not in serialized
    assert "secret" not in serialized
    assert "user:pass" not in serialized


def test_gateway_registry_maps_google_catalog_provider_to_gemini_gateway() -> None:
    provider = ModelCatalogProvider(
        id="google",
        name="Google",
        env_names=["GEMINI_API_KEY"],
        api_base_url="https://generativelanguage.googleapis.com/v1beta",
        provider_package="@ai-sdk/google",
        documentation_url="https://ai.google.dev/gemini-api/docs",
        models=[],
    )

    capability = catalog_provider_capability(provider)

    assert capability.status == "runnable"
    assert capability.gateway_provider == "google"


def test_anthropic_gateway_text_response_uses_messages_payload() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["endpoint"] = endpoint
        return {
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        transport=transport,
    )

    result = gateway.generate([{"role": "user", "content": "hi"}], [])

    assert result.content == "hello"
    assert result.tool_calls == []
    assert captured["payload"] == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert captured["api_key"] == "sk-ant-test"
    assert captured["endpoint"] == "https://api.anthropic.com/v1/messages"


def test_anthropic_gateway_converts_image_attachment_to_image_block(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "ok"}]}

    gateway = AnthropicMessagesGateway(api_key="test", transport=transport)
    generate(gateway, messages=[_image_user_message(tmp_path)])

    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(b"image-bytes").decode("ascii"),
        },
    }
    assert content[1] == {"type": "text", "text": "Describe this image"}


def test_anthropic_gateway_parses_usage_metadata() -> None:
    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        return {
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-test",
        transport=transport,
    )

    result = gateway.generate([{"role": "user", "content": "hi"}], [])

    assert result.usage == ModelUsage(
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
        raw_source="anthropic.messages.usage",
    )


def test_anthropic_gateway_normalizes_tool_use_blocks() -> None:
    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        assert payload["tools"] == [
            {
                "name": "file_read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ]
        return {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "file_read",
                    "input": {"path": "README.md"},
                }
            ],
            "stop_reason": "tool_use",
        }

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        transport=transport,
    )

    result = gateway.generate(
        [{"role": "user", "content": "read"}],
        [
            {
                "name": "file_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
    )

    assert result.content == ""
    assert result.tool_calls == [
        ToolCall(id="toolu_123", name="file_read", args={"path": "README.md"})
    ]


def test_anthropic_gateway_moves_system_message_to_top_level_system() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "ok"}]}

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        transport=transport,
    )

    gateway.generate(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "task"},
        ],
        [],
    )

    assert captured["payload"] == {
        "model": "claude-sonnet-4-5",
        "max_tokens": 4096,
        "system": "system instructions",
        "messages": [{"role": "user", "content": "task"}],
    }


def test_anthropic_gateway_converts_tool_loop_messages() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "done"}]}

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        transport=transport,
    )

    gateway.generate(
        [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "I will read it.",
                "tool_calls": [
                    {
                        "id": "toolu_123",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{\"path\": \"README.md\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_123",
                "name": "file_read",
                "content": "{\"content\": \"hello\"}",
            },
        ],
        [],
    )

    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will read it."},
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "file_read",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "{\"content\": \"hello\"}",
                }
            ],
        },
    ]


def test_anthropic_gateway_groups_multiple_tool_results_in_one_user_message() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {"content": [{"type": "text", "text": "done"}]}

    gateway = AnthropicMessagesGateway(
        api_key="sk-ant-test",
        model="claude-sonnet-4-5",
        transport=transport,
    )

    gateway.generate(
        [
            {"role": "user", "content": "read files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{\"path\": \"README.md\"}",
                        },
                    },
                    {
                        "id": "toolu_2",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{\"path\": \"AGENTS.md\"}",
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_1",
                "name": "file_read",
                "content": "{\"content\": \"readme\"}",
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_2",
                "name": "file_read",
                "content": "{\"content\": \"agents\"}",
            },
        ],
        [],
    )

    assert captured["payload"]["messages"] == [
        {"role": "user", "content": "read files"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "file_read",
                    "input": {"path": "README.md"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "file_read",
                    "input": {"path": "AGENTS.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "{\"content\": \"readme\"}",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "{\"content\": \"agents\"}",
                },
            ],
        },
    ]


def test_google_gemini_gateway_text_response_uses_generate_content_payload() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["endpoint"] = endpoint
        return {
            "candidates": [
                {"content": {"parts": [{"text": "hello"}], "role": "model"}}
            ]
        }

    gateway = GoogleGeminiGateway(
        api_key="gemini-test-key",
        model="gemini-2.5-pro",
        transport=transport,
    )

    result = gateway.generate([{"role": "user", "content": "hi"}], [])

    assert result.content == "hello"
    assert result.tool_calls == []
    assert captured["payload"] == {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
    }
    assert captured["api_key"] == "gemini-test-key"
    assert captured["endpoint"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-pro:generateContent"
    )


def test_google_gemini_gateway_converts_image_attachment_to_inline_data(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    gateway = GoogleGeminiGateway(api_key="test", transport=transport)
    generate(gateway, messages=[_image_user_message(tmp_path)])

    parts = captured["payload"]["contents"][0]["parts"]
    assert parts[0] == {"text": "Describe this image"}
    assert parts[1] == {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(b"image-bytes").decode("ascii"),
        },
    }


def test_google_gemini_gateway_parses_usage_metadata() -> None:
    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        return {
            "candidates": [
                {"content": {"parts": [{"text": "hello"}], "role": "model"}}
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 6,
                "totalTokenCount": 26,
            },
        }

    gateway = GoogleGeminiGateway(
        api_key="gemini-test-key",
        model="gemini-test",
        transport=transport,
    )

    result = gateway.generate([{"role": "user", "content": "hi"}], [])

    assert result.usage == ModelUsage(
        input_tokens=20,
        output_tokens=6,
        total_tokens=26,
        raw_source="google.gemini.usageMetadata",
    )


def test_google_gemini_gateway_normalizes_function_calls() -> None:
    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        assert payload["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "file_read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                        },
                    }
                ]
            }
        ]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "file_read",
                                    "args": {"path": "README.md"},
                                }
                            }
                        ],
                        "role": "model",
                    }
                }
            ]
        }

    gateway = GoogleGeminiGateway(
        api_key="gemini-test-key",
        model="gemini-2.5-pro",
        transport=transport,
    )

    result = gateway.generate(
        [{"role": "user", "content": "read"}],
        [
            {
                "name": "file_read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
    )

    assert result.content == ""
    assert result.tool_calls == [ToolCall(name="file_read", args={"path": "README.md"})]


def test_google_gemini_gateway_moves_system_message_to_system_instruction() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {
            "candidates": [
                {"content": {"parts": [{"text": "hello"}], "role": "model"}}
            ]
        }

    gateway = GoogleGeminiGateway(
        api_key="gemini-test-key",
        model="gemini-2.5-pro",
        transport=transport,
    )

    gateway.generate(
        [
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "task"},
        ],
        [],
    )

    assert captured["payload"] == {
        "systemInstruction": {"parts": [{"text": "system instructions"}]},
        "contents": [{"role": "user", "parts": [{"text": "task"}]}],
    }


def test_google_gemini_gateway_converts_tool_loop_messages() -> None:
    captured: dict[str, object] = {}

    def transport(
        payload: dict[str, object],
        api_key: str,
        endpoint: str,
    ) -> dict[str, object]:
        captured["payload"] = payload
        return {
            "candidates": [
                {"content": {"parts": [{"text": "done"}], "role": "model"}}
            ]
        }

    gateway = GoogleGeminiGateway(
        api_key="gemini-test-key",
        model="gemini-2.5-pro",
        transport=transport,
    )

    gateway.generate(
        [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "file_read",
                            "arguments": "{\"path\": \"README.md\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "file_read",
                "content": "{\"content\": \"hello\"}",
            },
        ],
        [],
    )

    assert captured["payload"]["contents"] == [
        {"role": "user", "parts": [{"text": "read"}]},
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "file_read",
                        "args": {"path": "README.md"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "file_read",
                        "response": {"content": "{\"content\": \"hello\"}"},
                    }
                }
            ],
        },
    ]


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


def test_model_selection_loads_connection_and_api_key_env(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )

    profile = load_model_selection_profile(
        ModelSelection("deepseek", "deepseek-v4-pro"),
        config_path=config_path,
        environ={"DEEPSEEK_API_KEY": "secret-key"},
    )

    assert profile.name == "deepseek:deepseek-v4-pro"
    assert profile.provider == "openai-chat"
    assert profile.base_url == "https://api.deepseek.com"
    assert profile.model == "deepseek-v4-pro"
    assert profile.api_key_env == "DEEPSEEK_API_KEY"
    assert profile.api_key == "secret-key"


def test_model_selection_loads_api_key_from_keyring_when_env_missing(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                        "credential_source": "keyring",
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )

    profile = load_model_selection_profile(
        ModelSelection("deepseek", "deepseek-v4-pro"),
        config_path=config_path,
        environ={},
        credential_store=FakeCredentialStore({"connection:deepseek": "keyring-secret"}),
    )

    assert profile.api_key == "keyring-secret"
    assert profile.credential_source == "keyring"
    assert profile.credential_source_used == "keyring"


def test_provider_connections_can_be_listed_without_secrets(tmp_path: Path) -> None:
    save_provider_connection(
        ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="keyring",
        ),
        config_dir=tmp_path,
    )

    records = list_provider_connection_records(config_path=tmp_path / "providers.json")

    assert [record.name for record in records] == ["router"]
    assert records[0].provider_id == "openrouter"
    assert "secret" not in records[0].to_dict()


def test_provider_connection_credential_status_for_named_connection(tmp_path: Path) -> None:
    save_provider_connection(
        ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="keyring",
        ),
        config_dir=tmp_path,
    )

    status = provider_connection_credential_status(
        "router",
        config_dir=tmp_path,
        credential_store=FakeCredentialStore({"connection:router": "sk-test-secret"}),
    )

    assert status.api_key_available is True
    assert status.credential_source_used == "keyring"
    assert "sk-test-secret" not in repr(status)


def test_model_selection_loads_api_key_from_explicit_insecure_file(tmp_path: Path) -> None:
    config_dir = tmp_path / ".haagent"
    config_path = config_dir / "providers.json"
    save_provider_connection_with_key(
        ProviderConnectionRecord(
            id="deepseek",
            name="deepseek",
            provider_id="deepseek",
            provider_name="DeepSeek",
            gateway_provider="openai-chat",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            credential_source="insecure_file",
        ),
        "plain-secret",
        config_dir=config_dir,
    )

    profile = load_model_selection_profile(
        ModelSelection("deepseek", "deepseek-v4-pro"),
        config_path=config_path,
        environ={},
        credential_store=FakeCredentialStore({}),
        config_dir=config_dir,
    )

    assert profile.api_key == "plain-secret"
    assert profile.credential_source == "insecure_file"
    assert profile.credential_source_used == "insecure_file"


def test_model_selection_missing_connection_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "openai-main",
                        "name": "openai-main",
                        "provider_id": "openai",
                        "provider_name": "OpenAI",
                        "gateway_provider": "openai",
                        "base_url": "https://api.openai.com",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="provider connection not found: deepseek"):
        load_model_selection_profile(
            ModelSelection("deepseek", "deepseek-v4-pro"),
            config_path=config_path,
            environ={"OPENAI_API_KEY": "key"},
        )


def test_model_selection_missing_api_key_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 2,
                "connections": [
                    {
                        "id": "deepseek",
                        "name": "deepseek",
                        "provider_id": "deepseek",
                        "provider_name": "DeepSeek",
                        "gateway_provider": "openai-chat",
                        "base_url": "https://api.deepseek.com",
                        "api_key_env": "DEEPSEEK_API_KEY",
                    },
                ],
                "custom_models": [],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="API key is not available"):
        load_model_selection_profile(
            ModelSelection("deepseek", "deepseek-v4-pro"),
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


def test_openai_responses_gateway_converts_image_attachment_to_input_image(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        return {"output_text": "ok"}

    gateway = OpenAIResponsesGateway(api_key="test", transport=transport)
    generate(gateway, messages=[_image_user_message(tmp_path)])

    user_input = captured["payload"]["input"][0]
    assert user_input["role"] == "user"
    assert user_input["content"][0] == {"type": "input_text", "text": "Describe this image"}
    assert user_input["content"][1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{base64.b64encode(b'image-bytes').decode('ascii')}",
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


def test_openai_chat_gateway_parses_usage_metadata() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": "provider text"}}],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 9,
                "total_tokens": 39,
            },
        }

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        model="chat-test",
        transport=transport,
    )

    response = generate(gateway)

    assert response.usage == ModelUsage(
        input_tokens=30,
        output_tokens=9,
        total_tokens=39,
        raw_source="openai.chat_completions.usage",
    )


def test_openai_chat_gateway_streams_text_deltas_and_returns_complete_response() -> None:
    deltas: list[str] = []

    def stream_transport(payload: dict[str, object], api_key: str, on_delta):
        assert payload["stream"] is True
        on_delta("chat ")
        on_delta("text")
        return {"choices": [{"message": {"content": "chat text"}}]}

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        model="chat-test",
        transport=lambda payload, api_key: {"choices": [{"message": {"content": "unused"}}]},
        stream_transport=stream_transport,
    )

    response = gateway.generate(
        [{"role": "user", "content": "standardized context"}],
        [],
        event_sink=deltas.append,
    )

    assert deltas == ["chat ", "text"]
    assert response == ModelResponse(content="chat text", tool_calls=[])


def test_openai_chat_gateway_stream_timeout_remains_model_call_error() -> None:
    def stream_transport(payload: dict[str, object], api_key: str, on_delta):
        raise socket.timeout("read timed out")

    gateway = OpenAIChatCompletionsGateway(
        api_key="test-key",
        transport=lambda payload, api_key: {"choices": [{"message": {"content": "unused"}}]},
        stream_transport=stream_transport,
    )

    with pytest.raises(ModelCallError, match="read timed out"):
        gateway.generate(
            [{"role": "user", "content": "slow"}],
            [],
            event_sink=lambda _delta: None,
        )


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


def test_openai_chat_gateway_converts_image_attachment_to_image_url_part(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = OpenAIChatCompletionsGateway(api_key="test", transport=transport)
    generate(gateway, messages=[_image_user_message(tmp_path)])

    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this image"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64.b64encode(b'image-bytes').decode('ascii')}"},
    }


def test_openai_chat_gateway_explains_text_only_compatible_image_rejection(tmp_path: Path) -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        raise RuntimeError(
            "OpenAI chat request failed with HTTP 400: "
            "Failed to deserialize the JSON body into the target type: "
            "messages[1]: unknown variant `image_url`, expected `text`"
        )

    gateway = OpenAIChatCompletionsGateway(api_key="test", transport=transport)

    with pytest.raises(ModelCallError, match="当前模型或接口不支持图片输入"):
        generate(gateway, messages=[_image_user_message(tmp_path)])


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


def test_openai_responses_gateway_parses_usage_metadata() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {
            "output_text": "provider text",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            },
        }

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    response = generate(gateway)

    assert response.usage == ModelUsage(
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        raw_source="openai.responses.usage",
    )


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


def test_openai_responses_gateway_leaves_usage_none_when_missing() -> None:
    def transport(payload: dict[str, object], api_key: str) -> dict[str, object]:
        return {"output_text": "provider text"}

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        model="gpt-test",
        transport=transport,
    )

    response = generate(gateway)

    assert response.usage is None


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
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

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


def test_model_usage_is_written_to_transcript_and_cost_by_orchestrator(tmp_path: Path) -> None:
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Record model usage
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    response = ModelResponse(
        content="done",
        tool_calls=[],
        usage=ModelUsage(
            input_tokens=12,
            output_tokens=3,
            total_tokens=15,
            raw_source="fake.usage",
        ),
    )

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=FakeModelGateway(response),
    ).run(task_path)

    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    model_response = next(record for record in transcript if record.get("event") == "model_response")
    assert model_response["usage"] == {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
        "raw_usage_source": "fake.usage",
    }
    cost = json.loads((result.episode_path / "cost.json").read_text(encoding="utf-8"))
    assert cost["usage_available"] is True
    assert cost["totals"]["total_tokens"] == 15
    assert cost["model_calls"][0]["turn"] == 1


def test_full_compact_contract_is_written_to_transcript_by_orchestrator(tmp_path: Path) -> None:
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Record full compact contract
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

    contract = next(record for record in transcript if record.get("event") == "full_compact_contract")
    assert contract["eligible"] is False
    assert contract["reason"] == "insufficient_compressible_history"
    assert contract["required_preserve_recent"] == 6


def test_full_compact_success_is_written_to_transcript_by_orchestrator(tmp_path: Path, monkeypatch) -> None:
    from haagent.context.builder import BuiltContext
    from haagent.context.manifest import ContextManifest
    from haagent.models.gateway import ModelResponse
    from haagent.context.compression.full import FullCompactEligibility
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Record full compact execution
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    compact_response = ModelResponse(
        content=json.dumps(
            {
                "task_focus": "compact orchestrator context",
                "completed_work": ["older context summarized"],
                "open_issues": [],
                "important_files": ["src/haagent/runtime/orchestrator.py"],
                "tool_results": [],
                "constraints": ["preserve recent messages"],
                "verification": ["pytest"],
                "risks": [],
            },
        ),
        tool_calls=[],
    )
    gateway = FakeModelGateway(response=compact_response)
    original_build = __import__("haagent.runtime.orchestration.orchestrator", fromlist=["ContextBuilder"]).ContextBuilder.build

    def fake_build(self):
        context = original_build(self)
        manifest = ContextManifest(
            context_id=context.context_id,
            provider=context.manifest.provider,
            workspace_root=context.manifest.workspace_root,
            generated_at=context.manifest.generated_at,
            message_count=8,
            system_chars=context.manifest.system_chars,
            task_chars=context.manifest.task_chars,
            compaction=context.manifest.compaction,
            source_diagnostics=context.manifest.source_diagnostics,
            compact_readiness=context.manifest.compact_readiness,
            auto_compact_trigger=context.manifest.auto_compact_trigger,
            session_compaction=context.manifest.session_compaction,
            full_compact_contract=FullCompactEligibility(True, "full_compact_candidate_after_deterministic_compaction", "full_compact_candidate", 2).to_dict(),
        )
        return BuiltContext(
            context_id=context.context_id,
            messages=[{"role": "user", "content": f"older-{index}"} for index in range(6)]
            + [{"role": "user", "content": "recent-user"}, {"role": "assistant", "content": "recent-assistant"}],
            manifest=manifest,
            diagnostics=context.diagnostics,
        )

    monkeypatch.setattr("haagent.runtime.orchestration.orchestrator.ContextBuilder.build", fake_build)

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    start_index = next(index for index, record in enumerate(transcript) if record.get("event") == "full_compact_start")
    success_index = next(index for index, record in enumerate(transcript) if record.get("event") == "full_compact_success")
    model_call_index = next(index for index, record in enumerate(transcript) if record.get("event") == "model_call")
    assert start_index < success_index < model_call_index
    success = transcript[success_index]
    assert success["reason"] == "applied"
    assert success["older_message_count"] == 6
    assert success["preserved_recent_count"] == 2
    context_manifest = json.loads((result.episode_path / "contexts" / "0001-manifest.json").read_text(encoding="utf-8"))
    assert context_manifest["full_compact"]["applied"] is True
    assert gateway.calls[0]["tool_schemas"] == []
    assert gateway.calls[1]["messages"][0]["content"] == "[full_compact_boundary older_messages=6 preserved_recent=2]"


def test_full_compact_failure_is_written_to_transcript_and_original_messages_continue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from haagent.context.builder import BuiltContext
    from haagent.context.manifest import ContextManifest
    from haagent.models.gateway import ModelResponse
    from haagent.context.compression.full import FullCompactEligibility
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    class TwoStepGateway:
        provider_name = "two-step"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, messages, tool_schemas):
            self.calls.append({"messages": messages, "tool_schemas": tool_schemas})
            if len(self.calls) == 1:
                return ModelResponse(content=json.dumps({"task_focus": "missing fields"}), tool_calls=[])
            return ModelResponse(content="continue after compact failure", tool_calls=[])

    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
goal: Record full compact failure
constraints: []
allowed_tools:
  - fake_tool
acceptance_criteria: []
verification_commands: []
""".strip(),
        encoding="utf-8",
    )
    original_build = __import__("haagent.runtime.orchestration.orchestrator", fromlist=["ContextBuilder"]).ContextBuilder.build

    def fake_build(self):
        context = original_build(self)
        manifest = ContextManifest(
            context_id=context.context_id,
            provider=context.manifest.provider,
            workspace_root=context.manifest.workspace_root,
            generated_at=context.manifest.generated_at,
            message_count=5,
            system_chars=context.manifest.system_chars,
            task_chars=context.manifest.task_chars,
            compaction=context.manifest.compaction,
            full_compact_contract=FullCompactEligibility(True, "full_compact_candidate_after_deterministic_compaction", "full_compact_candidate", 2).to_dict(),
        )
        return BuiltContext(
            context_id=context.context_id,
            messages=[{"role": "user", "content": f"original-{index}"} for index in range(5)],
            manifest=manifest,
            diagnostics=context.diagnostics,
        )

    monkeypatch.setattr("haagent.runtime.orchestration.orchestrator.ContextBuilder.build", fake_build)
    gateway = TwoStepGateway()

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
    ).run(task_path)

    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failed = next(record for record in transcript if record.get("event") == "full_compact_failed")
    assert failed["reason"] == "schema_invalid"
    assert gateway.calls[1]["messages"] == [{"role": "user", "content": f"original-{index}"} for index in range(5)]
    context_manifest = json.loads((result.episode_path / "contexts" / "0001-manifest.json").read_text(encoding="utf-8"))
    assert context_manifest["full_compact"]["applied"] is False
    assert context_manifest["full_compact"]["reason"] == "schema_invalid"


def test_orchestrator_microcompacts_old_tool_result_messages(tmp_path: Path) -> None:
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    class LargeToolResultGateway:
        provider_name = "large-tool-result"

        def __init__(self) -> None:
            self.model_inputs: list[str] = []
            self._turn = 0

        def generate(self, messages, tool_schemas):
            self._turn += 1
            self.model_inputs.append("\n".join(str(m.get("content", "")) for m in messages))
            if self._turn == 1:
                return ModelResponse("read", [ToolCall("file_read", {"path": "large.txt", "limit": 80})])
            return ModelResponse("done", [])

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
    gateway = LargeToolResultGateway()
    runtime_events: list[dict[str, object]] = []

    result = RunOrchestrator(
        runs_root=tmp_path / ".runs",
        model_gateway=gateway,
        max_turns=2,
        event_sink=runtime_events.append,
    ).run(task_path)

    second_input = gateway.model_inputs[1]
    assert "HEAD-" in second_input
    assert "TAIL" in second_input
    assert "...[collapsed " in second_input
    assert large_content not in second_input
    transcript = [
        json.loads(line)
        for line in (result.episode_path / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record.get("event") == "compression_diagnostic" for record in transcript)
    microcompact_events = [
        event for event in runtime_events if event.get("event_type") == "compression_diagnostic"
    ]
    assert len(microcompact_events) == 1
    assert "event" not in microcompact_events[0]
    assert microcompact_events[0]["stage"] == "historical_tool_message"
    assert microcompact_events[0]["reason"] == "long_text_result"
