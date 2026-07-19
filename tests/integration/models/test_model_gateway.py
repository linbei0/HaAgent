"""
tests/integration/models/test_model_gateway.py - ModelGateway 接口与 provider 行为测试

验证 fake model、OpenAI 适配和模型失败显式暴露。
"""

import base64
import json
import socket
from pathlib import Path

import pytest

from haagent.models.capabilities import ModelCapabilities
from haagent.models.catalog import ModelCatalogProvider
from haagent.models.fake import FakeModelGateway
from haagent.models.adapters.anthropic import AnthropicMessagesGateway
from haagent.models.adapters.google import GoogleGeminiGateway
from haagent.models.adapters.openai_chat import OpenAIChatCompletionsGateway
from haagent.models.adapters.openai_responses import OpenAIResponsesGateway
from haagent.models.adapters.transport import DEFAULT_CHAT_COMPLETIONS_ENDPOINT, DEFAULT_RESPONSES_ENDPOINT
from haagent.models.types import ModelCallError, ModelFailureDetails, ModelResponse, ModelUsage, ToolCall
from tests.support.model_credentials import FakeCredentialStore
from haagent.models.gateway_registry import (
    catalog_provider_capability,
    gateway_from_resolved,
    gateway_from_route,
)
from haagent.models.config.connections import (
    ProviderConnectionRecord,
    ProviderProfileError,
    provider_connection_credential_status,
    save_connection_api_key,
)
from haagent.models.config.config_store import ModelConfigStore
from haagent.models.model_ref import ModelInvocation, ModelRef, ResolvedCredential, ResolvedModel
from haagent.models.model_resolution import resolve_model
from haagent.models.model_settings import ModelSettings
from haagent.runtime.episodes.writer import EpisodeWriter
from haagent.runtime.contracts.task import TaskSpec
from haagent.runtime.execution.cancellation import CancellationToken, RunCancelled
from haagent.runtime.execution.retry import RetryController, RetryPolicy


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
    return gateway.generate(ModelInvocation(messages, tool_schemas or [], gateway.model_settings))


def _retry_controller() -> RetryController:
    return RetryController(
        RetryPolicy(max_attempts=2),
        sleep=lambda _: None,
        random_value=lambda: 0.0,
    )


def _request_config(options: dict[str, object]) -> ModelSettings:
    return ModelSettings.from_options(options)


def _resolved(provider: str, base_url: str, *, settings: ModelSettings | None = None) -> ResolvedModel:
    return ResolvedModel(
        ref=ModelRef(f"{provider}-main", "test-model"),
        provider=provider,
        base_url=base_url,
        runtime_kind="remote",
        settings=settings or ModelSettings.empty(),
        credential=ResolvedCredential("test-key", "TEST_API_KEY", "env", "env"),
    )


@pytest.fixture(autouse=True)
def _provider_invocation_test_adapter(monkeypatch):
    """本文件的 payload 测试用统一 invocation 调用 provider adapter。"""
    for gateway_type in (
        OpenAIResponsesGateway,
        OpenAIChatCompletionsGateway,
        AnthropicMessagesGateway,
        GoogleGeminiGateway,
    ):
        original = gateway_type.generate

        def adapted(self, invocation, tool_schemas=None, _original=original, **kwargs):
            if not isinstance(invocation, ModelInvocation):
                invocation = ModelInvocation(invocation, tool_schemas or [], self.model_settings)
            return _original(self, invocation, **kwargs)

        monkeypatch.setattr(gateway_type, "generate", adapted)


def _snapshot(config_path: Path):
    return ModelConfigStore(config_path).load()


def _save_connection(config_dir: Path, connection: ProviderConnectionRecord) -> Path:
    store = ModelConfigStore(config_dir / "providers.json")
    store.save_connection(connection, expected_digest=store.load().digest)
    return store.path


def test_openai_responses_gateway_retries_once_then_returns_response() -> None:
    attempts = 0
    retry_events = []

    def retrying_transport(payload, api_key):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelCallError(
                "temporary",
                details=ModelFailureDetails(category="server", status_code=503, retryable=True),
            )
        return {"output_text": "ok", "output": []}

    gateway = OpenAIResponsesGateway(
        api_key="key",
        model="model",
        transport=retrying_transport,
        retry_controller=_retry_controller(),
    )

    assert gateway.generate([], [], retry_event_sink=retry_events.append).content == "ok"
    assert attempts == 2
    assert [(event.attempt, event.next_attempt) for event in retry_events] == [(1, 2)]


def test_openai_responses_replays_interleaved_reasoning_and_calls_in_original_order() -> None:
    original_output = [
        {"type": "reasoning", "id": "r1", "encrypted_content": "opaque-1"},
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"path":"a"}'},
        {"type": "reasoning", "id": "r2", "encrypted_content": "opaque-2"},
        {"type": "function_call", "call_id": "c2", "name": "read", "arguments": '{"path":"b"}'},
    ]
    payloads: list[dict[str, object]] = []

    def transport(payload, api_key):
        del api_key
        payloads.append(payload)
        if len(payloads) == 1:
            return {"output_text": "", "output": original_output}
        return {"output_text": "done", "output": []}

    gateway = OpenAIResponsesGateway(api_key="test", transport=transport)
    first = gateway.generate([{"role": "user", "content": "start"}], [])
    tool_calls = [
        {
            "type": "function",
            "id": call.id,
            "function": {"name": call.name, "arguments": json.dumps(call.args, separators=(",", ":"))},
        }
        for call in first.tool_calls
    ]
    gateway.generate(
        [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": first.content,
                "tool_calls": tool_calls,
                "provider_turn_state": {"provider": first.provider_turn_state.provider, "payload": dict(first.provider_turn_state.payload)},
            },
            {"role": "tool", "tool_call_id": "c1", "content": "a"},
            {"role": "tool", "tool_call_id": "c2", "content": "b"},
        ],
        [],
    )

    assert payloads[1]["input"][1:5] == original_output


def test_provider_request_options_reach_each_gateway_final_payload() -> None:
    captured: dict[str, dict[str, object]] = {}

    def responses_transport(payload, api_key):
        del api_key
        captured["responses"] = payload
        return {"output_text": "ok", "output": []}

    def chat_transport(payload, api_key):
        del api_key
        captured["chat"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    def anthropic_transport(payload, api_key, endpoint):
        del api_key, endpoint
        captured["anthropic"] = payload
        return {"content": [{"type": "text", "text": "ok"}]}

    def google_transport(payload, api_key, endpoint):
        del api_key, endpoint
        captured["google"] = payload
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    OpenAIResponsesGateway(
        api_key="test",
        transport=responses_transport,
        request_config=_request_config({"reasoning": {"effort": "high"}, "max_output_tokens": 12000}),
    ).generate([{"role": "user", "content": "hi"}], [])
    OpenAIChatCompletionsGateway(
        api_key="test",
        transport=chat_transport,
        request_config=_request_config({"temperature": 0.3, "max_tokens": 2048}),
    ).generate([{"role": "user", "content": "hi"}], [])
    AnthropicMessagesGateway(
        api_key="test",
        transport=anthropic_transport,
        request_config=_request_config({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}}),
    ).generate([{"role": "user", "content": "hi"}], [])
    GoogleGeminiGateway(
        api_key="test",
        transport=google_transport,
        request_config=_request_config({"generationConfig": {"temperature": 0.4, "thinkingConfig": {"thinkingBudget": 1024}}}),
    ).generate([{"role": "user", "content": "hi"}], [])

    assert captured["responses"]["reasoning"] == {"effort": "high"}
    assert captured["responses"]["max_output_tokens"] == 12000
    assert captured["chat"]["temperature"] == 0.3
    assert captured["chat"]["max_tokens"] == 2048
    assert captured["anthropic"]["max_tokens"] == 8192
    assert captured["anthropic"]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert captured["google"]["generationConfig"] == {
        "temperature": 0.4,
        "thinkingConfig": {"thinkingBudget": 1024},
    }


def test_google_gateway_hides_thought_text_and_replays_signed_parts_in_order() -> None:
    original_content = {
        "role": "model",
        "parts": [
            {"text": "private reasoning", "thought": True},
            {
                "functionCall": {"name": "read", "args": {"path": "a"}},
                "thoughtSignature": "signed-1",
            },
            {"text": "visible"},
        ],
    }
    payloads: list[dict[str, object]] = []

    def transport(payload, api_key, endpoint):
        del api_key, endpoint
        payloads.append(payload)
        if len(payloads) == 1:
            return {"candidates": [{"content": original_content}]}
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}]}

    gateway = GoogleGeminiGateway(api_key="test", transport=transport)
    first = gateway.generate([{"role": "user", "content": "start"}], [])
    assert first.content == "visible"
    gateway.generate(
        [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": first.content,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a"}'},
                    }
                ],
                "provider_turn_state": {"provider": first.provider_turn_state.provider, "payload": dict(first.provider_turn_state.payload)},
            },
            {"role": "tool", "name": "read", "content": "a"},
        ],
        [],
    )

    assert payloads[1]["contents"][1] == original_content


def test_parameter_4xx_is_exposed_without_retrying_or_removing_options() -> None:
    attempts: list[dict[str, object]] = []

    def transport(payload, api_key):
        del api_key
        attempts.append(payload)
        raise ModelCallError(
            "unsupported reasoning value",
            details=ModelFailureDetails(
                category="invalid_request",
                status_code=400,
                retryable=False,
            ),
        )

    gateway = OpenAIResponsesGateway(
        api_key="test",
        transport=transport,
        request_config=_request_config({"reasoning": {"effort": "extreme"}}),
        retry_controller=_retry_controller(),
    )

    with pytest.raises(ModelCallError, match="unsupported reasoning value"):
        gateway.generate([{"role": "user", "content": "hi"}], [])

    assert len(attempts) == 1
    assert attempts[0]["reasoning"] == {"effort": "extreme"}


@pytest.mark.parametrize(
    "gateway_type, response, transport_arity",
    [
        (OpenAIChatCompletionsGateway, {"choices": [{"message": {"content": "ok"}}]}, 2),
        (AnthropicMessagesGateway, {"content": [{"type": "text", "text": "ok"}]}, 3),
        (GoogleGeminiGateway, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}, 3),
    ],
)
def test_provider_gateways_retry_a_retryable_model_failure(
    gateway_type,
    response,
    transport_arity: int,
) -> None:
    attempts = 0

    def retrying_transport(*args):
        nonlocal attempts
        assert len(args) == transport_arity
        attempts += 1
        if attempts == 1:
            raise ModelCallError(
                "temporary",
                details=ModelFailureDetails(category="server", status_code=503, retryable=True),
            )
        return response

    gateway = gateway_type(
        api_key="key",
        model="model",
        transport=retrying_transport,
        retry_controller=_retry_controller(),
    )

    assert gateway.generate([], []).content == "ok"
    assert attempts == 2


def test_openai_chat_rejects_embedded_tool_markup_without_structured_calls() -> None:
    gateway = OpenAIChatCompletionsGateway(
        api_key="key",
        model="model",
        transport=lambda payload, api_key: {
            "choices": [
                {
                    "message": {
                        "content": "<｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name=\"file_read\"></｜｜DSML｜｜tool_calls>",
                        "tool_calls": [],
                    },
                },
            ],
        },
    )

    with pytest.raises(ModelCallError, match="embedded tool markup") as raised:
        gateway.generate([], [])

    assert raised.value.details is not None
    assert raised.value.details.category == "protocol"
    assert raised.value.details.retryable is False


def test_stream_delta_then_failure_is_not_retried() -> None:
    attempts = 0
    deltas: list[str] = []

    def interrupted_stream_transport(payload, api_key, sink):
        nonlocal attempts
        attempts += 1
        sink("partial")
        raise ModelCallError(
            "reset",
            details=ModelFailureDetails(category="network", retryable=True),
        )

    gateway = OpenAIResponsesGateway(
        api_key="key",
        model="model",
        stream_transport=interrupted_stream_transport,
        retry_controller=_retry_controller(),
    )

    with pytest.raises(ModelCallError) as raised:
        gateway.generate([], [], event_sink=deltas.append)

    assert attempts == 1
    assert deltas == ["partial"]
    assert raised.value.details is not None
    assert raised.value.details.category == "stream_interrupted"


def test_stream_interruption_reports_final_attempt_for_audit() -> None:
    def interrupted_stream_transport(payload, api_key, sink):
        del payload, api_key
        sink("partial")
        raise ModelCallError(
            "reset",
            details=ModelFailureDetails(category="network", retryable=True),
        )

    exhausted = []
    gateway = OpenAIResponsesGateway(
        api_key="key",
        model="model",
        stream_transport=interrupted_stream_transport,
        retry_controller=_retry_controller(),
    )

    with pytest.raises(ModelCallError):
        gateway.generate([], [], event_sink=lambda _: None, retry_exhausted_sink=lambda failure, attempt: exhausted.append((failure.category, attempt)))

    assert exhausted == [("stream_interrupted", 1)]


def test_gateway_propagates_retry_controller_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    gateway = OpenAIResponsesGateway(
        api_key="key",
        model="model",
        transport=lambda payload, api_key: {"output_text": "unexpected", "output": []},
        retry_controller=_retry_controller(),
    )

    with pytest.raises(RunCancelled):
        gateway.generate([], [], cancellation_token=token)


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


def test_gateway_registry_maps_catalog_providers_to_supported_gateways() -> None:
    cases = [
        ("anthropic", "Anthropic", None, "@ai-sdk/anthropic", "anthropic"),
        ("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", None, "openai-chat"),
        (
            "requesty",
            "Requesty",
            "https://router.requesty.ai/v1",
            "@ai-sdk/openai-compatible",
            "openai-chat",
        ),
        ("deepseek", "DeepSeek", "https://api.deepseek.com", "@ai-sdk/openai-compatible", "openai-chat"),
        (
            "lmstudio",
            "LMStudio",
            "http://127.0.0.1:1234/v1",
            "@ai-sdk/openai-compatible",
            "openai-chat",
        ),
        (
            "ollama-cloud",
            "Ollama Cloud",
            "https://ollama.com/v1",
            "@ai-sdk/openai-compatible",
            "openai-chat",
        ),
        (
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            "@openrouter/ai-sdk-provider",
            "openai-chat",
        ),
        (
            "google",
            "Google",
            "https://generativelanguage.googleapis.com/v1beta",
            "@ai-sdk/google",
            "google",
        ),
    ]

    for provider_id, name, api_base_url, provider_package, expected_gateway in cases:
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
        assert capability.gateway_provider == expected_gateway


def test_gateway_registry_builds_supported_provider_gateways() -> None:
    cases = [
        ("openai-chat", "https://openrouter.ai/api/v1", OpenAIChatCompletionsGateway),
        ("anthropic", "https://api.anthropic.com", AnthropicMessagesGateway),
        ("google", "https://generativelanguage.googleapis.com/v1beta", GoogleGeminiGateway),
    ]

    for provider, base_url, expected_type in cases:
        resolved = _resolved(provider, base_url)
        gateway = gateway_from_resolved(resolved)
        # gateway_from_resolved 外包审计层；底层仍是对应 provider adapter。
        inner = getattr(gateway, "_gateway", gateway)
        assert isinstance(inner, expected_type)
        metadata = gateway.metadata()
        assert metadata.profile_name == resolved.ref.connection_id
        assert metadata.request_config is not None
        assert metadata.request_config["connection_id"] == resolved.ref.connection_id
        assert "settings_digest" in metadata.request_config
        # 审计包装层必须转发重试控制器，避免 transcript max_attempts 变成 None。
        assert getattr(gateway, "_retry_controller", None) is getattr(inner, "_retry_controller", None)
        assert getattr(gateway._retry_controller.policy, "max_attempts", None) is not None


def test_responses_options_do_not_leak_into_chat_protocol_fallback() -> None:
    profile = _resolved(
        "openai",
        "https://api.openai.com/v1",
        settings=_request_config({"reasoning": {"effort": "high"}, "max_output_tokens": 12000}),
    )
    captured: dict[str, object] = {}

    def chat_transport(payload, api_key):
        del api_key
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = gateway_from_route(profile)
    try:
        assert gateway._primary.metadata().request_config["configured"] is True
        assert gateway._primary_chat.metadata().request_config["configured"] is False
        assert gateway._primary_chat.metadata().request_config["options_summary"] == {}
        # 真实协议 fallback：即使 invocation 带 primary Responses options，Chat payload 也不得包含它们。
        primary = gateway._primary
        if hasattr(primary, "_gateway"):
            primary = primary._gateway
        primary_chat = gateway._primary_chat
        if hasattr(primary_chat, "_gateway"):
            primary_chat = primary_chat._gateway
        primary._transport = lambda payload, api_key, **_: (_ for _ in ()).throw(
            ModelCallError(
                "not implemented",
                details=ModelFailureDetails(category="client", status_code=501),
            )
        )
        primary_chat._transport = chat_transport
        response = gateway.generate(
            ModelInvocation(
                [{"role": "user", "content": "hi"}],
                [],
                profile.settings,
            )
        )
        assert response.content == "ok"
        payload = captured["payload"]
        assert "reasoning" not in payload
        assert "max_output_tokens" not in payload
        assert payload["model"] == "test-model"
    finally:
        gateway.close()


def test_model_fallback_uses_target_settings_only() -> None:
    primary = _resolved(
        "openai-chat",
        "https://primary.example/v1",
        settings=_request_config({"temperature": 0.9, "max_tokens": 111}),
    )
    fallback = _resolved(
        "openai-chat",
        "https://fallback.example/v1",
        settings=_request_config({"temperature": 0.1, "max_tokens": 222}),
    )
    fallback = ResolvedModel(
        ref=ModelRef("fallback-main", "fallback-model"),
        provider=fallback.provider,
        base_url=fallback.base_url,
        runtime_kind=fallback.runtime_kind,
        settings=fallback.settings,
        credential=fallback.credential,
    )
    captured: dict[str, object] = {}

    def fallback_transport(payload, api_key):
        del api_key
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "fallback-ok"}}]}

    gateway = gateway_from_route(
        primary,
        fallback_model=fallback,
        primary_capabilities=ModelCapabilities(context_window_tokens=200_000),
        fallback_capabilities=ModelCapabilities(
            context_window_tokens=64_000,
            input_window_tokens=64_000,
        ),
    )
    try:
        primary_gateway = gateway._primary
        fallback_gateway = gateway._fallback
        while hasattr(primary_gateway, "_gateway"):
            primary_gateway = primary_gateway._gateway
        while hasattr(fallback_gateway, "_gateway"):
            fallback_gateway = fallback_gateway._gateway
        primary_gateway._transport = lambda payload, api_key, **_: (_ for _ in ()).throw(
            ModelCallError(
                "network down",
                details=ModelFailureDetails(category="network", retryable=True),
            )
        )
        fallback_gateway._transport = fallback_transport
        response = gateway.generate(
            ModelInvocation(
                [{"role": "user", "content": "hi"}],
                [],
                primary.settings,
            )
        )
        assert response.content == "fallback-ok"
        assert gateway.capabilities().context_window_tokens == 64_000
        assert gateway.metadata().context_window_tokens == 64_000
        payload = captured["payload"]
        assert payload["temperature"] == 0.1
        assert payload["max_tokens"] == 222
        assert payload["model"] == "fallback-model"
        assert "max_context_tokens" not in payload
    finally:
        gateway.close()


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


def test_anthropic_gateway_replays_interleaved_thinking_and_tool_blocks_in_original_order() -> None:
    original_blocks = [
        {"type": "thinking", "thinking": "opaque-1", "signature": "sig-1"},
        {"type": "tool_use", "id": "tool-1", "name": "read", "input": {"path": "a"}},
        {"type": "redacted_thinking", "data": "opaque-2"},
        {"type": "tool_use", "id": "tool-2", "name": "read", "input": {"path": "b"}},
    ]
    payloads: list[dict[str, object]] = []

    def transport(payload, api_key, endpoint):
        del api_key, endpoint
        payloads.append(payload)
        if len(payloads) == 1:
            return {"content": original_blocks}
        return {"content": [{"type": "text", "text": "done"}]}

    gateway = AnthropicMessagesGateway(api_key="test", transport=transport)
    first = gateway.generate([{"role": "user", "content": "start"}], [])
    assistant = {
        "role": "assistant",
        "content": first.content,
        "tool_calls": [
            {
                "type": "function",
                "id": call.id,
                "function": {"name": call.name, "arguments": json.dumps(call.args)},
            }
            for call in first.tool_calls
        ],
        "provider_turn_state": {"provider": first.provider_turn_state.provider, "payload": dict(first.provider_turn_state.payload)},
    }
    gateway.generate(
        [
            {"role": "user", "content": "start"},
            assistant,
            {"role": "tool", "tool_call_id": "tool-1", "content": "a"},
            {"role": "tool", "tool_call_id": "tool-2", "content": "b"},
        ],
        [],
    )

    assert payloads[1]["messages"][1]["content"] == original_blocks


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
            "usage": {
                "input_tokens": 12,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 5,
                "output_tokens": 4,
            },
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
        context_input_tokens=20,
    )


def test_route_overlays_catalog_input_windows_on_primary_and_fallback() -> None:
    primary = _resolved("openai-chat", "https://primary.example/v1")
    fallback = _resolved("openai-chat", "https://fallback.example/v1")

    gateway = gateway_from_route(
        primary,
        fallback_model=fallback,
        primary_capabilities=ModelCapabilities(
            context_window_tokens=200_000,
            input_window_tokens=180_000,
        ),
        fallback_capabilities=ModelCapabilities(context_window_tokens=128_000),
    )
    try:
        assert gateway._primary.capabilities().input_window_tokens == 180_000
        assert gateway._primary.capabilities().context_window_tokens == 200_000
        assert gateway._fallback.capabilities().context_window_tokens == 128_000
    finally:
        gateway.close()


def test_gateway_metadata_exposes_effective_context_window() -> None:
    gateway = gateway_from_resolved(
        _resolved("openai-chat", "https://example.test/v1"),
        capabilities=ModelCapabilities(
            context_window_tokens=400_000,
            input_window_tokens=400_000,
        ),
    )
    try:
        assert gateway.metadata().context_window_tokens == 400_000
    finally:
        gateway.close()


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
                "version": 4,
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
            },
        ),
        encoding="utf-8",
    )

    profile = resolve_model(
        ModelRef("deepseek", "deepseek-v4-pro"),
        snapshot=_snapshot(config_path),
        environ={"DEEPSEEK_API_KEY": "secret-key"},
    )

    assert profile.provider == "openai-chat"
    assert profile.base_url == "https://api.deepseek.com"
    assert profile.ref.model == "deepseek-v4-pro"
    assert profile.credential.api_key_env == "DEEPSEEK_API_KEY"
    assert profile.credential.api_key == "secret-key"


def test_model_selection_loads_api_key_from_keyring_when_env_missing(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 4,
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
            },
        ),
        encoding="utf-8",
    )

    profile = resolve_model(
        ModelRef("deepseek", "deepseek-v4-pro"),
        snapshot=_snapshot(config_path),
        environ={},
        credential_store=FakeCredentialStore({"connection:deepseek": "keyring-secret"}),
    )

    assert profile.credential.api_key == "keyring-secret"
    assert profile.credential.source == "keyring"
    assert profile.credential.source_used == "keyring"


def test_provider_connections_can_be_listed_without_secrets(tmp_path: Path) -> None:
    connection = ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="keyring",
        )
    _save_connection(tmp_path, connection)

    records = _snapshot(tmp_path / "providers.json").list_connections()

    assert [record.name for record in records] == ["router"]
    assert records[0].provider_id == "openrouter"
    assert "secret" not in records[0].to_dict()


def test_provider_connection_credential_status_for_named_connection(tmp_path: Path) -> None:
    connection = ProviderConnectionRecord(
            id="router",
            name="router",
            provider_id="openrouter",
            provider_name="OpenRouter",
            gateway_provider="openai-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            credential_source="keyring",
        )
    _save_connection(tmp_path, connection)

    status = provider_connection_credential_status(
        connection,
        config_dir=tmp_path,
        credential_store=FakeCredentialStore({"connection:router": "sk-test-secret"}),
    )

    assert status.api_key_available is True
    assert status.credential_source_used == "keyring"
    assert "sk-test-secret" not in repr(status)


def test_model_selection_loads_api_key_from_explicit_insecure_file(tmp_path: Path) -> None:
    config_dir = tmp_path / ".haagent"
    config_path = config_dir / "providers.json"
    connection = ProviderConnectionRecord(
            id="deepseek",
            name="deepseek",
            provider_id="deepseek",
            provider_name="DeepSeek",
            gateway_provider="openai-chat",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            credential_source="insecure_file",
        )
    store = ModelConfigStore(config_path)
    store.save_connection(connection, expected_digest=store.load().digest)
    save_connection_api_key(
        connection,
        "plain-secret",
        config_dir=config_dir,
    )

    profile = resolve_model(
        ModelRef("deepseek", "deepseek-v4-pro"),
        snapshot=_snapshot(config_path),
        environ={},
        credential_store=FakeCredentialStore({}),
    )

    assert profile.credential.api_key == "plain-secret"
    assert profile.credential.source == "insecure_file"
    assert profile.credential.source_used == "insecure_file"


def test_model_selection_missing_connection_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 4,
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
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="provider connection not found: deepseek"):
        resolve_model(
            ModelRef("deepseek", "deepseek-v4-pro"),
            snapshot=_snapshot(config_path),
            environ={"OPENAI_API_KEY": "key"},
        )


def test_model_selection_missing_api_key_fails_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / ".haagent" / "providers.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "version": 4,
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
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProviderProfileError, match="API key is not available"):
        resolve_model(
            ModelRef("deepseek", "deepseek-v4-pro"),
            snapshot=_snapshot(config_path),
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

    assert gateway.metadata().endpoint == DEFAULT_RESPONSES_ENDPOINT


def test_openai_gateway_reads_base_url_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://compatible.example/v1")

    gateway = OpenAIResponsesGateway(
        api_key="test-key",
        transport=lambda payload, api_key: {"output_text": "ok"},
    )

    assert gateway.metadata().endpoint == "https://compatible.example/v1/responses"


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

    assert gateway.metadata().endpoint == expected_endpoint


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

    assert gateway.metadata().endpoint == expected_endpoint


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
        termination="tool_calls",
    )
    assert captured["payload"] == {
        "model": "chat-test",
        "messages": [{"role": "user", "content": "context with tools"}],
        "parallel_tool_calls": True,
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
        "parallel_tool_calls": True,
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
        termination="tool_calls",
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

    with pytest.raises(ModelCallError, match="model request failed"):
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
    from haagent.models.types import ModelResponse
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
    from haagent.models.types import ModelResponse
    from haagent.context.compression.full import FullCompactEligibility
    from haagent.runtime.orchestration.orchestrator import RunOrchestrator

    class TwoStepGateway:
        provider_name = "two-step"

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, invocation, **kwargs):
            self.calls.append({"messages": invocation.messages, "tool_schemas": invocation.tool_schemas})
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

        def generate(self, invocation, **kwargs):
            self._turn += 1
            self.model_inputs.append("\n".join(str(m.get("content", "")) for m in invocation.messages))
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
    from haagent.runtime.events.bus import bus_event_to_dict

    gateway = LargeToolResultGateway()
    runtime_events: list[object] = []

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
        bus_event_to_dict(event)
        for event in runtime_events
        if bus_event_to_dict(event).get("event_type") == "compression_diagnostic"
    ]
    assert len(microcompact_events) == 1
    assert "event" not in microcompact_events[0]
    assert microcompact_events[0]["stage"] == "historical_tool_message"
    assert microcompact_events[0]["reason"] == "long_text_result"
