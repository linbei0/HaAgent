"""
tests/unit/models/test_local_runtime.py - 本地模型运行时发现测试

覆盖 Ollama 与 LM Studio 的模型枚举、能力映射和显式失败状态。
"""

import httpx

from haagent.models.local_runtime import discover_lm_studio, discover_ollama


def test_discover_ollama_maps_model_capabilities_and_context_window() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "qwen3:8b",
                            "model": "qwen3:8b",
                            "details": {"family": "qwen3"},
                        },
                    ],
                },
            )
        if request.method == "POST" and request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools", "vision", "thinking"],
                    "model_info": {"qwen3.context_length": 32_768},
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    result = discover_ollama(transport=httpx.MockTransport(handler))

    assert result.status == "available"
    assert result.runtime_kind == "ollama"
    assert result.base_url == "http://127.0.0.1:11434/v1"
    assert len(result.models) == 1
    model = result.models[0]
    assert model.id == "qwen3:8b"
    assert model.loaded is False
    assert model.capabilities.tools == "supported"
    assert model.capabilities.tools_mode == "native"
    assert model.capabilities.vision == "supported"
    assert model.capabilities.reasoning == "supported"
    assert model.capabilities.context_window_tokens == 32_768
    assert model.capabilities.protocols == frozenset({"responses", "chat_completions"})


def test_discover_lm_studio_filters_embeddings_and_prefers_loaded_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v1/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "type": "llm",
                        "key": "local/qwen",
                        "display_name": "Local Qwen",
                        "max_context_length": 131_072,
                        "loaded_instances": [
                            {"id": "local/qwen", "config": {"context_length": 8_192}},
                        ],
                        "capabilities": {
                            "vision": False,
                            "trained_for_tool_use": False,
                            "reasoning": {"allowed_options": ["on", "off"]},
                        },
                    },
                    {"type": "embedding", "key": "local/embed"},
                ],
            },
        )

    result = discover_lm_studio(transport=httpx.MockTransport(handler))

    assert result.status == "available"
    assert result.runtime_kind == "lm_studio"
    assert result.base_url == "http://127.0.0.1:1234/v1"
    assert [model.id for model in result.models] == ["local/qwen"]
    model = result.models[0]
    assert model.loaded is True
    assert model.capabilities.context_window_tokens == 8_192
    assert model.capabilities.tools == "supported"
    assert model.capabilities.tools_mode == "compat"
    assert model.capabilities.vision == "unsupported"
    assert model.capabilities.reasoning == "supported"


def test_discovery_reports_unauthorized_without_exposing_response_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "secret token leaked"}),
    )

    result = discover_lm_studio(transport=transport)

    assert result.status == "unauthorized"
    assert result.models == ()
    assert result.reason == "LM Studio discovery requires authentication"
    assert "secret token leaked" not in result.reason


def test_discovery_reports_unreachable_for_network_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = discover_ollama(transport=httpx.MockTransport(handler))

    assert result.status == "unreachable"
    assert result.models == ()
    assert result.reason == "Ollama is not reachable"


def test_discovery_reports_invalid_response_for_malformed_json() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"not-json"),
    )

    result = discover_lm_studio(transport=transport)

    assert result.status == "invalid_response"
    assert result.models == ()
    assert result.reason == "LM Studio returned an invalid discovery response"
