"""
tests/unit/models/test_model_catalog.py - 模型目录层测试

验证 Models.dev 目录解析、缓存回退和显式失败。
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from haagent.models import catalog as catalog_module
from haagent.models.catalog import ModelCatalogError, fetch_model_catalog, search_catalog


def test_models_dev_catalog_parses_provider_model_and_cache(tmp_path: Path) -> None:
    payload = {
        "requesty": {
            "id": "requesty",
            "name": "Requesty",
            "env": ["REQUESTY_API_KEY"],
            "api": "https://router.requesty.ai/v1",
            "npm": "@ai-sdk/openai-compatible",
            "doc": "https://requesty.ai/models",
            "models": {
                "openai/gpt-5.2-chat": {
                    "id": "openai/gpt-5.2-chat",
                    "name": "GPT 5.2 Chat",
                    "family": "gpt",
                    "tool_call": True,
                    "reasoning": True,
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "limit": {"context": 128000, "output": 16000},
                    "cost": {"input": 1.25, "output": 10},
                    "release_date": "2026-01-01",
                    "last_updated": "2026-01-10",
                }
            },
        }
    }

    def transport() -> dict[str, object]:
        return payload

    result = fetch_model_catalog(
        cache_path=tmp_path / "models_catalog_cache.json",
        transport=transport,
    )

    provider = result.providers[0]
    assert result.used_cache is False
    assert provider.id == "requesty"
    assert provider.api_base_url == "https://router.requesty.ai/v1"
    assert provider.env_names == ["REQUESTY_API_KEY"]
    assert provider.provider_package == "@ai-sdk/openai-compatible"
    assert provider.models[0].id == "openai/gpt-5.2-chat"
    assert provider.models[0].supports_tool_call is True

    matches = search_catalog(result, "gpt 5.2")
    assert [item.id for item in matches] == ["requesty"]


def test_default_models_dev_transport_sends_json_request_headers(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b'{"openai":{"id":"openai","name":"OpenAI","env":["OPENAI_API_KEY"],"models":{}}}'

    def fake_urlopen(request, timeout: int):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(catalog_module, "urlopen", fake_urlopen)

    result = catalog_module._default_transport()

    request = captured["request"]
    assert captured["timeout"] == 30
    assert request.full_url == "https://models.dev/api.json"
    assert request.get_header("User-agent").startswith("HaAgent/")
    assert request.get_header("Accept") == "application/json"
    assert result["openai"]["name"] == "OpenAI"


def test_model_catalog_uses_cache_when_refresh_fails(tmp_path: Path) -> None:
    cache_path = tmp_path / "models_catalog_cache.json"

    def first_transport() -> dict[str, object]:
        return {
            "openrouter": {
                "id": "openrouter",
                "name": "OpenRouter",
                "env": ["OPENROUTER_API_KEY"],
                "api": "https://openrouter.ai/api/v1",
                "npm": "@openrouter/ai-sdk-provider",
                "models": {"anthropic/claude-sonnet": {"id": "anthropic/claude-sonnet"}},
            }
        }

    fetch_model_catalog(cache_path=cache_path, transport=first_transport)

    def failing_transport() -> dict[str, object]:
        raise OSError("network down")

    result = fetch_model_catalog(cache_path=cache_path, transport=failing_transport)

    assert result.used_cache is True
    assert result.error == "network down"
    assert result.providers[0].id == "openrouter"


def test_model_catalog_uses_fresh_cache_without_refreshing(tmp_path: Path) -> None:
    cache_path = tmp_path / "models_catalog_cache.json"
    calls = 0

    def first_transport() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "api": "https://router.requesty.ai/v1",
                "models": {"openai/gpt-5.2-chat": {"id": "openai/gpt-5.2-chat"}},
            }
        }

    fetch_model_catalog(cache_path=cache_path, transport=first_transport)

    def forbidden_transport() -> dict[str, object]:
        raise AssertionError("fresh cache should avoid network refresh")

    result = fetch_model_catalog(
        cache_path=cache_path,
        transport=forbidden_transport,
        max_cache_age=timedelta(hours=24),
    )

    assert calls == 1
    assert result.used_cache is True
    assert result.providers[0].id == "requesty"


def test_model_catalog_refreshes_when_cache_is_expired(tmp_path: Path) -> None:
    cache_path = tmp_path / "models_catalog_cache.json"

    def old_transport() -> dict[str, object]:
        return {
            "old": {
                "id": "old",
                "name": "Old",
                "env": ["OLD_API_KEY"],
                "api": "https://old.example/v1",
                "models": {"old-model": {"id": "old-model"}},
            }
        }

    fetch_model_catalog(cache_path=cache_path, transport=old_transport)
    stale_at = datetime.now(tz=UTC) - timedelta(hours=25)
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["fetched_at"] = stale_at.isoformat()
    cache_path.write_text(json.dumps(raw), encoding="utf-8")

    def new_transport() -> dict[str, object]:
        return {
            "new": {
                "id": "new",
                "name": "New",
                "env": ["NEW_API_KEY"],
                "api": "https://new.example/v1",
                "models": {"new-model": {"id": "new-model"}},
            }
        }

    result = fetch_model_catalog(
        cache_path=cache_path,
        transport=new_transport,
        max_cache_age=timedelta(hours=24),
    )

    assert result.used_cache is False
    assert result.providers[0].id == "new"


def test_model_catalog_force_refresh_ignores_fresh_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "models_catalog_cache.json"

    fetch_model_catalog(
        cache_path=cache_path,
        transport=lambda: {
            "cached": {
                "id": "cached",
                "name": "Cached",
                "env": ["CACHED_API_KEY"],
                "api": "https://cached.example/v1",
                "models": {"cached-model": {"id": "cached-model"}},
            }
        },
    )

    result = fetch_model_catalog(
        cache_path=cache_path,
        transport=lambda: {
            "fresh": {
                "id": "fresh",
                "name": "Fresh",
                "env": ["FRESH_API_KEY"],
                "api": "https://fresh.example/v1",
                "models": {"fresh-model": {"id": "fresh-model"}},
            }
        },
        max_cache_age=timedelta(hours=24),
        force_refresh=True,
    )

    assert result.used_cache is False
    assert result.providers[0].id == "fresh"


def test_model_catalog_fails_explicitly_without_cache(tmp_path: Path) -> None:
    def failing_transport() -> dict[str, object]:
        raise OSError("network down")

    with pytest.raises(ModelCatalogError, match="network down"):
        fetch_model_catalog(
            cache_path=tmp_path / "missing_cache.json",
            transport=failing_transport,
        )


def test_model_catalog_rejects_dirty_model_entries(tmp_path: Path) -> None:
    def transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "api": "https://router.requesty.ai/v1",
                "npm": "@ai-sdk/openai-compatible",
                "doc": "https://requesty.ai/models",
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "GPT 5.2 Chat",
                    },
                    "broken-model": "not-an-object",
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="model entry"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,
        )


def test_model_catalog_rejects_non_mapping_transport_result(tmp_path: Path) -> None:
    def transport() -> object:
        return ["not", "a", "mapping"]

    with pytest.raises(ModelCatalogError, match="transport result must be a JSON object"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,  # type: ignore[arg-type]
        )


def test_model_catalog_rejects_dirty_model_field_types(tmp_path: Path) -> None:
    def transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "GPT 5.2 Chat",
                        "tool_call": "yes",
                        "modalities": [],
                        "limit": {},
                        "cost": {},
                    },
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="model field must be boolean"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,
        )


def test_model_catalog_rejects_dirty_provider_env_entries(tmp_path: Path) -> None:
    def transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY", 42],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "GPT 5.2 Chat",
                    },
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="provider env entry must be a string"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,
        )


def test_model_catalog_rejects_empty_provider_id(tmp_path: Path) -> None:
    def transport() -> dict[str, object]:
        return {
            "": {
                "id": "",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "GPT 5.2 Chat",
                    },
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="provider id must be a non-empty string"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,
        )


@pytest.mark.parametrize("provider_name", ["", 42, None])
def test_model_catalog_rejects_invalid_provider_name(
    tmp_path: Path,
    provider_name: object,
) -> None:
    def transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": provider_name,
                "env": ["REQUESTY_API_KEY"],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "GPT 5.2 Chat",
                    },
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="provider field must be a non-empty string: name"):
        fetch_model_catalog(
            cache_path=tmp_path / "models_catalog_cache.json",
            transport=transport,
        )


def test_model_catalog_rejects_invalid_model_identity_fields(tmp_path: Path) -> None:
    def bad_id_transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "",
                        "name": "GPT 5.2 Chat",
                    },
                },
            }
        }

    def bad_name_transport() -> dict[str, object]:
        return {
            "requesty": {
                "id": "requesty",
                "name": "Requesty",
                "env": ["REQUESTY_API_KEY"],
                "models": {
                    "openai/gpt-5.2-chat": {
                        "id": "openai/gpt-5.2-chat",
                        "name": "",
                    },
                },
            }
        }

    with pytest.raises(ModelCatalogError, match="model field must be a non-empty string: id"):
        fetch_model_catalog(
            cache_path=tmp_path / "bad_id_cache.json",
            transport=bad_id_transport,
        )

    with pytest.raises(ModelCatalogError, match="model field must be a non-empty string: name"):
        fetch_model_catalog(
            cache_path=tmp_path / "bad_name_cache.json",
            transport=bad_name_transport,
        )
